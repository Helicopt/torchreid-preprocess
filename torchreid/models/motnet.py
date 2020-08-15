"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import division, absolute_import
import re
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import model_zoo
import torchvision.models as tvm


class MOTNet(nn.Module):

    def __init__(self, backbone_config={}, groups=16, feat_dim=16, **kwargs):
        super(MOTNet, self).__init__()
        self.backbone = tvm.resnet18(**backbone_config)

        self._classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.a_fc = nn.Linear(1000, 1024)
        self.groups = groups
        self.feat_dim = feat_dim
        self.wave_length = 1000.0

    def init_weights(self):
        for fc in self._classifier:
            if isinstance(fc, nn.Linear):
                nn.init.kaiming_normal_(fc.weight, a=1)
                nn.init.constant_(fc.bias, 0)
        for fc in [self.a_fc]:
            nn.init.kaiming_normal_(fc.weight, a=1)
            nn.init.constant_(fc.bias, 0)

    def pos_vec(self, dts):
        n, m, c = dts.shape
        dts = dts.reshape(n * m, c)
        xmin, ymin, xmax, ymax = torch.chunk(dts, 4, dim=1)
        bbox_width_ref = xmax - xmin
        bbox_height_ref = ymax - ymin
        center_x_ref = 0.5 * (xmin+xmax)
        center_y_ref = 0.5 * (ymin+ymax)

        delta_x = center_x_ref - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width_ref
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y_ref - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height_ref
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width_ref / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height_ref / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack(
            [delta_x, delta_y, delta_width, delta_height], dim=2
        )

        feat_range = torch.arange(
            0, self.feat_dim / 8, device=position_matrix.device
        )
        dim_mat = torch.full(
            (len(feat_range), ), self.wave_length, device=position_matrix.device
        ).pow(8.0 / self.feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_matrix.shape, -1)

        position_mat = position_matrix.unsqueeze(3).expand(
            -1, -1, -1, dim_mat.shape[3]
        )
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()


        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(
            embedding.shape[0], embedding.shape[1],
            embedding.shape[2] * embedding.shape[3]
        )
        embedding = embedding.permute(2, 0, 1)
        out = []
        for i in range(n):
            out.append(embedding[:, i*m:i*m+m, i*m:i*m+m].reshape(-1))
        out = torch.stack(out)

        return out

    def appr_vec(self, v):
        n, m, c = v.shape
        out = []
        for i in range(n):
            v_ = v[i].reshape(m, self.groups, -1)
            v_ = v_.permute(1, 0, 2)
            mat = torch.bmm(v_, v_.transpose(1, 2))
            out.append(mat.reshape(-1))
        # v = v.reshape(n * m, self.groups, -1)
        # v = v.permute(1, 0, 2)
        # mat = torch.bmm(v, v.transpose(1, 2))
        # for i in range(n):
        #     out.append(mat[:, i*m:i*m+m, i*m:i*m+m].reshape(-1))
        out = torch.stack(out)
        return out

    def classifier(self, x, softmax=False):
        out = self._classifier(x)
        if softmax:
            return F.softmax(out, dim=1)
        else:
            return out

    def features(self, im_tiles):
        return self.a_fc(F.relu(self.backbone(im_tiles)))

    def forward(self, data, raw=True):
        if raw:
            # print(data['cur_im'].shape)
            # print(data['ref_im'].shape)
            im_tiles = torch.cat([data['cur_im'], data['ref_im']], dim=1)
            n, m, c, h, w = im_tiles.shape
            im_tiles = im_tiles.reshape(n * m, c, h, w)
            feats = self.features(im_tiles)
            feats = feats.reshape(n, m, -1)
        else:
            feats = torch.cat([data['cur_im'], data['ref_im']], dim=1)
        # print(feats.shape)
        # import time
        # st = time.time()
        appr_features = self.appr_vec(feats)
        # en = time.time()
        # print(en - st)
        dt_tiles = torch.cat(
            [data['cur_dt'].unsqueeze(1), data['ref_dt']], dim=1
        )
        pos_features = self.pos_vec(dt_tiles)
        # print(appr_features.shape, appr_features.dtype)
        # print(pos_features.shape, pos_features.dtype)
        features = torch.cat([appr_features, pos_features], dim=1)
        # print(features.shape)
        out = self.classifier(features, softmax=not self.training)
        return out

    def loss(self, out, targets):
        return F.cross_entropy(out, targets)
