from __future__ import division, print_function, absolute_import
import re
import glob
import json
import yaml
import os.path as osp

from torch.utils.data import Dataset
from senseTk.common import *
from senseTk.functions import LAP_Matching
import numpy as np
import torch

from torchreid.utils import read_image
from torchreid.data.transforms import build_transforms


class MOT16(Dataset):
    """MOT16 format tracking dataset.

    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_
    
    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'MOT16'

    # dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'

    def generate_paths(self):
        img_pattern = 'train/%s/img1/'
        img_pattern_test = 'test/%s/img1/'
        det_pattern = 'train/%s/det/det.txt'
        det_pattern_test = 'test/%s/det/det.txt'
        gt_pattern = 'train/%s/gt/gt.txt'
        self.train_im_dir = osp.join(self.dataset_dir, img_pattern)
        self.val_im_dir = osp.join(self.dataset_dir, img_pattern)
        self.test_im_dir = osp.join(self.dataset_dir, img_pattern_test)
        self.train_dt_dir = osp.join(self.dataset_dir, det_pattern)
        self.val_dt_dir = osp.join(self.dataset_dir, det_pattern)
        self.test_dt_dir = osp.join(self.dataset_dir, det_pattern_test)
        self.train_gt_dir = osp.join(self.dataset_dir, gt_pattern)
        self.val_gt_dir = osp.join(self.dataset_dir, gt_pattern)
        im_mappings = {
            'test': self.test_im_dir,
            'val': self.val_im_dir,
            'train': self.train_im_dir,
        }
        dt_mappings = {
            'test': self.test_dt_dir,
            'val': self.val_dt_dir,
            'train': self.train_dt_dir,
        }
        gt_mappings = {
            'val': self.val_gt_dir,
            'train': self.train_gt_dir,
        }

        def gen_pattern(mode, item, seq):
            if item == 'img':
                return im_mappings[mode] % seq
            elif item == 'det':
                return dt_mappings[mode] % seq
            elif item == 'gt':
                return gt_mappings[mode] % seq
            else:
                raise

        self.path_pattern = gen_pattern
        self.gt_formatter = 'fr.i id.i x1 y1 w h st.i la.i cf'
        self.gt_filter = lambda d: d.status == 1
        self.dt_formatter = None

    def __init__(
        self,
        root='',
        mode='train',
        test_in_train_format=False,
        meta_file='',
        step=30,
        h_len=3,
        height=256,
        width=128,
        loc_std=1,
        **kwargs
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.generate_paths()

        # required_files = [
        #     self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        # ]
        # self.check_before_run(required_files)

        self.test_mode = mode
        self.meta_file = meta_file
        self.test_in_train_format = test_in_train_format

        self.step = step
        self.h_len = h_len
        self.height = height
        self.width = width
        self.loc_std = loc_std
        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
        )

        assert self.test_mode in [
            'train', 'val', 'test'
        ], 'mode must be one of [train, val or test]'
        self.sequences = []
        self.data = self._process()

        self._cache_flag = False

    def __len__(self):
        return len(self.data)

    def _match_gt(self, dt, gt):
        for f in gt.frameRange():
            d = dt[f]
            g = gt[f]
            m, l, r = LAP_Matching(d, g, Det.iou)
            for i, j in m:
                if d[i].iou(g[j]) < 0.55:
                    continue
                tmp = d[i]
                dt.delete(tmp)
                tmp.uid = g[j].uid
                dt.append_data(tmp)

    def _process(self):
        if self.meta_file == '':
            self.meta_file = osp.join(
                self.dataset_dir, self.test_mode + '.list.txt'
            )
        data = []
        with open(self.meta_file) as fd:
            lines = fd.readlines()
        for one in lines:
            one = one.strip()
            seq_imdir = self.path_pattern(self.test_mode, 'img', one)
            vid = VideoClipReader(seq_imdir)
            # seq_dt = osp.join(dt_mappings[self.test_mode], one + '.txt')
            # seq_dt = self.path_pattern(self.test_mode, 'det', one)
            # dt = TrackSet(seq_dt, formatter='fr.i id.i x1 y1 w h cf -1 -1 -1')
            # seq_gt = osp.join(gt_mappings[self.test_mode], one + '.txt')
            seq_gt = self.path_pattern(self.test_mode, 'gt', one)
            if hasattr(self, 'gt_parse'):
                gt = self.gt_parse(seq_gt)
            else:
                gt = TrackSet(
                    seq_gt,
                    formatter=self.gt_formatter,
                    filter=self.gt_filter,
                )
            # self._match_gt(dt, gt)
            for gid in gt.allId():
                if gid < 0:
                    continue
                o = gt(gid)
                frs = list(o.allFr())
                frs = [frs[0], frs[len(frs) // 2], frs[-1]]
                tuples = []
                for fr in frs:
                    if isinstance(vid.backend, ImgVideoCapture):
                        im_path = osp.join(
                            vid.backend.i_root,
                            vid.backend.fmt % (vid.backend.start + fr - 1)
                        )
                    elif seq_imdir.endswith(('.jpg', '.jpeg', '.png')):
                        im_path = seq_imdir
                    else:
                        im_path = (seq_imdir, fr)
                        raise NotImplementedError(
                            'unable to get filename for a video frame'
                        )
                    tuples.append((im_path, o[fr][0]))
                data_ = {'img': [], 'dets': [], 'uid': gid, 'seq': one}
                for im, pre_d in tuples:
                    data_['img'].append(im)
                    data_['dets'].append(
                        [pre_d.x1, pre_d.y1, pre_d.x2 + 1, pre_d.y2 + 1]
                    )
                data.append(data_)
        return data

    def _cache_init(self):
        if not self._cache_flag:
            # self._cache_ims = dequeue(self.h_len * 5)
            self._cache_flag = True

    def _read_im(self, p):
        # for i in self._cache_ims:
        #     if i[0] == p:
        #         return i[1]
        im = read_image(p)
        # self._cache_ims.append((p, im))
        return im

    def _crop(self, im, d):
        x1, y1, x2, y2 = map(int, d[:4])
        crop = im.crop((x1, y1, x2, y2))
        return crop

    def _process_im(self, path, dets):
        self._cache_init()
        single = False
        if isinstance(path, str):
            path = [path]
            dets = dets.reshape(1, -1)
            single = True
        ret = []
        for p, d in zip(path, dets):
            im = self._read_im(p)
            ret.append(self._crop(im, d))
        if single:
            return ret[0]
        else:
            return ret

    def _pipeline_im(self, d):
        if self.test_mode == 'train' or self.test_in_train_format:
            return self.transform_tr(d)
        else:
            return self.transform_te(d)

    def pipeline(self, data):
        for k in data:
            if 'im' in k:
                if isinstance(data[k], list):
                    for i in range(len(data[k])):
                        data[k][i] = self._pipeline_im(data[k][i])
                    data[k] = torch.stack(data[k])
                else:
                    data[k] = self._pipeline_im(data[k])
                    data[k] = data[k].unsqueeze(0)
            if 'dt' in k:
                data[k] = torch.from_numpy(data[k]).float()
                data[k] /= self.loc_std
            # if 'dets' in k:
            #     data['dt'] = torch.from_numpy(data[k]).float()
            #     data['dt'] /= self.loc_std
        return data

    def __getitem__(self, ind):
        data = self.data[ind]
        ret = {}
        for k in ['seq', 'uid']:
            ret[k] = data[k]
        ret['dets'] = np.array(data['dets'])
        ret['filename'] = data['img']
        ret['im'] = self._process_im(data['img'], ret['dets'])
        ret = self.pipeline(ret)
        return ret


class HIE20(MOT16):

    dataset_dir = 'HIE20'

    def generate_paths(self):
        img_pattern = 'imgs/train/%s/'
        img_pattern_test = 'imgs/test/%s/'
        det_pattern = 'dts/train/%s.txt'
        det_pattern_test = 'dts/test/%s.txt'
        gt_pattern = 'labels/train/track1/%s.txt'
        self.train_im_dir = osp.join(self.dataset_dir, img_pattern)
        self.val_im_dir = osp.join(self.dataset_dir, img_pattern)
        self.test_im_dir = osp.join(self.dataset_dir, img_pattern_test)
        self.train_dt_dir = osp.join(self.dataset_dir, det_pattern)
        self.val_dt_dir = osp.join(self.dataset_dir, det_pattern)
        self.test_dt_dir = osp.join(self.dataset_dir, det_pattern_test)
        self.train_gt_dir = osp.join(self.dataset_dir, gt_pattern)
        self.val_gt_dir = osp.join(self.dataset_dir, gt_pattern)
        im_mappings = {
            'test': self.test_im_dir,
            'val': self.val_im_dir,
            'train': self.train_im_dir,
        }
        dt_mappings = {
            'test': self.test_dt_dir,
            'val': self.val_dt_dir,
            'train': self.train_dt_dir,
        }
        gt_mappings = {
            'val': self.val_gt_dir,
            'train': self.train_gt_dir,
        }

        def gen_pattern(mode, item, seq):
            if item == 'img':
                return im_mappings[mode] % seq
            elif item == 'det':
                return dt_mappings[mode] % seq
            elif item == 'gt':
                return gt_mappings[mode] % seq
            else:
                raise

        self.path_pattern = gen_pattern
        self.gt_formatter = 'fr.i id.i x1 y1 w h st.i -1 -1 -1'
        self.gt_filter = lambda d: d.status == 1
        self.dt_formatter = None


class CrowdHuman(MOT16):

    dataset_dir = 'crowd_human'

    def generate_paths(self):
        img_pattern = 'train/Images/%s.jpg'
        img_pattern_test = 'val/Images/%s.jpg'
        det_pattern = 'train/Dets/%s.txt'
        det_pattern_test = 'val/Dets/%s.txt'
        gt_pattern = 'annotation_train.odgt'
        self.train_im_dir = osp.join(self.dataset_dir, img_pattern)
        self.val_im_dir = osp.join(self.dataset_dir, img_pattern)
        self.test_im_dir = osp.join(self.dataset_dir, img_pattern_test)
        self.train_dt_dir = osp.join(self.dataset_dir, det_pattern)
        self.val_dt_dir = osp.join(self.dataset_dir, det_pattern)
        self.test_dt_dir = osp.join(self.dataset_dir, det_pattern_test)
        self.train_gt_dir = osp.join(self.dataset_dir, gt_pattern)
        self.val_gt_dir = osp.join(self.dataset_dir, gt_pattern)
        im_mappings = {
            'test': self.test_im_dir,
            'val': self.val_im_dir,
            'train': self.train_im_dir,
        }
        dt_mappings = {
            'test': self.test_dt_dir,
            'val': self.val_dt_dir,
            'train': self.train_dt_dir,
        }
        gt_mappings = {
            'val': self.val_gt_dir,
            'train': self.train_gt_dir,
        }

        def gen_pattern(mode, item, seq):
            u = json.loads(seq)
            seq = u['ID']
            if item == 'img':
                return im_mappings[mode] % seq
            elif item == 'det':
                return dt_mappings[mode] % seq
            elif item == 'gt':
                return u
            else:
                raise

        self.path_pattern = gen_pattern
        self.gt_formatter = None
        self.gt_filter = None
        self.dt_formatter = None

    def gt_parse(self, gt):
        ret = TrackSet()
        for g in gt['gtboxes']:
            ignore = g['extra'].pop('ignore', 0)
            if g['tag'] == 'person' and not ignore:
                x1, y1, w, h = g['fbox']
                uid = g['extra']['box_id']
                d = Det(x1, y1, w, h, cls=1, uid=uid)
                d.fr = 1
                ret.append_data(d)
        return ret
