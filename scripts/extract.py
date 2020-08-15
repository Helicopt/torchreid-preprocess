#!/usr/bin/python
# -*- coding: utf8 -*-
#########################################################################
# File Name: tests/flow.py
# Author: Toka
# mail: <empty>
# Created Time: Mon Jul 27 02:47:46 2020
#########################################################################
import os
import sys
sys.path.insert(0, '')
import torchreid
from torchreid.data.datasets.image.TRACK_MOT16 import MOT16
import torch
import torch.nn.functional as F
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)
from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)

def to_cuda(*args, device='cuda'):
    ret = []
    for i in args:
        if isinstance(i, torch.Tensor):
            ret.append(i.to(device))
        elif isinstance(i, (list, tuple)):
            ret.append(to_cuda(*i, device=device))
        elif isinstance(i, dict):
            vs = to_cuda(*i.values(), device=device)
            ret.append(dict(zip(i.keys(), vs)))
        else:
            ret.append(i)
    return ret

if __name__=='__main__':
    # __author__ == '__toka__'
    d = MOT16(root='/mnt/lustre/share/fengweitao')
    dl = torch.utils.data.DataLoader(d, batch_size=16, num_workers=4)
    print(dl)
    config_file = 'configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml'
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    cfg.merge_from_file(config_file)
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=1024,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    load_pretrained_weights(model, cfg.model.load_weights)
    # m = None
    model.eval()
    model.cuda()
    all_results = []
    for i, data in enumerate(dl):
        data = to_cuda(data, device='cuda')[0]
        print(data.keys())
        n, m, c, h, w = data['im'].shape
        indata = data['im'].reshape(n * m, c, h, w)
        with torch.no_grad():
            o = model(indata)
        o = o.reshape(n, m, -1)
        dm = 0.
        for j in range(o.size(1)):
            for k in range(o.size(1)):
                f1 = F.normalize(o[:, j], dim=1)
                f2 = F.normalize(o[:, (j+k+1)%o.size(1)], dim=1)
                d = (f1 * f2).sum(dim=1)
                dm += d
        dm /= o.size(1) ** 2
        print(dm)
        o_u = o.mean(dim=1)
        for j in range(o_u.size(0)):
            all_results.append({'seq': data['seq'][j], 'uid': data['uid'][j].cpu().numpy(), 'feat': o_u[j].cpu().numpy()})
        #dm = o_u.matmul(o_u.transpose(1, 0))
        #print(dm.max(dim=1))
    torch.save(all_results, 'MOT16_cluster.pkl')
        
