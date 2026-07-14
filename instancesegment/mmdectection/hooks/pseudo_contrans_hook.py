import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmcv.ops import bbox_overlaps, roi_align
from pycocotools import mask as mask_utils


def mask2json(mask: np.ndarray) -> dict:
    """把二值 mask 转成 COCO RLE 格式的 dict（counts 要解码成 str）。"""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def iou(box, gt_box):
    """计算单对单框的 IoU."""
    # box 和 gt_box 都是 [x1,y1,x2,y2]
    # 转为 numpy 一维数组
    box_arr = np.array(box, dtype=np.float32).reshape(4,)
    gt_arr  = np.array(gt_box, dtype=np.float32).reshape(4,)

    # 规范化坐标，保证 x1<=x2, y1<=y2
    x1, y1, x2, y2 = box_arr
    x1n, x2n = min(x1, x2), max(x1, x2)
    y1n, y2n = min(y1, y2), max(y1, y2)
    box_norm = np.array([x1n, y1n, x2n, y2n], dtype=np.float32)

    gx1, gy1, gx2, gy2 = gt_arr
    gx1n, gx2n = min(gx1, gx2), max(gx1, gx2)
    gy1n, gy2n = min(gy1, gy2), max(gy1, gy2)
    gt_norm = np.array([gx1n, gy1n, gx2n, gy2n], dtype=np.float32)

    # 转 torch.Tensor（在 CPU 上）
    b = torch.from_numpy(box_norm[None, :])  # shape [1,4]
    g = torch.from_numpy(gt_norm[None, :])   # shape [1,4]


    # bbox_overlaps 返回 shape (1,1) 的 overlap 矩阵
    return float(bbox_overlaps(b, g, mode='iou')[0, 0])


# @HOOKS.register_module()
# class PseudoLabelContransHook(Hook):
#     def __init__(self,
#                  pseudo_thr: float = 0.8,
#                  tau_exclude: float = 0.3,
#                  sim_thr: float = 0.7,
#                  interval: int = 50):
#         """
#         Args:
#             pseudo_thr: 伪标签置信度阈值
#             tau_exclude: 与真实框 IoU 排除阈值
#             sim_thr: 伪标签与真实框特征余弦相似度阈值
#             interval: 多少个 epoch 生成一次伪标签
#         """
#         super().__init__()
#         self.pseudo_thr = pseudo_thr
#         self.tau_exclude = tau_exclude
#         self.sim_thr = sim_thr
#         self.interval = interval

#     def after_train_epoch(self, runner):
#         epoch = runner.epoch + 1
#         if epoch % self.interval != 0:
#             return

#         runner.model.eval()

#         # 读原始标注
#         ds_cfg = runner.cfg.train_dataloader.dataset
#         rel_ann = ds_cfg.ann_file
#         gt_json = rel_ann if os.path.isabs(rel_ann) else os.path.join(ds_cfg.data_root, rel_ann)
#         with open(gt_json, 'r') as f:
#             gt = json.load(f)

#         images     = gt['images']
#         categories = gt['categories']
#         merged_anns = gt['annotations'].copy()
#         next_id = max(a['id'] for a in merged_anns) + 1
        
#         # 2. 用 runner.build_dataloader 构造一个 shuffle=False, batch_size=1 的推理 loader
#         infer_cfg = runner.cfg.train_dataloader.copy()
#         # 不改变原有训练 loader
#         # a) batch_size -> 1
#         if 'samples_per_gpu' in infer_cfg:
#             infer_cfg['samples_per_gpu'] = 1
#         elif 'batch_size' in infer_cfg:
#             infer_cfg['batch_size'] = 1
#         # b) 关 shuffle：如果用 sampler 配置，就改 sampler.shuffle；否则直接改 top-level shuffle
#         if 'sampler' in infer_cfg and 'shuffle' in infer_cfg.sampler:
#             infer_cfg.sampler['shuffle'] = False
#         else:
#             infer_cfg['shuffle'] = False

#         infer_loader = runner.build_dataloader(infer_cfg)


#         # 流式处理每个 batch
#         for data in infer_loader:
#             if isinstance(data, (list, tuple)):
#                 data = data[0]
#             # 拆 imgs
#             imgs = data.get('inputs', data.get('img', data.get('imgs')))
#             if isinstance(imgs, (list, tuple)):
#                 imgs = torch.stack(imgs, dim=0)
#             device = next(runner.model.parameters()).device
#             imgs = imgs.to(device, non_blocking=True)

#             with torch.no_grad():
#                 feats   = runner.model.extract_feat(imgs)
#                 results = runner.model.test_step(data)

                
#             # 逐图处理
#             for idx, ds in enumerate(results):
#                 img_id = ds.metainfo['img_id']
#                 file_name = ds.metainfo.get('img_path',
#                                             ds.metainfo.get('img_filename'))
#                 print(f"[PseudoHook] infer idx={idx}  img_id={img_id}  file={file_name}")
#                 inst   = ds.pred_instances
#                 # bboxes [N,4], scores [N]
#                 bboxes = inst.bboxes.cpu().numpy()
#                 scores = inst.scores.cpu().numpy()
#                 bboxes = np.concatenate([bboxes, scores[:, None]], axis=1)  # [N,5]
#                 # masks [N,H,W]
#                 mf = inst.masks
#                 masks = mf.cpu().numpy() if isinstance(mf, torch.Tensor) else mf.masks

#                 # IoU 初筛
#                 prelim = []
#                 for i, score in enumerate(scores):
#                     if score < self.pseudo_thr:
#                         continue
#                     box = bboxes[i, :4].tolist()
#                     # 计算与所有 GT 的最大 IoU
#                     gts = [ann for ann in gt['annotations'] if ann['image_id']==img_id]
#                     ious = [iou(box, [g['bbox'][0], g['bbox'][1], g['bbox'][0]+g['bbox'][2], g['bbox'][1]+g['bbox'][3]]) for g in gts]
#                     max_iou = max(ious) if ious else 0.0
#                     if max_iou >= self.tau_exclude:
#                         continue
#                     prelim.append((i, box))

#                 if not prelim:
#                     continue

#                 # ROIAlign + 余弦相似度二次筛选
#                 feat = feats[0][idx:idx+1]  # [1,C,H,W]
#                 device = feat.device
#                 # 构造 rois
#                 gt_rois = []
#                 for g in gt['annotations']:
#                     if g['image_id'] == img_id:
#                         x,y,w,h = g['bbox']
#                         gt_rois.append([0, x, y, x+w, y+h])
#                 ps_rois = [[0, *p[1]] for p in prelim]
#                 if not gt_rois:
#                     continue
#                 gt_rois = torch.tensor(gt_rois, dtype=torch.float32, device=device)
#                 ps_rois = torch.tensor(ps_rois, dtype=torch.float32, device=device)

#                 out_size = (7, 7)
#                 spatial_scale = 1.0 / 4
#                 sampling_ratio = 2
#                 gt_feats = roi_align(feat, gt_rois, out_size, spatial_scale, sampling_ratio)
#                 ps_feats = roi_align(feat, ps_rois, out_size, spatial_scale, sampling_ratio)

#                 gt_vecs = gt_feats.mean(dim=[2, 3])  # [M,C]
#                 ps_vecs = ps_feats.mean(dim=[2, 3])  # [K,C]

#                 sims = F.cosine_similarity(ps_vecs.unsqueeze(1), gt_vecs.unsqueeze(0), dim=-1)  # [K,M]
#                 max_sims, _ = sims.max(dim=1)  # [K]

#                 keep = (max_sims >= self.sim_thr).nonzero(as_tuple=True)[0].cpu().tolist()
#                 for k in keep:
#                     i, box = prelim[k]
#                     x1,y1,x2,y2 = box
#                     w, h = x2 - x1, y2 - y1
#                     merged_anns.append({
#                         'id': next_id,
#                         'image_id': img_id,
#                         'category_id': 1,
#                         'bbox': [x1, y1, w, h],
#                         'segmentation': mask2json(masks[i] > 0.5),
#                         'area': float(w*h),
#                         'iscrowd': 0
#                     })
#                     next_id += 1

#             # 释放本 batch 占用的显存
#             del imgs, feats, results
#             torch.cuda.empty_cache()

#         # 将合并后的标注写到新 JSON
#         out_name = f'instances_train_pseudo_epoch{epoch}.json'
#         out_path = os.path.join(os.path.dirname(gt_json), out_name)
#         merged = {
#             'images': images,
#             'annotations': merged_anns,
#             'categories': categories
#         }
#         with open(out_path, 'w') as f:
#             json.dump(merged, f)

#         # 更新 runner 配置并重新加载 dataloader
#         runner.cfg.train_dataloader.dataset.ann_file = out_path
#         # 如果 mmengine 版本支持 reload_dataloader：
#         try:
#             runner.reload_dataloader(data_name='train_dataloader')
#         except AttributeError:
#             # 否则用 build_dataloader + 替换 train_loop.dataloader
#             new_loader = runner.build_dataloader(runner.cfg.train_dataloader)
#             runner.train_loop.dataloader = new_loader






@HOOKS.register_module()
class PseudoLabelContransHook(Hook):
    def __init__(self,
                 pseudo_thr: float = 0.8,
                 tau_exclude: float = 0.3,
                 sim_thr: float = 0.7,
                 interval: int = 50):
        """
        Args:
            pseudo_thr:   置信度阈值
            tau_exclude:  与真实框 IoU 排除阈值
            sim_thr:      余弦相似度阈值
            interval:     每隔多少 epoch 生成一次
        """
        super().__init__()
        self.pseudo_thr  = pseudo_thr
        self.tau_exclude = tau_exclude
        self.sim_thr     = sim_thr
        self.interval    = interval

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch % self.interval != 0:
            return

        runner.model.eval()

        # 1) 读原始 COCO 标注
        ds_cfg  = runner.cfg.train_dataloader.dataset
        rel_ann = ds_cfg.ann_file
        gt_json = rel_ann if os.path.isabs(rel_ann) else os.path.join(ds_cfg.data_root, rel_ann)
        with open(gt_json, 'r') as f:
            gt = json.load(f)

        images      = gt['images']
        categories  = gt['categories']
        merged_anns = gt['annotations'][:]  # 拷贝原注释
        next_id     = max(a['id'] for a in merged_anns) + 1

        # 2) 基于 train_dataloader 配置，构造推理 loader（batch=1, shuffle=False）
        infer_cfg = runner.cfg.train_dataloader.copy()
        if 'samples_per_gpu' in infer_cfg:
            infer_cfg['samples_per_gpu'] = 1
        elif 'batch_size' in infer_cfg:
            infer_cfg['batch_size'] = 1
        infer_cfg['shuffle'] = False
        infer_loader = runner.build_dataloader(infer_cfg)

        # 记录哪些图生成了伪标签
        generated = []

        # 3) 流式推理 & 伪标签生成
        for data in infer_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            imgs = data.get('inputs', data.get('img', data.get('imgs')))
            if isinstance(imgs, (list, tuple)):
                imgs = torch.stack(imgs, dim=0)
            device = next(runner.model.parameters()).device
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                feats   = runner.model.extract_feat(imgs)
                results = runner.model.test_step(data)

            # 每张图逐个处理（batch_size=1）
            for idx, ds in enumerate(results):
                img_id   = ds.metainfo['img_id']
                img_path = ds.metainfo.get('img_path', ds.metainfo.get('img_filename', 'UNKNOWN'))
                inst     = ds.pred_instances

                # 提取 bboxes, scores, masks
                bboxes_np = inst.bboxes.cpu().numpy()             # [N,4]
                scores_np = inst.scores.cpu().numpy()             # [N]
                bboxes    = np.concatenate([bboxes_np, scores_np[:,None]], axis=1)  # [N,5]
                mf        = inst.masks
                masks_np  = mf.cpu().numpy() if isinstance(mf, torch.Tensor) else mf.masks  # [N,H,W]

                # IoU 初筛
                gt_boxes = [
                    [a['bbox'][0],
                     a['bbox'][1],
                     a['bbox'][0]+a['bbox'][2],
                     a['bbox'][1]+a['bbox'][3]]
                    for a in gt['annotations']
                    if a['image_id']==img_id
                ]
                prelim = []
                for i, score in enumerate(scores_np):
                    if score < self.pseudo_thr:
                        continue
                    box    = bboxes[i,:4].tolist()
                    max_i  = max((iou(box, gb) for gb in gt_boxes), default=0.0)
                    if max_i >= self.tau_exclude:
                        continue
                    prelim.append((i, box))

                # 如果有新增伪标签，记录并生成
                if prelim:
                    generated.append({'image_id': img_id, 'path': img_path})

                    # ROIAlign + Cosine 相似度二次筛选
                    feat0   = feats[0][idx:idx+1]   # [1,C,H,W]
                    device0 = feat0.device
                    # 构造 RoIs
                    gt_rois = torch.tensor(
                        [[0, *gb] for gb in gt_boxes],
                        dtype=torch.float32, device=device0
                    )
                    ps_rois = torch.tensor(
                        [[0, *pb] for (_, pb) in prelim],
                        dtype=torch.float32, device=device0
                    )
                    out_size      = (7,7)
                    spatial_scale = 1.0 / 4
                    samp_ratio    = 2
                    gt_feats = roi_align(feat0, gt_rois, out_size, spatial_scale, samp_ratio)
                    ps_feats = roi_align(feat0, ps_rois, out_size, spatial_scale, samp_ratio)
                    gt_vecs  = gt_feats.mean(dim=[2,3])  # [M,C]
                    ps_vecs  = ps_feats.mean(dim=[2,3])  # [K,C]
                    sims     = F.cosine_similarity(ps_vecs.unsqueeze(1),
                                                   gt_vecs.unsqueeze(0),
                                                   dim=-1)  # [K,M]
                    max_sims, _ = sims.max(dim=1)        # [K]
                    keep = (max_sims >= self.sim_thr).nonzero(as_tuple=True)[0].cpu().tolist()

                    # 追加伪标签 annotation
                    for k in keep:
                        i, box = prelim[k]
                        x1,y1,x2,y2 = box
                        w, h = x2-x1, y2-y1
                        merged_anns.append({
                            'id':          next_id,
                            'image_id':    img_id,
                            'category_id': 1,
                            'bbox':        [x1, y1, w, h],
                            'segmentation': mask2json(masks_np[i] > 0.5),
                            'area':        float(w*h),
                            'iscrowd':     0
                        })
                        next_id += 1

            # 释放显存
            del imgs, feats, results
            torch.cuda.empty_cache()

        # 4) 打印生成伪标签的 image_id & 路径
        print(f"[Epoch {epoch}] 生成伪标签的 images:")
        for item in generated:
            print(f"  image_id={item['image_id']}  path={item['path']}")

        # 5) 写新 JSON 并 reload dataloader
        out_name = f'instances_train_pseudo_epoch{epoch}.json'
        out_path = os.path.join(os.path.dirname(gt_json), out_name)
        with open(out_path, 'w') as f:
            json.dump({
                'images':     images,
                'annotations': merged_anns,
                'categories': categories
            }, f)

        runner.cfg.train_dataloader.dataset.ann_file = out_path
        try:
            runner.train_loop.reload_dataloader(data_name='train_dataloader')
        except AttributeError:
            new_loader = runner.build_dataloader(runner.cfg.train_dataloader)
            runner.train_loop.dataloader = new_loader