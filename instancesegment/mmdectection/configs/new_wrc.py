_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py'
]

# custom_imports = dict(
#     imports=['projects.self_training.sobel_pseudo_hook'],
#     allow_failed_imports=False
# )

# from projects.self_training import pseudo
# from projects.self_training import pseudo_hook

default_scope = 'mmdet'
log_processor = dict(
    type='LogProcessor',  # 日志处理器用于处理运行时日志
    window_size=50,  # 日志数值的平滑窗口
    by_epoch=True)  # 是否使用 epoch 格式的日志。需要与训练循环的类型保存一致。

log_level = 'INFO'  # 日志等级
load_from = '/root/segment3/mmdetection-main/configs/new_wrc/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。

# dataset settings
dataset_type = 'NewCocoDataset' 
data_root = '/root/segment3/mmdetection-main/data/coco/'

model = dict(
    # _delete_=True,
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # mask_head=dict(
        #     type='FCNMaskHead',
        #     num_convs=4,
        #     in_channels=256,
        #     conv_out_channels=256,
        #     num_classes=2,
        #     loss_mask=dict(
        #         type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        # mask_head = dict(
        #     type='DCTFCNMaskHead',          # ← 用 DCT 版的 Mask Head
        #     num_convs=4,
        #     in_channels=256,
        #     conv_out_channels=256,
        #     num_classes=2,                  # 你的原始配置是 2，就保持 2
        #     dct=dict(                       # ← 新增 DCT 模块
        #         type='DCTFusion',
        #         in_channels=256,
        #         fusion='concat',            # 可选 'sum'；concat 后用 1x1 压回
        #         out_channels=256,
        #         init='lowpass',             # 'lowpass' | 'highpass' | 'allpass'
        #         learnable=True,
        #         roi_size=14                 # 必须与 RoIAlign 的 output_size 一致
        #     ),
        #     loss_mask=dict(
        #         type='CrossEntropyLoss', use_mask=True, loss_weight=1.0
        #     )
        # )
        mask_head=dict(
            type='DCTFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=2,   # 保持你的类别数
            dct=dict(
                type='PatchDCTFusion',
                in_channels=256,
                tile=7,                 # 14x14 RoI → 7x7 块更自然（2x2 网格）
                stride=7,               # =tile 表示不重叠；想更细致可设 4/6 实现重叠
                fusion='concat',        # 或 'sum'
                out_channels=256,       # concat 后压回 256
                per_channel=False,      # True=通道级掩膜，更细但更慢
                init='allpass',         # 可换 'lowpass'/'highpass' 做先验
                learnable=True,
                pad_mode='replicate'    # RoI 不是整除 tile 时，自动 pad
            ),
            # —— 这里接上 PatchDCTCriterion（关键）——
            patch_dct_loss=dict(
                type='PatchDCTCriterion',
                tile=7, stride=7,
                N_global=64,    # 整图 DCT 向量长度
                n_patch=16,     # patch DCT 向量长度
                loss_type='smooth_l1',
                lambda0=1.0,    # L_dct^N 的权重
                lambdas=None,   # 多阶段时设成 [1.0, 1.0, ...]；单阶段可 None
                thr_fg=0.95, thr_bg=0.05,
                logits_input=True,
                use_mean_as_classifier=True,  # 先用均值近似；之后可换成专门的分类 head
                temperature=0.25
            ),
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
            )
        ),

    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=1000,
            mask_thr_binary=0.5)))


# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [  # 训练数据处理流程
    # dict(_delete_=True, type='LoadImageFromFile'),
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像。
    dict(
        type='LoadAnnotations',  # 第 2 个流程，对于当前图像，加载它的注释信息。
        with_bbox=True,  # 是否使用标注框(bounding box)， 目标检测需要设置为 True。
        with_mask=True,  # 是否使用 instance mask，实例分割需要设置为 True。
        poly2mask=False),  # 是否将 polygon mask 转化为 instance mask, 设置为 False 以加速和节省内存。
    dict(
        type='Resize',  # 变化图像和其标注大小的流程。
        scale=(1333, 800),  # 图像的最大尺寸
        keep_ratio=True  # 是否保持图像的长宽比。
        ),
    # dict(
    #     type='RandomFlip',  # 翻转图像和其标注的数据增广流程。
    #     prob=0.8),  # 翻转图像的概率。
    # dict(type='RandomCrop', crop_size=(800, 800), allow_negative_crop=True),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    ),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')  # 将数据转换为检测器输入格式的流程
]
test_pipeline = [  # 测试数据处理流程
    # dict(_delete_=True),            
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像。
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 变化图像大小的流程。
    dict(
        type='PackDetInputs',  # 将数据转换为检测器输入格式的流程
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(  # 训练 dataloader 配置
    # _delete_=True,
    batch_size=14,  # 单个 GPU 的 batch size
    num_workers=2,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(  # 训练数据的采样器
        type='DefaultSampler',  # 默认的采样器，同时支持分布式和非分布式训练。请参考 https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=False),  # 随机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',  # 标注文件路径
        data_prefix=dict(img='train2017/'),  # 图片路径前缀
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 图片和标注的过滤配置
        pipeline=train_pipeline))  # 这是由之前创建的 train_pipeline 定义的数据处理流程。
val_dataloader = dict(  # 验证 dataloader 配置
    # _delete_=True,
    batch_size=1,  # 单个 GPU 的 Batch size。如果 batch-szie > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=2,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,  # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,  # 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline))
test_dataloader = val_dataloader  # 测试 dataloader 配置



val_evaluator = dict(  # 验证过程使用的评测器
    # _delete_=True,
    type='CocoMetric',  # 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=data_root + 'annotations/instances_val2017.json',  # 标注文件路径
    metric=['bbox', 'segm'],  # 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False)
test_evaluator = val_evaluator  # 测试过程使用的评测器



# 在测试集上推理，
# 并将检测结果转换格式以用于提交结果
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/image_info_test-dev2018.json',
        data_prefix=dict(img='test2018/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/image_info_test-dev2018.json',
    metric=['bbox', 'segm'],
    format_only=True,  # 只将模型输出转换为 coco 的 JSON 格式并保存
    outfile_prefix='./work_dirs/coco_detection/test')  # 要保存的 JSON 文件的前缀


train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=200,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型


optim_wrapper = dict(  # 优化器封装的配置
    # _delete_=True,  # 删除继承的 optim_wrapper 配置               
    type='AmpOptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器。请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # 随机梯度下降优化器
        lr=0.005,  # 基础学习率
        momentum=0.9,  # 带动量的随机梯度下降
        weight_decay=0.0001),  # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )



param_scheduler = [
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=500),  # 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=0,   # 从第一个 epoch 开始
        end=12,  # 到第 12 个 epoch 结束
        milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
]


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    visualization=dict(type='DetVisualizationHook')
)


# custom_hooks = [
#     dict(
#       type='PseudoLabelContransHook',
#       pseudo_thr=0.8,      # 伪标签置信度阈值
#       tau_exclude=0.3,     # 与真标签最大 IoU 排除阈值
#       interval=50          # 每 50 epoch 触发一次
#     )
# ]


# custom_hooks = [
#     dict(
#         type='DirectionalSplitPseudoHook',
#         pseudo_thr=0.8,
#         tau_exclude=0.3,
#         interval=50,
#         gabor_ksize=31,
#         gabor_sigma=4.0,
#         gabor_lambd=10.0,
#         gabor_gamma=0.5)  # 根据需求调整方向列表
# ]

# custom_hooks = [
#     dict(
#         type='ContrastDirectionPseudoHook',
#         pseudo_thr=0.6,
#         tau_exclude=0.3,
#         interval=50,
#         gabor_ksize=31,
#         gabor_sigma=4.0,
#         gabor_lambd=10.0,
#         gabor_gamma=0.5
#     )
# ]


# custom_hooks = [
#     dict(
#         type='Main45PseudoHook',
#         pseudo_thr=0.3,
#         tau_exclude=0.3,
#         sim_thr=0.5,
#         interval=50,
#         delta_deg=22.5
#     )
# ]


# custom_hooks = [
#     dict(
#         type='Main48PseudoHook',
#         pseudo_thr=0.6,
#         tau_exclude=0.3,
#         sim_thr=0.7,
#         interval=50,
#         sobel_ksize=3,
#         orient_bins=36,
#         orient_thresh=22.5
#     )
# ]

gpu_ids = [1]