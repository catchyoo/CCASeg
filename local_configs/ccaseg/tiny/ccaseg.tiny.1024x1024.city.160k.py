_base_ = [
    '../../_base_/models/ccaseg-tiny.py', '../../_base_/datasets/cityscapes_1024x1024.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k_adamw.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
randomness = dict(seed=2050920549)
checkpoint = 'ckpt/mscan_t_20230227-119e8c9f.pth' 

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)),
    decode_head=dict(num_classes=19))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
