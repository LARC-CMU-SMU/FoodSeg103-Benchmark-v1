_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/FoodSeg103.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(pretrained='./pretrained_model/R50_ReLeM.pth', 
            backbone=dict(type='ResNet'), 
            decode_head=dict(num_classes=104))

optimizer_config = dict()

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
