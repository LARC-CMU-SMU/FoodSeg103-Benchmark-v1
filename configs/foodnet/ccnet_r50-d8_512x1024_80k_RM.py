_base_ = [
    '../_base_/models/ccnet_r50-d8.py', '../_base_/datasets/FoodSeg103.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(pretrained='./pretrained_model/R50_ReLeM.pth', backbone=dict(type='ResNet'))
