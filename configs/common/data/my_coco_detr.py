import itertools

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

# HRT2
# register_coco_instances("my_dataset_train", {}, '/home/kb535/lss/data/200png/object_detection/coco_style/annotations/instances_train2014.json',
#                         '/home/kb535/lss/data/200png/object_detection/coco_style/train2014')
# register_coco_instances("my_dataset_test", {}, '/home/kb535/lss/data/200png/object_detection/coco_style/annotations/instances_val2014.json',
#                         '/home/kb535/lss/data/200png/object_detection/coco_style/val2014')

# LRT2
register_coco_instances("my_dataset_train", {}, '/media/kb535/Data/data_lss/coco2017/coco/annotations/instances_train2017.json',
                        '/media/kb535/Data/data_lss/coco2017/coco/train2017')
register_coco_instances("my_dataset_test", {}, '/media/kb535/Data/data_lss/coco2017/coco/annotations/instances_val2017.json',
                        '/media/kb535/Data/data_lss/coco2017/coco/val2017')


dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="my_dataset_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            # L(T.RandomFlip)(),
            # L(T.ResizeShortestEdge)(
            #     short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
            #     max_size=1333,
            #     sample_style="choice",
            # ),
            # L(T.ResizeShortestEdge)(
            #     short_edge_length=512,
            #     max_size=512,
            # ),
            L(T.Resize)(
                shape=(512, 512)
            ),
        ],
        # augmentation_with_crop=[
        #     L(T.RandomFlip)(),
        #     L(T.ResizeShortestEdge)(
        #         short_edge_length=(400, 500, 600),
        #         sample_style="choice",
        #     ),
        #     L(T.RandomCrop)(
        #         crop_type="absolute_range",
        #         crop_size=(384, 600),
        #     ),
        #     L(T.ResizeShortestEdge)(
        #         short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
        #         max_size=1333,
        #         sample_style="choice",
        #     ),
        # ],
        augmentation_with_crop=None,
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="my_dataset_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            # L(T.ResizeShortestEdge)(
            #     short_edge_length=800,
            #     max_size=1333,
            # ),
            # L(T.ResizeShortestEdge)(
            #     short_edge_length=512,
            #     max_size=512,
            # ),
            L(T.Resize)(
                shape=(512, 512)
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
