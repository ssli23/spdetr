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



dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="my_dataset_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[

            L(T.ResizeShortestEdge)(
                short_edge_length=512,
                max_size=512,
            ),
        ],

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

            L(T.ResizeShortestEdge)(
                short_edge_length=512,
                max_size=512,
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
