## SP-DETR: Spatially Priortized DETR for Rectal Cancer Detection in MRI Scans



## Training
Configs can be trained with:
```bash
python tools/train_net.py --config-file projects/sp_detr/configs/path/to/config.py --num-gpus 8
```
By default, we use 8 GPUs with total batch size as 16 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
python tools/train_net.py --config-file projects/sp_detr/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```
