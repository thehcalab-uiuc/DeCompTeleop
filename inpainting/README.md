# Instructions for Training ResNet Inpainting Model, Saving Predictions, and Evaluating Accuracy

We assume you have activated the [`dav2`](../depth/dav2/README.md) Conda environment.
Or, you can activate the [`terra-torch3d`](../README.md) Conda environment, and install requirements for [`pytorch-msssim`](https://pypi.org/project/pytorch-msssim/) and [`lpips`](https://pypi.org/project/lpips/).
You should also have generated delayed reprojections using a given depth model by following instructions [here](../depth/collect_eval_metrics/README.md).

## Train Model

- Update file names and valid timesteps to load for train, validation, and test splits in the dataset script [here](dataset.py)
- Update path to pretrained VisionNavNet in the model file [here](refinement.py)
- Run [`python train_resnet_inpaint.py`](train_resnet_inpaint.py) with desired arguments to train an inpainting model

## Collect Predictions

- Save predictions from pretrained ResNet inpainting model:
    - Update path to trained checkpoint to load and names of videos to evaluate in [`save_resnet_inpaint.py`](save_resnet_inpaint.py)
    - Run [`python save_resnet_inpaint.py`](save_resnet_inpaint.py) for each video in dataset that you want to inpaint
- Save predictions from classic Telea inpainting method:
    - Update names of videos to evaluate in [`save_telea_inpaint.py`](save_telea_inpaint.py)
    - Run [`python save_telea_inpaint.py`](save_telea_inpaint.py) for each video in dataset that you want to inpaint

## Evaluate Accuracy

- To compute quantitative accuracy of inpainted images
    - Run [`python eval_inpaint.py`](eval_inpaint.py) for each video in dataset that you want to evaluate
