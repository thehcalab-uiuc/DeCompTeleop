# Instructions for Training DINOv2 Model and Saving Predictions 

## Conda Environment Creation

We use a separate Conda environment for training and generation predictions with DINOv2.
Create the DINOv2 Conda environment from our [yaml file](dino.yml) by running:

```
conda env create -f dino.yml
conda activate dinov2
```
**NOTE:** We trained DINOv2 models on a compute cluster with a ppc64le architecture, rather than the usual x86_64 architecture. As such, the Conda environment may need to be updated for your specific compute setup.

## Train Model

- Update file names and valid timesteps to load for train, validation, and test splits in the dataset script [here](dataset.py)
- Run [`python trainer.py`](trainer.py) with desired arguments to train a DINOv2 model

## Collect Predictions

- Update path to trained checkpoint to load and names of videos to evaluate in [`save_dino_depth.py`](save_dino_depth.py)
- Run [`python save_dino_depth.py`](save_dino_depth.py) to generate depth predictions for a set of videos
