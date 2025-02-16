# Instructions for Training DAv2 Model and Saving Predictions 

## Conda Environment Creation

We use a separate Conda environment for training and generation predictions with DAv2.
Create the DAv2 Conda environment from our [yaml file](dav2.yml) by running:

```
conda env create -f dav2.yml
conda activate dav2
```

## Train Model

- Update file names and valid timesteps to load for train, validation, and test splits in the dataset script [here](train/metric_depth/dataset/terra.py)
- Run [`python train/metric_depth/train.py`](train/metric_depth/train.py) with desired arguments to train a DAv2 model
    - An example of arguments are provided at [`train/metric_depth/dist_train.sh`](train/metric_depth/dist_train.sh)

## Collect Predictions

- Update path to trained checkpoint to load in [`inference/save_dav2_depth.py`](inference/save_dav2_depth.py)
- Run [`python inference/save_dav2_depth.py`](inference/save_dav2_depth.py) for each video in dataset that you want to generate depth predictions for
