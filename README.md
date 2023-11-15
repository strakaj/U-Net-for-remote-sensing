# Modernized Training of U-Net for Aerial Semantic Segmentation

## Introduction
This repository is based on [FLAIR #2](https://github.com/IGNF/FLAIR-2-AI-Challenge) baseline model. We improved the training procedure of the model and conducted an extensive ablation study on the influence of different parameters and components on the overall performance.

## Environment
Since we built on the baseline solution, it is possible to use the original repository environment:
```bash
conda create -n flair python=3.8.10
pip install -r  requirements.txt
```

## Data
We used the FLAIR #2 dataset for training and evaluation. Data can be downloaded from the [FLAIR website](https://ignf.github.io/FLAIR/)

## Usage
We trained the model with multiple backbones on different folds of the dataset. As a final solution, we tested multiple ensembles created as a combination of these models. We provide checkpoints for two ensembles. Trained checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1kLUI0jKwKpUE2s1XMT3vI7OBaCGGYzeV?usp=sharing). In order for prediction scripts to work checkpoints should be placed in the `weights` folder. First is an ensemble composed of four models which achieved the best performance on the test set. The second is an ensemble composed of three models which we chose for the best trade-off between performance and speed. 

Before running any script it is necessary to edit the path to data in `flair-2-config.yml`

Predictions can be made by running:
```bash
python run/predict/predict-ensemble_03.sh
python run/predict/predict-ensemble_04.sh
```

Models are trained in two stages. The first is necessary to train U-Net, for that can be used scripts in `run/train_unet`. Then can be trained full model initialized from U-Net, scripts for training of full model are in  `/run/train_main`. We used Wandb for logging, for that it is necessary to enter the wandb key into the training files. By removing the wandb key and argument from the script call, data will be logged by tensorboard.
