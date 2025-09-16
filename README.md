Source for "Mind the Dialect: NLP Advancements Uncover Fairness Disparities for Arabic Users in Recommendation Systems"

## Create the conda environment

conda create --name ENV_NAME --file conda_requirements.txt python=3.7 \
conda activate ENV_NAME \
conda install pip \
python3 -m pip install -r requirements.txt

## Reproducing Results

The main results in the paper can be reproduced by running the following commands:

export DATASETS_DIR=PATH/TO/DATASETS

TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset=BARD --dataset_dir=DATASETS_DIR --device=gpu --batch_size=1024 --print_freq=128 --lr=2e-5 --epochs=100 --margin=1 --num_negatives=20 --num_workers=32

## Hardware Requiremnts

The code is designed to run on a single GPU. The training time for a single epoch depends on the dataset, but typically takes two minutes on a desktop-class GPU. The code has been tested on a machine with over 100GB of RAM, which allows data loading to be efficient.

## Base Repo

This repo is based on [BiasedUserHistorySynthesis](https://github.com/lkp411/BiasedUserHistorySynthesis) by [lkp411](https://github.com/lkp411).

