# :briefcase: Learning TSP With Jumping Skip Connect and Attention GCN

This repository contains code for the paper [**"Learning TSP With Jumping Skip Connect and Attention GCN"**]([https://arxiv.org/abs/2006.07054](https://www.jstage.jst.go.jp/article/pjsai/JSAI2022/0/JSAI2022_4E3GS203/_article/-char/ja/)) by Nguyen Huu Bao Long, accepted to the **36th The Japanese Society for Artificial Intelligence** (JSAI2022).

## Overview

- End-to-end training of neural network solvers for combinatorial problems such as the **Travelling Salesman Problem** is intractable and inefficient beyond a few hundreds of nodes. 
While state-of-the-art Machine Learning approaches perform closely to classical solvers for trivially small sizes, they are **unable to generalize** the learnt policy to larger instances of practical scales.
- Towards leveraging transfer learning to **solve large-scale TSPs**, this paper identifies inductive biases, model architectures and learning algorithms that promote generalization to instances larger than those seen in training. 
Our controlled experiments provide the first principled investigation into such **zero-shot generalization**, revealing that extrapolating beyond training data requires rethinking the entire neural combinatorial optimization pipeline, from network layers and learning paradigms to evaluation protocols.

## End-to-end Neural Combinatorial Optimization Pipeline

Towards a controlled study of **neural combinatorial optimization**, we unify several state-of-the-art architectures and learning paradigms into one experimental pipeline and provide the first principled investigation on zero-shot generalization to large instances.

![End-to-end neural combinatorial optimization pipeline](/img/pipeline.png)

1. **Problem Definition:** The combinatorial problem is formulated via a graph.
2. **Graph Embedding:** Embeddings for each graph node areobtained using a Graph Neural Network encoder.
3. **Solution Decoding:** Probabilities are assigned to each node for belonging to the solution set, either independent of one-another (i.e. Non-autoregressive decoding) or conditionally through graph traversal (i.e. Autoregressive decoding).
4. **Solution Search:** The predicted probabilities are converted intodiscrete decisions through classical graph search techniques such as greedy search or beam search.
5. **Policy Learning:** The entire model in trained end-to-end via imitating anoptimal solver (i.e. supervised learning) or through minimizing a cost function (i.e. reinforcement learning).

**We open-source our framework and datasets to encourage the community to go beyond evaluating performance on fixed TSP sizes, develop more expressive and scale-invariant GNNs, as well as study transfer learning for combinatorial problems.**

## Installation
We ran our code on Ubuntu 16.04, using Python 3.6.7, PyTorch 1.2.0 and CUDA 10.0. 
We highly recommend installation via Anaconda.

```sh
# Clone the repository. 
git clone https://github.com/chaitjo/learning-tsp.git
cd learning-tsp

# Set up a new conda environment and activate it.
conda create -n tsp python=3.6.7
source activate tsp

# Install all dependencies and Jupyter Lab (for using notebooks).
conda install pytorch=1.2.0 cudatoolkit=10.0 -c pytorch  
conda install numpy scipy cython tqdm scikit-learn matplotlib seaborn tensorboard pandas
conda install jupyterlab -c conda-forge
pip install tensorboard_logger

# Download datasets and unpack to the /data/tsp directory.
pip install gdown
gdown https://drive.google.com/uc?id=152mpCze-v4d0m9kdsCeVkLdHFkjeDeF5
tar -xvzf tsp-data.tar.gz ./data/tsp/
```


## Usage

For reproducing experiments, we provide a set of scripts for training, finetuning and evaluation in the `/scripts` directory. 
Pre-trained models for some experiments described in the paper can be found in the `/pretrained` directory.

Refer to `options.py` for descriptions of each option. 
High-level commands are as follows:
```sh
# Training
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python run.py 
    --problem <tsp/tspsl> 
    --model <attention/nar> 
    --encoder <gnn/gat/mlp> 
    --baseline <rollout/critic> 
    --min_size <20/50/100> 
    --max_size <50/100/200>
    --batch_size 128 
    --train_dataset data/tsp/tsp<20/50/100/20-50>_train_concorde.txt 
    --val_datasets data/tsp/tsp20_val_concorde.txt data/tsp/tsp50_val_concorde.txt data/tsp/tsp100_val_concorde.txt
    --lr_model 1e-4
    --run_name <custom_run_name>
    
# Evaluation
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python eval.py data/tsp/tsp10-200_concorde.txt
    --model outputs/<custom_run_name>_<datetime>/
    --decode_strategy <greedy/sample/bs> 
    --eval_batch_size <128/1/16>
    --width <1/128/1280>
```

