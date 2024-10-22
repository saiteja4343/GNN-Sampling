# Sampling in GNNs - Implementation

This repository is the implementation of our project which is based on how the neighborhood  sampling affects the metrics in the GNNs with a focus on GraphSAGE. 

The report can be found [here](https://drive.google.com/file/d/1lvz6ZNK7Kl5nxZ7xkyX3n8XVPmCPYcC-/view?usp=sharing):


> **Abstract:** *GNNs (Graph Neural Networks) have become essential in the field of machine learning for learning graph-structured data. They have
been employed in different areas from social network analysis to molecular property prediction. Among these, GraphSAGE is the
most popular, which can learn inductively and scale to large graphs. However, the influence of different hyperparameters on the
performance of GraphSAGE, especially the sample layer sizes, has still been a topic to be researched upon. This study investigates the
impact of varying sample layer sizes on the performance of GraphSAGE, a prominent inductive framework for graph neural networks.
We have assumed that the neighbour layer sample sizes would significantly affect model metrics that are correlated with accuracy
and runtime efficiency. Our findings, however, were contrary to expectations, as we found that the changes in sample sizes did not
uniformly affect these metrics across different datasets. We gradually varied the sample layer sizes and studied their impact on
model performance, discovering that GraphSAGE’s performance is stable to such variations. This flexibility is indicative of the
fact that practitioners should not focus on the hyperparameters that are of less importance and may be overridden but rather should
concentrate on the ones that really count, e.g. learning rates and aggregation functions, that could make a more notable difference
to the model performance. Our study contributes to the research on graph neural networks and gives insights into the factors
influencing GraphSAGE’s performance. The recommendations present valuable information for the researchers and professionals
who are working on the application of GraphSAGE to various graph-based tasks. The findings will have a positive impact on the
development of more effective and efficient graph-learning models.*

## Data

The datasets which are used for the experiments in the project are taken from [DGL](https://docs.dgl.ai/en/1.1.x/api/python/dgl.data.html#node-prediction-datasets) and [OGB](https://ogb.stanford.edu/docs/nodeprop/) libraries respectively 

## Structure of Repository

The repository is structured with the files as follows:

* **train_exp.py**: This file contains the code for training the GraphSAGE model with different sample sizes and different datasets and storing the metrics using MLFlow.

* **metrics_plots.ipynb**: This file contains the code for plotting the metrics obtained from the MLFlow.
* **requirements.txt**: This file contains the dependencies required for the project.

## Installation
 Please install the dependencies using the following command:

```pip install -r requirements.txt ```


## Hardware used for the experiments

The experiments were conducted on a machine with the following specifications:

* **CPU**: Intel(R) Core(TM) i9-12900H CPU @ 2.50GHz
* **GPU**: NVIDIA GeForce RTX 3060 GPU with 6GB VRAM
* **RAM**: 16 GB



