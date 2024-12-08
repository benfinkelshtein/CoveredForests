# Covered Forests #

This repository contains the official code base of the paper "Covered Forests: Fine-grained generalization analysis of graph neural networks".

## Installation ##
To reproduce the results please use Python 3.9, PyTorch version 2.4.1, Cuda 12.1 and PyG .

```bash
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric
pip install matplotlib
pip install cvxpy
pip install torchmetrics
pip install neptune
```

## Running

We use the following scripts to save the experimental results onto neptune:
- ``q1_real_graphs_cover.py``: Calculates the covering number of our unique graph classes for varying graph sizes and radii. 
- ``q1_unique_graphs_cover.py``: Calculates the covering number of our real-world datasets for varying graph sizes and radii. 
- ``q2_lipschitzness.py``: Samples graphs from real-world datasets and calculates the Forest distance and the distance between MPNN outputs.
- ``q3_generalization_gap.py``: Trains a model over real-world datasets and calculates the gap and ``$m_{n,d,L}$``.

We use the following scripts to plots the experimental results:
- ``q1_plot_cover.py``: Loads the experimental results of q1_real_graphs_cover.py or q1_unique_graphs_cover.py and plots Figures 4 and 5.
- ``q1_plot_bounds.py``: Loads the experimental results of q1_real_graphs_cover.py and plots Figures 6.
- ``q2_plot_lipschitzness.py``: Loads the experimental results of q2_lipschitzness.py and plots Figures 7.
- ``q3_print_generalization_gap.py``: Loads the experimental results of q3_generalization_gap.py and ouputs the first 3 rows of Table 1.
- ``q3_print_generalization_bound.py``: Outputs the last 2 rows of Table 1.

Make sure to fill in your API token as a string in utils.constants under the name ``API_TOKEN``.

Note that the script should be run with the repository being the main directory or source root.

The names of parameters in the scripts are self-explanatory.

## Example running

```bash
python -u q1_real_graphs_cover.py --graph_family MUTAG --distance_type FD --graph_size_list 15 20 25 --radius_list 4 8 12 16 20
python -u q1_plot_cover.py 
```
