import neptune.new as neptune
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np

from utils.constants import API_TOKEN
from q2_lipschitzness import SAMPLES

FONT_SIZE = 40
mpl.rcParams['font.size'] = FONT_SIZE
mpl.rcParams['font.family'] = 'sans-serif'

# Load data from Neptune
project = neptune.init_project(project="CoveredForestsArxiv/Q2Lischpitz", api_token=API_TOKEN)
df = project.fetch_runs_table().to_pandas()
df['sys/creation_time'] = pd.to_datetime(df['sys/creation_time'])
df = df[df['params/distance_type'] == 'DistanceType.FD']
latest_indices = df.groupby(['params/graph_family', 'params/num_layers'])['sys/creation_time'].idxmax()
grouped_results = df.loc[latest_indices].reset_index(drop=True)

for _, row in grouped_results.iterrows():
    # recover the data
    graph_family = row['params/graph_family'].split('.')[1].upper()
    mpngraph_size_list = []
    graph_distance_list = []
    for sample in range(SAMPLES):
        mpngraph_size_list.append(row[f'mpnn_{sample}'])
        graph_distance_list.append(row[f'stochastic_{sample}'])

    correlation, _ = pearsonr(graph_distance_list, mpngraph_size_list)

    # prepare the folders
    plot_dir = os.path.join(os.getcwd(), '..', '..', 'plots', 'FD', graph_family)
    os.makedirs(plot_dir, exist_ok=True)

    # plot
    fig, ax1 = plt.subplots(figsize=(10, 8))  # Consistent figure size
    if graph_family == 'MOLHIV':
        ax2 = ax1.twinx()
        ax2.set_ylabel(f'{row["params/num_layers"]} layers', fontsize=FONT_SIZE)
        ax2.set_yticks([])
    else:
        ax2 = None
    num_layers = row['params/num_layers']
    ax1.scatter(graph_distance_list, mpngraph_size_list)
    ax1.set_xlabel(fr'$\mathrm{{FD}}_{{{num_layers}}}$', fontsize=FONT_SIZE)
    graph_family = graph_family.replace('_', '-')
    if graph_family == 'MUTAG':
        ax1.set_ylabel(r'$||h(G_1) - h(G_2)||_2$', fontsize=FONT_SIZE)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    if num_layers == 0:
        ax1.set_title(f'{graph_family}', fontsize=FONT_SIZE)
    ax1.text(
        0.35, 0.85,
        f'r = {correlation: .2f}',
        fontsize=FONT_SIZE,
        transform=ax1.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right'
    )
    fig.subplots_adjust(bottom=0.2, left=0.15)
    plt.savefig(os.path.join(plot_dir, f"{num_layers}layers_lipchitz.jpg"))
    plt.close()

    if num_layers == 3:
        x_values = np.array(graph_distance_list)
        mask = x_values != 0
        x_values = x_values[mask]
        y_values = np.array(mpngraph_size_list)[mask]
        slope = np.max(y_values / x_values)
        print(f'graph_family: {graph_family}, SLOPE: {slope}')
