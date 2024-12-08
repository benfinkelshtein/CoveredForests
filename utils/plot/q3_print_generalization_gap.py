import neptune.new as neptune
import pandas as pd
from collections import defaultdict

from utils.constants import API_TOKEN

# Load data from Neptune
project = neptune.init_project(project="CoveredForestsArxiv/Q3Generalization", api_token=API_TOKEN)
df = project.fetch_runs_table().to_pandas()
df['sys/creation_time'] = pd.to_datetime(df['sys/creation_time'])
latest_indices = df.groupby(['params/graph_family'])['sys/creation_time'].idxmax()
grouped_results = df.loc[latest_indices].reset_index(drop=True)

# Initialize results dictionary for storing table data
results = defaultdict(lambda: defaultdict(int))
cover_r_fields = [col for col in df.columns if col.startswith('cover_r_')]
for _, row in grouped_results.iterrows():
    graph_family = row['params/graph_family']
    graph_family = graph_family.split('.')[1]  # Simplify the graph family name

    # recover the data
    results[graph_family]['train_loss_mean'] = row['train_loss_mean']
    results[graph_family]['test_loss_mean'] = row['test_loss_mean']
    results[graph_family]['diff_loss_mean'] = row['diff_loss_mean']
    results[graph_family]['train_loss_std'] = row['train_loss_std']
    # results[graph_family]['test_loss_std'] = row['test_loss_std']
    results[graph_family]['diff_loss_std'] = row['diff_loss_std']
    results[graph_family]['m_n_d_3'] = row['m_n_d_3']
    results[graph_family]['max_nodes'] = row['max_nodes']
    for field in cover_r_fields:
        results[graph_family][field] = row[field]

    # Prepare transposed table
    row_names = ['train_loss', 'test_loss', 'diff_loss', 'm_n_d_3', 'max_nodes']
    row_names += [f"{field}" for field in cover_r_fields]
    columns = list(results.keys())
    transposed_data = []
    for row_name in row_names:
        row = []
        for graph_family in columns:
            if 'loss' in row_name:
                if 'std' in row_name:
                    continue
                value = f"{results[graph_family][row_name + '_mean']:.3f} ± {results[graph_family][row_name + '_std']:.3f}"
            else:  # cover_..., m_n_d_3 or max_nodes
                value = f"{results[graph_family][row_name]:.3f}"
            row.append(value)
        transposed_data.append(row)

# Convert table to DataFrame for pretty display
pd.set_option('display.max_columns', None)
transposed_df = pd.DataFrame(transposed_data, index=row_names, columns=columns)
print(transposed_df)
