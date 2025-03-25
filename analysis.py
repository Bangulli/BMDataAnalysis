import pandas as pd
import sklearn
from sklearn.cluster import MeanShift, KMeans, DBSCAN, HDBSCAN, OPTICS
import core
import plotly.graph_objects as go
import pathlib as pl
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def explore_cluster_centers(cluster_object, df, output_dir, data_cols):
    os.makedirs(output_dir, exist_ok=True)
    try:
        c_centers = cluster_object.cluster_centers_
    except:
        unique_labels = np.unique(cluster_object.labels_)
        print('found unique labels:', unique_labels)
        c_centers = np.zeros((len(unique_labels), len(data_cols)))
        idx = 0
        for label in unique_labels:
            c_centers[idx, :] = np.mean(df[df['cluster'] == label][data_cols], axis=0)
            idx += 1

    idx = 0
    for i in np.unique(cluster_object.labels_):
        cluster_meta = {}
        
        sub_cluster = df[df['cluster'] == i]
        cluster_meta['#Members'] = len(sub_cluster)
        var = np.var(sub_cluster[data_cols], axis=0)
        std = np.std(sub_cluster[data_cols], axis=0)
        cluster_meta['Variances'] = var
        cluster_meta['StdDeviations'] = std
        cluster_meta['MemberIDs'] = list(sub_cluster['Lesion ID'])
        
        print('== plotting cluster:', i)
        x = [int(elem) for elem in data_cols]
        y = c_centers[idx, :]

        plt.figure(figsize=(8,5))
        plt.errorbar(x, y, yerr=std, fmt='-o', color='blue', ecolor='black', capsize=5, label='Cluster Trajectory')


        plt.xlabel('Elapsed time [days]')
        plt.ylabel('Relative Change in Volume [%]')
        plt.title(f'Trajecotry of Cluster {i}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir/f'trajecotry_cluster_{i}.png')
        print(cluster_meta['#Members'])
        plt.clf()

        idx += 1

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_embedded = tsne.fit_transform(df[data_cols])  # X = your high-dimensional data

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=df['cluster'], cmap='tab10')
    plt.title("t-SNE Visualization")
    plt.savefig(output_dir/'tSNE_cluster_visualization.png')
    plt.clf()

    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap_model.fit_transform(df[data_cols])  # X = your high-dimensional data

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=df['cluster'], cmap='tab10', s=10)
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir/'UMAP_cluster_visualization.png')
    plt.clf()

def make_sankey(df, path):
    # Get all unique labels across timepoints, with time info
    labels = []
    label_lookup = {}
    for t_idx, col in enumerate(df.columns):
        for label in df[col].unique():
            node_name = f"{label} (T{t_idx})"
            if node_name not in label_lookup:
                label_lookup[node_name] = len(labels)
                labels.append(node_name)

    # Build the flows: between T0→T1, T1→T2, etc.
    source = []
    target = []
    value = []

    timepoints = df.columns
    for i in range(len(timepoints) - 1):
        from_tp = timepoints[i]
        to_tp = timepoints[i + 1]

        transition_counts = df.groupby([from_tp, to_tp]).size().reset_index(name='count')
        for _, row in transition_counts.iterrows():
            source_name = f"{row[from_tp]} (T{i+1})"
            target_name = f"{row[to_tp]} (T{i+2})"
            source.append(label_lookup[source_name])
            target.append(label_lookup[target_name])
            value.append(row['count'])

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightblue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text="RANO Flow Over Time", font_size=12)
    fig.write_image(path/"rano_flow_sankey_plot.png", width=1000, height=600)

if __name__ == '__main__':
    output =  pl.Path('/home/lorenz/BMDataAnalysis/output/csv_linear_multiclass_reseg/hdbscan2')
    ## load volume data
    volume_data = pd.read_csv('/mnt/nas6/data/Target/PARSED_METS_mrct1000_nobatch/csv_linear_multiclass_reseg/volumes.csv', index_col=None)
    data_cols = ["60", "120", "180", "240", "300", "360"]
    print(f'clustering {len(volume_data)} metastases')
    
    # Normalize by t0 Volume
    volume_data[data_cols] = volume_data[data_cols].div(volume_data["0"], axis=0)
    ## Do the clustering
    cluster = HDBSCAN(min_cluster_size=10) # basically use it as a outlier rejector
    cluster.fit(volume_data[data_cols])
    volume_data['cluster'] = cluster.labels_


    DB_score = sklearn.metrics.davies_bouldin_score(volume_data[data_cols], volume_data['cluster'])
    print(f'Clustering achieved a Davies-Bouldin score of {DB_score}')

    ## Make plots for cluster
    explore_cluster_centers(cluster, volume_data, output, data_cols)

    ## load rano data
    rano_data = pd.read_csv('/mnt/nas6/data/Target/PARSED_METS_mrct1000_nobatch/csv_linear_multiclass_reseg/rano.csv', index_col=None)
    make_sankey(rano_data[data_cols], output)




