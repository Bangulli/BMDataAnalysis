import pandas as pd
import plotly.graph_objects as go
import pathlib as pl
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import tempfile
from PIL import Image
from matplotlib.gridspec import GridSpec

def plot_sankey(df, path, use_tempdir=False):
    # Get all unique labels across timepoints, with time info
    labels = []
    label_lookup = {}
    for t_idx, col in enumerate(df.columns):
        for label in df[col].unique():
            node_name = f"{label} (T{t_idx+1})"
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
    if not use_tempdir:
        fig.write_image(path/"rano_flow_sankey_plot.png", width=1000, height=600)
    else:
        temp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.write_image(temp_path, width=1000, height=600)
        return temp_path


def plot_cluster_centers(df, output_dir, data_cols, rano_cols, label_col="cluster", init_col="0"):
    os.makedirs(output_dir, exist_ok=True)

    unique_labels = np.unique(df[label_col])
    print('found unique labels:', unique_labels)
    c_centers = np.zeros((len(unique_labels), len(data_cols)))
    idx = 0
    for label in unique_labels:
        c_centers[idx, :] = np.mean(df[df[label_col] == label][data_cols], axis=0)
        idx += 1

    idx = 0
    for i in unique_labels:
        cluster_meta = {}
        
        sub_cluster = df[df[label_col] == i]
        cluster_meta['#Members'] = len(sub_cluster)
        var = np.var(sub_cluster[data_cols], axis=0)
        std = np.std(sub_cluster[data_cols], axis=0)
        init_vol = np.mean(sub_cluster[init_col])
        init_std = np.std(sub_cluster[init_col])
        cluster_meta['Variances'] = var
        cluster_meta['StdDeviations'] = std
        cluster_meta['MemberIDs'] = list(sub_cluster['Lesion ID'])
        
        print('== plotting cluster:', i)
        x = [int(elem) for elem in data_cols]
        y = c_centers[idx, :]

        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2,2, figure=fig)

        ax00 = fig.add_subplot(gs[0,0])
        ax01 = fig.add_subplot(gs[0,1])

        # --- Left plot: trajectory ---
        ax00.errorbar(x, y, yerr=std, fmt='-o', color='blue', ecolor='black', capsize=5, label='Cluster Trajectory')
        ax00.set_xlabel('Elapsed time [days]')
        ax00.set_ylabel('Relative Change in Volume [%]')
        ax00.set_title(f'Trajectory of Cluster {i}')
        ax00.legend()
        ax00.grid(True)

        # --- Right plot: categorical distributions ---
        categorical_features = ['CR', 'PR', 'SD', 'PD']
        counts = [np.sum(sub_cluster[rano_cols[-1]] == feat) for feat in categorical_features]
        print(counts)

        # Bar plot layout
        for j, (feature, count_series) in enumerate(zip(categorical_features, counts)):
            ax01.barh(
                f"{feature}: {count_series}",
                count_series,
                label=feature
            )

        ax01.set_title('1 Year RANO Response Distribution [volumetric]')
        ax01.legend()
        ax01.grid(True)

        ax10 = fig.add_subplot(gs[1,:])

        # --- bottom left plot sankey flow ---
        sankey = plot_sankey(sub_cluster[rano_cols], None, True)
        ax10.imshow(Image.open(sankey))
        os.remove(sankey.name)
        ax10.axis('off')


        # --- Shared plot settings ---
        fig.suptitle(f"Members = {cluster_meta["#Members"]}, Init Vol = {init_vol:.1f} ± {init_std:.1f} [mm³]", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        fig.savefig(output_dir / f'trajectory_cluster_{i}.png')
        plt.clf()


        idx += 1

def plot_tsne(df, data_cols, path):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_embedded = tsne.fit_transform(df[data_cols])  # X = your high-dimensional data

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=df['cluster'], cmap='tab10')
    plt.title("t-SNE Visualization")
    plt.legend()
    plt.savefig(path/'tSNE_cluster_visualization.png')
    plt.clf()

def plot_umap(df, data_cols, path):
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap_model.fit_transform(df[data_cols])  # X = your high-dimensional data

    # Plot
    plt.figure(figsize=(8, 6))
    unique_clusters = np.unique(df['cluster'])
    for cluster in unique_clusters:
        indices = df['cluster'] == cluster
        plt.scatter(
            X_umap[indices, 0],
            X_umap[indices, 1],
            label=str(cluster),
            s=10
        )

    plt.title('UMAP Visualization')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.grid(True)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(path / 'UMAP_cluster_visualization.png')
    plt.clf()