import pandas as pd
import plotly.graph_objects as go
import pathlib as pl
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import tempfile
from collections import Counter
from PIL import Image
from matplotlib.gridspec import GridSpec
import seaborn as sns
fs = 20
#plt.rcParams.update({'font.size': fs})

def plot_sankey(df, path, use_tempdir=False, tag=''):
    fixed_order = ['CR', 'PR', 'SD', 'PD']
    base_colors = {
        'CR': '#1f77b4',  # blue
        'PR': '#ff7f0e',  # orange
        'SD': '#2ca02c',  # green
        'PD': '#d62728'   # red
    }

    labels = ['N/A']  # single origin node
    values = [len(df)]
    label_lookup = {'T0': 0}
    node_colors = ['#aaaaaa']
    node_y = [0.5]  # center it vertically

    n_timepoints = len(df.columns)
    step_y = 1 / len(fixed_order)

    # Build the rest of the nodes
    for t_idx in range(n_timepoints):
        for i, cat in enumerate(fixed_order):
            label = f"T{t_idx+1}-{cat}"
            label_lookup[label] = len(labels)
            labels.append(cat)
            node_colors.append(base_colors.get(cat, "gray"))
            node_y.append(i * step_y)

    # Build flows (links)
    source = []
    target = []
    value = []
    link_colors = []

    # 🔹 T0 → T1 links
    first_tp = df.columns[0]
    first_counts = df[first_tp].value_counts().to_dict()
    for cat in fixed_order:
        if cat not in first_counts.keys():
            continue
        count = first_counts[cat]
        from_label = 'T0'
        to_label = f"T1-{cat}"
        if to_label in label_lookup:
            source.append(label_lookup[from_label])
            target.append(label_lookup[to_label])
            value.append(count)
            link_colors.append(base_colors.get(cat, 'gray'))


    # 🔹 T1 → T2, T2 → T3, ... regular links
    for i in range(n_timepoints - 1):
        from_tp = df.columns[i]
        to_tp = df.columns[i + 1]

        transitions = df.groupby([from_tp, to_tp]).size().reset_index(name='count')
        for _, row in transitions.iterrows():
            from_label = f"T{i+1}-{row[from_tp]}"
            to_label = f"T{i+2}-{row[to_tp]}"

            if from_label in label_lookup and to_label in label_lookup:
                source.append(label_lookup[from_label])
                target.append(label_lookup[to_label])
                value.append(row['count'])
                link_colors.append(base_colors.get(row[from_tp], "gray"))

    #labels = [f"{l}={value[i]}" for i, l in enumerate(labels)]


    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        textfont=dict(size=fs, color='black'),
        arrangement="fixed",  # allows decent auto-layout with T0
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            y=node_y
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    #fig.update_layout(title_text="RANO Flow Over Time", font_size=12)

    if not use_tempdir:
        fig.write_image(path /f"{tag}rano_flow_sankey_plot.svg", width=1000, height=600)
    else:
        temp_path = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        fig.write_image(temp_path, width=1000, height=600)
        return temp_path


def plot_cluster_centers(df, output_dir, data_cols, rano_cols, label_col="cluster", init_col="t0_volume"):
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = np.unique(df[label_col])
    print('found unique labels:', unique_labels)
    print(data_cols)
    c_centers = np.ones((len(unique_labels), len(data_cols)))
    print(c_centers.shape)
    idx = 0
    for label in unique_labels:
        c_centers[idx, :] = np.mean(df[df[label_col] == label][data_cols], axis=0)
        c_centers[idx, 0] = 1
        idx += 1

    idx = 0
    for i in unique_labels:
        cluster_meta = {}
        
        sub_cluster = df[df[label_col] == i]
        cluster_meta['#Members'] = len(sub_cluster)
        init_mean = np.mean(sub_cluster[init_col])
        init_std = np.std(sub_cluster[init_col])
        init_vol = sub_cluster[init_col]

        var = np.var(sub_cluster[data_cols], axis=0)
        var[0] = 0
        std = np.std(sub_cluster[data_cols], axis=0)
        std[0] = 0

        cluster_meta['Variances'] = var
        cluster_meta['StdDeviations'] = std
        cluster_meta['MemberIDs'] = list(sub_cluster['Lesion ID'])
        
        print('== plotting cluster:', i)
        x = data_cols
        y = c_centers[idx, :]

        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2,3, figure=fig)

        ax00 = fig.add_subplot(gs[0,:2])
        ax01 = fig.add_subplot(gs[:,2])

        # --- Left plot: trajectory ---
        ax00.errorbar(x, y, yerr=std, fmt='-o', color='blue', ecolor='black', capsize=5, label='Cluster Trajectory')
        ax00.set_xlabel('Elapsed time [days]')
        ax00.set_ylabel('Relative Change in Volume [%]')
        ax00.set_title(f'Trajectory of Cluster {i}')
        ax00.legend()
        ax00.grid(True)
        ax00.tick_params(axis='x', labelrotation=45) 

        # --- Right plot: initial volumes boxplot ---
        ax01.boxplot(init_vol, vert=True, patch_artist=True)
        ax01.set_title('Boxplot of initial Volume')
        ax01.set_ylabel('Volume [mm³]')
        ax01.grid(True)

        ax10 = fig.add_subplot(gs[1,:2])

        # --- bottom left plot sankey flow ---
        sankey = plot_sankey(sub_cluster[rano_cols], None, True)
        ax10.imshow(Image.open(sankey))
        os.remove(sankey.name)
        ax10.axis('off')


        # --- Shared plot settings ---
        fig.suptitle(f"""Members = {cluster_meta["#Members"]}, Init Vol = {init_mean:.1f} ± {init_std:.1f} [mm³]""", fontsize=16)
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


def plot_combined_trajectories(df, data_cols, label_col, path):
    """
    Creates a scatter plot of all trajectories in the set; trajectories of the same cluster share a color.
    Visualization is clipped at 2 on Y for visibility.
    Each cluster has its average trajectory drawn as a line plot.
    Intended for direct comparison of clusters in one figure.
    """
    os.makedirs(path, exist_ok=True)

    df.loc[:, '0'] = 1
    data_cols = ['0']+data_cols
 

    unique_labels = np.unique(df[label_col])
    print('found unique labels:', unique_labels)

    colors = sns.color_palette("hls", len(unique_labels))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # Last one is a custom dash-dot-dot pattern

    plt.figure(figsize=(12, 8))
    cluster_dicts = {}
    sizes = []
    # Plot individual points
    for i, label in enumerate(unique_labels):
        cluster_df = df[df[label_col] == label]
        curr_dict = {}
        curr_dict['avg_end_size'] = cluster_df[data_cols].mean(axis=0)[-1]
        curr_dict['label'] = label
        sizes.append(cluster_df[data_cols].mean(axis=0)[-1])
        cnt = Counter(cluster_df['t6_rano'])
        curr_dict = {**curr_dict, **cnt}
        print(curr_dict)
        cluster_dicts[cluster_df[data_cols].mean(axis=0)[-1]] = curr_dict

        #for _, row in cluster_df.iterrows():
            #plt.scatter(['0', '60', '120', '180', '240', '300', '360'], row[data_cols], color=colors[i], alpha=0.3, s=10)

    ordering = {v:k for k, v in enumerate(sorted(sizes))}
    print(ordering)

    # Plot average trajectory per cluster
    for i, v in enumerate(sorted(sizes)):
        print(v)
        label = cluster_dicts[v]['label']
        print(label)
        #clustid=ordering[cluster_dicts[label]['avg_end_size']]
        cluster_df = df[df[label_col] == label]
        mean_traj = cluster_df[data_cols].mean(axis=0)
        
        make_cluster_pie_chart(cluster_dicts[v], path/f'pie_chart_clust_{i}.png', f'Cluster {i}')

        plt.plot(['0', '60', '120', '180', '240', '300', '360'], mean_traj*100, linestyle=linestyles[i], color='gray', linewidth=2.5, label=f'Cluster {i} = {len(cluster_df)}')
        # for x, y in zip(['0', '60', '120', '180', '240', '300', '360'], mean_traj):
        #     if y > 16:
        #         plt.text(x, 16, f'{y:.1f}', fontsize=10, ha='center', va='top', color=colors[i], rotation=90, clip_on=False)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylim(-0.1, 16)
    plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600])
    #plt.yscale('log')
    plt.xlabel("Time after treatment [days]", fontsize=16)
    plt.ylabel("Relative volume to treatment volume [%]", fontsize=16)
    #plt.title("Combined Trajectories by Cluster", fontsize=20)
    plt.legend(loc="upper left", fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    out_file = os.path.join(path, "combined_trajectories.svg")
    plt.savefig(out_file)
    plt.close()

def make_cluster_pie_chart(data, path, title):
    response_data = {k: v for k, v in data.items() if k in ['CR', 'PR', 'SD', 'PD']}

    # Extract labels and sizes
    labels = list(response_data.keys())
    sizes = list(response_data.values())

    base_colors = {
        'CR': '#1f77b4',  # blue
        'PR': '#ff7f0e',  # orange
        'SD': '#2ca02c',  # green
        'PD': '#d62728'   # red
    }
    def autopct_func(pct):
        return ('%1.1f%%' % pct) if pct > 5 else ''
    # Match colors to the labels
    colors = [base_colors[label] for label in labels]
    # Create pie chart
    # Create local figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    explode = [0.05 if v < 5 else 0 for v in sizes]
    ax.pie(sizes, autopct='', startangle=90, colors=colors, textprops={'fontsize': 20})
    ax.axis('equal')  # Equal aspect ratio ensures pie is a circle.
    fig.suptitle(title, fontsize=14)
    # Save to file and close
    fig.savefig(path)
    plt.close(fig)

def plot_recur_probs(df, rano_cols, label_col, path):
    categories = ['CR', 'PR', 'PD', 'SD']

    results = {}
    for cat in categories:
        print('= Analyzing category:', cat)
        subset = df[df[rano_cols[0]]==cat]
        ref = len(subset)
        distribution = {}
        sub = {}
        for c in categories:
            if c == cat:
                sub[c] = ref
            else: sub[c] = 0
        distribution['t1'] = sub

        print(f'= {cat} has {ref} instances at t1 = {rano_cols[0]} [days]')
        for i in range(1, len(rano_cols)):
            sub = {}
            for c in categories:
                sub[c] = len(subset[subset[rano_cols[i]] == c])
            subset = subset[subset[rano_cols[i]] == cat]
            print(f'== {cat} still has {sub[cat]} instances at t{i+1} = {rano_cols[i]} [days]')
            distribution[f"t{i+1}"] = sub
        
        results[cat] = distribution

    example_data = next(iter(results.values()))
    timepoints = list(example_data.keys())
    n_timepoints = len(timepoints)
    
    x = np.arange(n_timepoints)
    bar_width = 0.2

    fig, ax = plt.subplots(len(categories), 1, figsize=(14, 12), sharex=True, sharey=True)

    all_bars = []

    for idx, cat in enumerate(categories):
        counts_per_tp = []
        data = results[cat]
        base_val = max(data['t1'][cat], 1e-6)  # normalizing against t1 for this category

        for tp in timepoints:
            counts_per_tp.append(np.asarray(list(data[tp].values())).sum())

        prob_matrix = []
        for c in categories:
            
            probs = [data[timepoints[0]][c] / base_val]
            for i, tp in enumerate(timepoints):
                if i != 0:
                    ratio = data[tp][c] / data[timepoints[i-1]][cat] if data[timepoints[i-1]][cat]>0 else 0
                    probs.append(ratio)
            prob_matrix.append(probs)


        for j, c in enumerate(categories):
            bars = ax[idx].bar(x + j * bar_width, prob_matrix[j], width=bar_width, label=c if idx == 0 else "")
            if idx == 0:
                all_bars.append(bars[0])

        # Annotate with number of lesions
        for j, n_lesions in enumerate(counts_per_tp):
            ax[idx].text(
                x[j] + bar_width * 2,  # center under the bars
                -0.05,  # just below x-axis
                f"n={n_lesions}",
                ha='center',
                va='top',
                fontsize=10
            )

        ax[idx].set_title(f"Probabilities given category {cat}")
        ax[idx].grid(axis='y', linestyle='--', alpha=0.5)

    ax[-1].set_xticks(x + bar_width * 1.5)
    ax[-1].set_xticklabels([f"\n{tp} ({(i+1)*60}d)" for i, tp in enumerate(timepoints)])

    fig.suptitle("Recurrence Probabilities by Initial Category", fontsize=18)
    fig.text(0.04, 0.5, "Category change probability after spending all previous timepoints in the same Category", va='center', rotation='vertical', fontsize=14)
    fig.legend(all_bars, categories, loc='upper right', fontsize=12)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    os.makedirs(path, exist_ok=True)
    out_file = os.path.join(path, "probabilities.png")
    fig.savefig(out_file)
    plt.close(fig)


def plot_recur_probs_noncausal(df, rano_cols, path):
    categories = ['CR', 'PR', 'PD', 'SD']

    # Define consistent colors per category
    category_colors = {
        'CR': '#1f77b4',  # blue
        'PR': '#ff7f0e',  # orange
        'PD': '#2ca02c',  # green
        'SD': '#d62728',  # red
    }

    # Initialize result dict
    results = {cat: {f"t{i+1}": {c: 0 for c in categories} for i in range(6)} for cat in categories}

    # Fill result dict
    for _, row in df[rano_cols].iterrows():
        row = row.to_list()
        ref = None
        count = 0
        for idx, value in enumerate(row):
            if count == 0:
                ref = value
                count += 1
                results[ref][f"t{count}"][value] += 1
            elif ref == value:
                count += 1
                results[ref][f"t{count}"][value] += 1
            else:
                results[ref][f"t{count+1}"][value] += 1
                ref = value
                count = 0

    # Plotting
    timepoints = list(next(iter(results.values())).keys())
    n_timepoints = len(timepoints)
    x = np.arange(n_timepoints)
    bar_width = 0.2

    fig, ax = plt.subplots(len(categories), 1, figsize=(14, 12), sharex=True, sharey=True)
    all_bars = []

    for idx, cat in enumerate(categories):
        prob_matrix = []
        counts_per_tp = []  # for n=

        data = results[cat]
        for tp in timepoints:
            probs = np.asarray(list(data[tp].values()))
            n = probs.sum()
            counts_per_tp.append(n)
            probs = list(probs / n) if n > 0 else [0] * len(categories)
            prob_matrix.append(probs)

        # Plot bars with consistent colors
        for j, tp in enumerate(timepoints):
            probs = prob_matrix[j]
            for k, c in enumerate(categories):
                bar = ax[idx].bar(
                    x[j] + k * bar_width,
                    probs[k],
                    width=bar_width,
                    color=category_colors[c],
                    label=c if idx == 0 and j == 0 else ""
                )
                if idx == 0 and j == 0:
                    all_bars.append(bar[0])

        # Annotate with number of lesions
        for j, n_lesions in enumerate(counts_per_tp):
            ax[idx].text(
                x[j] + bar_width * 2,  # center under the bars
                -0.05,  # just below x-axis
                f"n={n_lesions}",
                ha='center',
                va='top',
                fontsize=10
            )

        ax[idx].set_title(f"Probabilities given category {cat}")
        ax[idx].grid(axis='y', linestyle='--', alpha=0.5)

    # X-axis ticks
    ax[-1].set_xticks(x + bar_width * 1.5)
    ax[-1].set_xticklabels([f"\n{tp} ({(i+1)*60}d)" for i, tp in enumerate(timepoints)])

    # Global labels
    fig.suptitle("Recurrence Probabilities by Initial Category", fontsize=18)
    fig.text(0.04, 0.5, "Category change probability after spending all previous timepoints in the same Category", 
             va='center', rotation='vertical', fontsize=14)
    fig.legend(all_bars, categories, loc='upper right', fontsize=12)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # Save figure
    os.makedirs(path, exist_ok=True)
    out_file = os.path.join(path, "probabilities_noncausal.png")
    fig.savefig(out_file)
    plt.close(fig)
