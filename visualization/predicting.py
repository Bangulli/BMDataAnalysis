import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def plot_prediction_metrics_sweep(results, output, next_value_prefix='nxt', one_year_prefix='1yr', classes='auto', distribution=None):
    os.makedirs(output, exist_ok=True)

    classification_metrics = ['balanced_accuracy', 'accuracy', 'f1', 'recall', 'precision']

    def make_grouped_metric_plot(data_dicts, keys, prefix, title):
        metrics = [m for m in data_dicts[0].keys() if m in classification_metrics]
        num_metrics = len(metrics)
        num_keys = len(keys)

        bar_width = 0.8 / num_metrics
        x = np.arange(num_keys)
        colors = plt.cm.tab10.colors

        plt.figure(figsize=(max(8, num_keys * 1.5), 6))
        for i, metric in enumerate(metrics):
            values = [d[metric] for d in data_dicts]
            bars = plt.bar(x + i * bar_width, values, width=bar_width, label=metric, color=colors[i % len(colors)])

            # Annotate each bar with its value
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha='center',
                    va='top',
                    fontsize=10,
                    rotation=90  # Optional: rotate for better fit
                )

        plt.xticks(x + bar_width * (num_metrics - 1) / 2, keys, rotation=45, ha='right')
        plt.xlabel("Input timepoints")
        plt.ylabel("Metric Value")
        plt.title(f"{title}")
        plt.legend(loc='lower right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output, f"{prefix}_grouped_metrics.png"))
        plt.close()

    def plot_confusions(matrices, prefix):
        fig = plot_confusion_matrices(matrices, classes, title_prefix=f'''Confusion Matrix''')# - Class Distribution = {distribution if distribution is not None else 'unknown'}''')
        fig.savefig(os.path.join(output, f"{prefix}_confusion_matrices.png"))
        plt.close(fig)
        plt.clf()

    # --- Next Value Prediction ---
    if next_value_prefix:
        nxt_keys = [k for k in results if k.startswith(next_value_prefix)]
        nxt_data = [results[k] for k in nxt_keys]
        if nxt_data:
            make_grouped_metric_plot(nxt_data, nxt_keys, "next_value", "Next Value Prediction Metrics")
            if 'confusion_matrix' in nxt_data[0]:
                matrices = [d['confusion_matrix'] for d in nxt_data]
                plot_confusions(matrices, "next_value")

    # --- One Year Prediction ---
    yr_keys = [k for k in results if k.startswith(one_year_prefix)]
    yr_data = [results[k] for k in yr_keys]
    if yr_data:
        make_grouped_metric_plot(yr_data, yr_keys, "one_year", f"""One-Year Prediction Metrics - Class Distribution = {distribution if distribution is not None else 'unknown'}""")
        if 'confusion_matrix' in yr_data[0]:
            matrices = [d['confusion_matrix'] for d in yr_data]
            plot_confusions(matrices, "one_year")

def plot_confusion_matrices(conf_matrices, class_labels='auto', title_prefix=f'Confusion Matrix'):
    """
    Plots confusion matrices in subplots using seaborn heatmaps.
    
    Parameters:
    - conf_matrices: list of numpy arrays returned by sklearn.metrics.confusion_matrix
    - class_labels: list of class labels (optional)
    - title_prefix: title prefix for each subplot
    """
    num_matrices = len(conf_matrices)
    if num_matrices == 0:
        print("No confusion matrices to plot.")
        return
    
    # Compute number of subplot rows/cols (make it as square as possible)
    cols = int(np.ceil(np.sqrt(num_matrices)))
    rows = int(np.ceil(num_matrices / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case it's 2D or 1D
    
    for idx, cm in enumerate(conf_matrices):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='.4f', cmap='Blues', cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f"{title_prefix} {idx + 1}")
    
    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def plot_prediction_metrics(result, output):
    values = []
    xticks = []
    for k, v in result.items():
        if k in ['balanced_accuracy', 'accuracy', 'f1', 'recall', 'precision']:
            values.append(v)
            xticks.append(k)
    plt.bar(xticks, values)
    plt.xlabel("Metric")
    plt.ylabel(f"Value")
    plt.title(f"Evaluation Metrics")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output/f"eval.png")
    plt.close()
    plt.clf()
    fig = plot_confusion_matrices([result['confusion_matrix']])
    fig.savefig(output/f"confusion_matrix.png")
    fig.clf()
    plt.close()
    plt.clf()

def plot_regression_metrics(result, output, tag=''):
    values = []
    xticks = []
    for k, v in result.items():
        values.append(v)
        xticks.append(k)
    plt.bar(xticks, values)
    plt.xlabel("Metric")
    plt.ylabel(f"Value")
    plt.title(f"Evaluation Metrics")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output/f"{tag}_eval.png")
    plt.close()
    plt.clf()

def plot_regression_metrics_sweep(results, output, tag='', next_value_prefix='nxt', one_year_prefix='1yr'):
    """
    Ive had gpt change my function. I didnt know it was possible to define a func inside a func. wtf is this???
    """
    os.makedirs(output, exist_ok=True)

    def make_grouped_bar_plot(data_dicts, prefix, title):
        # Extract keys and metrics
        keys = [k for k in results if k.startswith(prefix)]
        metrics = list(data_dicts[0].keys())
        num_metrics = len(metrics)
        num_keys = len(keys)

        # Prepare plot data
        bar_width = 0.8 / num_metrics  # to keep bars in a group tightly packed
        x = np.arange(num_keys)
        colors = plt.cm.tab10.colors  # use standard color set

        plt.figure(figsize=(max(8, num_keys * 1.5), 6))

        for i, metric in enumerate(metrics):
            if metric.endswith('-sep'): continue
            values = [d[metric] for d in data_dicts]
            bars = plt.bar(x + i * bar_width, values, width=bar_width, label=metric, color=colors[i % len(colors)])

            # Annotate each bar with its value
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha='center',
                    va='top',
                    fontsize=8,
                    rotation=90  # Optional: rotate for better fit
                )

        # X-ticks in the middle of the grouped bars
        plt.xticks(x + bar_width * (num_metrics - 1) / 2, [k.replace('_volume', '') for k in keys], rotation=45, ha='right')
        plt.xlabel("Feature/Target Configuration")
        plt.ylabel("Metric Value")
        plt.title(f"{title} ({tag})" if tag else title)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(output, f"{prefix}_grouped_metrics.png"))
        plt.close()

        plt.figure(figsize=(max(8, num_keys * 1.5), 6))

        for i, metric in enumerate(metrics):
            if not metric.endswith('-sep'): continue
            values = [d[metric] for d in data_dicts]
            bars = plt.bar(x + i * bar_width, values, width=bar_width, label=metric, color=colors[i % len(colors)])

            # Annotate each bar with its value
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha='center',
                    va='top',
                    fontsize=8,
                    rotation=90  # Optional: rotate for better fit
                )

        # X-ticks in the middle of the grouped bars
        plt.xticks(x + bar_width * (num_metrics - 1) / 2, [k.replace('_volume', '') for k in keys], rotation=45, ha='right')
        plt.xlabel("Feature/Target Configuration")
        plt.ylabel("Metric Value")
        plt.title(f"{title} ({tag})" if tag else title)
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(output, f"{prefix}_grouped_metrics_sep.png"))
        plt.close()

    # Prepare data
    nxt_keys = [k for k in results if k.startswith(next_value_prefix)]
    one_year_keys = [k for k in results if k.startswith(one_year_prefix)]
    nxt_data = [results[k] for k in nxt_keys]
    one_year_data = [results[k] for k in one_year_keys]

    # Plot
    if nxt_data:
        make_grouped_bar_plot(nxt_data, next_value_prefix, "Next Value Prediction Metrics")

    if one_year_data:
        make_grouped_bar_plot(one_year_data, one_year_prefix, "One-Year Prediction Metrics")

