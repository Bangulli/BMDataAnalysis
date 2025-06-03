import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def plot_prediction_metrics_sweep_fold(fold_results_list, output, next_value_prefix='nxt', one_year_prefix='1yr', classes='auto', distribution=None):
    os.makedirs(output, exist_ok=True)
    classification_metrics = ['balanced_accuracy', 'accuracy', 'f1', 'recall', 'precision']

    def aggregate_metrics_across_folds(fold_results, keys):
        """Average metrics and std across folds per key."""
        aggregated = {}
        for key in keys:
            per_key_values = {m: [] for m in classification_metrics}
            for fold in fold_results:
                if key in fold:
                    for m in classification_metrics:
                        per_key_values[m].append(fold[key][m])
            aggregated[key] = {
                m: {
                    'mean': np.mean(per_key_values[m]),
                    'std': np.std(per_key_values[m])
                }
                for m in classification_metrics
            }
        return aggregated

    def make_grouped_metric_plot(aggregated_data, keys, prefix, title):
        metrics = classification_metrics
        num_metrics = len(metrics)
        num_keys = len(keys)

        bar_width = 0.8 / num_metrics
        x = np.arange(num_keys)
        colors = plt.cm.tab10.colors

        plt.figure(figsize=(max(8, num_keys * 1.5), 6))
        for i, metric in enumerate(metrics):
            means = [aggregated_data[k][metric]['mean'] for k in keys]
            stds = [aggregated_data[k][metric]['std'] for k in keys]
            bars = plt.bar(x + i * bar_width, means, width=bar_width, label=metric,
                           yerr=stds, capsize=5, color=colors[i % len(colors)])

            for bar, std in zip(bars, stds):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height - std - 0.11,
                         f"{height:.2f}", ha='center', va='bottom', fontsize=10, rotation=90)

        plt.xticks(x + bar_width * (num_metrics - 1) / 2, keys, rotation=45, ha='right')
        plt.xlabel("Input timepoints")
        plt.ylabel("Metric Value")
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output, f"{prefix}_grouped_metrics.png"))
        plt.close()

    def plot_confusions_across_folds(fold_results, keys, prefix):
        from matplotlib import pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        for key in keys:
            fig, axs = plt.subplots(1, len(fold_results), figsize=(len(fold_results) * 4, 4))
            for i, fold in enumerate(fold_results):
                if key in fold and 'confusion_matrix' in fold[key]:
                    ConfusionMatrixDisplay(fold[key]['confusion_matrix'], display_labels=classes).plot(ax=axs[i], colorbar=False)
                    axs[i].set_title(f"Fold {i+1}")
            plt.suptitle(f"Confusion Matrices for {key}")
            plt.tight_layout()
            fig.savefig(os.path.join(output, f"{prefix}_{key}_confusion_matrices.png"))
            plt.close(fig)

    # One Year
    if one_year_prefix:
        yr_keys = [k for k in fold_results_list[0] if k.startswith(one_year_prefix)]
        aggregated_yr = aggregate_metrics_across_folds(fold_results_list, yr_keys)
        title = f"""One-Year Prediction Metrics - Class Distribution = {distribution if distribution is not None else 'unknown'}"""
        make_grouped_metric_plot(aggregated_yr, yr_keys, "one_year", title)
        #plot_confusions_across_folds(fold_results_list, yr_keys, "one_year")

def plot_prediction_metrics_sweep(results, output, next_value_prefix='nxt', one_year_prefix='1yr', classes='auto', distribution=None, std=None):
    os.makedirs(output, exist_ok=True)

    classification_metrics = ['balanced_accuracy', 'accuracy', 'f1', 'recall', 'precision']

    def make_grouped_metric_plot(data_dicts, keys, prefix, title, data_std):
        metrics = [m for m in data_dicts[0].keys() if m in classification_metrics]
        num_metrics = len(metrics)
        num_keys = len(keys)

        bar_width = 0.8 / num_metrics
        x = np.arange(num_keys)
        colors = plt.cm.tab10.colors

        plt.figure(figsize=(max(8, num_keys * 1.5), 6))
        for i, metric in enumerate(metrics):
            values = [d[metric] for d in data_dicts]
            if data_std is not None:
                stds = [d[metric] for d in data_std]
                bars = plt.bar(x + i * bar_width, values, width=bar_width, label=metric,
                            yerr=stds, capsize=5, color=colors[i % len(colors)])

                for bar, std in zip(bars, stds):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height - std - 0.11,
                            f"{height:.2f}", ha='center', va='bottom', fontsize=10, rotation=90)
            else:
                bars = plt.bar(x + i * bar_width, values, width=bar_width, label=metric,
                            capsize=5, color=colors[i % len(colors)])

                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height - 0.11,
                            f"{height:.2f}", ha='center', va='bottom', fontsize=10, rotation=90)
        


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


    # --- One Year Prediction ---
    yr_keys = [k for k in results if k.startswith(one_year_prefix)]
    yr_data = [results[k] for k in yr_keys]
    if std is not None: yr_std = [std[k] for k in yr_keys]
    else: yr_std = None
    if yr_data:
        make_grouped_metric_plot(yr_data, yr_keys, "one_year", f"""One-Year Prediction Metrics - Class Distribution = {distribution if distribution is not None else 'unknown'}""", yr_std)
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

