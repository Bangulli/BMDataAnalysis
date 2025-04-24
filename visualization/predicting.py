import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def plot_prediction_metrics_sweep(results, output, next_value_prefix = 'nxt', one_year_prefix = '1yr'):
    os.makedirs(output, exist_ok=True)

    nxts = [results[d] for d in list(results.keys()) if d.startswith(next_value_prefix)]
    yrs = [results[d] for d in list(results.keys()) if d.startswith(one_year_prefix)]

    # make plots for next value prediction
    measures = list(nxts[0].keys())
    values = {}
    for m in measures:
        if m in ['balanced_accuracy', 'accuracy', 'f1', 'recall', 'precision']:
            values[m] = [n[m] for n in nxts]
            x = range(len(values[m]))

            plt.bar(x, values[m])

            # Plot dotted line connecting the tops of the bars
            plt.plot(x, values[m], linestyle=':', color='black', marker='o')  # marker='o' is optional

            # Optional: Improve layout
            plt.xlabel("Input timepoints")
            plt.ylabel(f"Value - {m}")
            plt.title(f"{m} of next value prediction given input timepoints")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(output/f"next_value_{m}.png")
            plt.close()
            plt.clf()
        elif m == 'confusion_matrix':
            values[m] = [n[m] for n in nxts]
            fig = plot_confusion_matrices(values[m])
            fig.savefig(output/f"next_value_{m}.png")
            fig.clf()
            plt.close()
            plt.clf()
            


    # make plots for next value prediction
    measures = list(yrs[0].keys())
    values = {}
    for m in measures:
        if m in ['balanced_accuracy', 'accuracy', 'f1', 'recall', 'precision']:
            values[m] = [n[m] for n in yrs]
            x = range(len(values[m]))

            plt.bar(x, values[m])

            # Plot dotted line connecting the tops of the bars
            plt.plot(x, values[m], linestyle=':', color='black', marker='o')  # marker='o' is optional

            # Optional: Improve layout
            plt.xlabel("Input timepoints")
            plt.ylabel(f"Value - {m}")
            plt.title(f"{m} of one year prediction given input timepoints")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(output/f"one_year_{m}.png")
            plt.close()
            plt.clf()
        elif m == 'confusion_matrix':
            values[m] = [n[m] for n in nxts]
            fig = plot_confusion_matrices(values[m])
            fig.savefig(output/f"one_year_{m}.png")
            fig.clf()
            plt.close()
            plt.clf()

def plot_confusion_matrices(conf_matrices, class_labels='auto', title_prefix='Confusion Matrix'):
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
