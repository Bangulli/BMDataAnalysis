import matplotlib.pyplot as plt
import os

def plot_prediction_metrics(results, output, next_value_prefix = 'nxt', one_year_prefix = '1yr'):
    os.makedirs(output, exist_ok=True)

    nxts = [results[d] for d in list(results.keys()) if d.startswith(next_value_prefix)]
    yrs = [results[d] for d in list(results.keys()) if d.startswith(one_year_prefix)]

    # make plots for next value prediction
    measures = list(nxts[0].keys())
    values = {}
    for m in measures:
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

    # make plots for next value prediction
    measures = list(yrs[0].keys())
    values = {}
    for m in measures:
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
    