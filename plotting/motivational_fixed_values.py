import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(times):
    avg = np.mean(times)
    p99 = np.percentile(times, 99)
    return avg, p99

def plot_averages(incast_avg_cc1, incast_avg_cc2, other_avg_cc1, other_avg_cc2, third_avg_cc1=None, third_avg_cc2=None):


    categories = ['Incast\nMicrobenchmark', 'Permutation\nMicrobenchmark', 'LLAMA\nRealistic Workload']
    values_cc1 = [incast_avg_cc1, other_avg_cc1, third_avg_cc1]
    values_cc2 = [incast_avg_cc2, other_avg_cc2, third_avg_cc2]

    
    x = np.arange(len(categories))
    width = 0.35  # width of the bars

    # Create a wider figure to make the plot more horizontal
    fig, ax = plt.subplots(figsize=(8, 3))
    rects1 = ax.bar(x - width/2, values_cc1, width, label='Swift', color="#e2526c")
    rects2 = ax.bar(x + width/2, values_cc2, width, label='MPRDMA', color="#49cbb3")

    ax.set_ylabel('Average Flow\nCompletion Time (us)')
    ax.set_title('Average Flow Completion Time Comparison Across Workloads')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Annotate only Swift bars (cc1) with percentage difference compared to MPRDMA (cc2)
    for i, rect in enumerate(rects1):
        swift = values_cc1[i]
        mprdma = values_cc2[i]
        # Calculate percentage difference: ((mprdma - swift) / mprdma)*100
        raw_percent_diff = ((mprdma - swift) / mprdma) * 100

        if raw_percent_diff > 0:
            # Swift is faster, annotate with a negative percentage in green
            display_text = f'-{raw_percent_diff:.2f}%'
            text_color = "green"
        else:
            # Swift is slower, annotate with a positive percentage in red
            display_text = f'+{abs(raw_percent_diff):.2f}%'
            text_color = "red"

        ax.annotate(display_text,
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=text_color)

    plt.tight_layout()
    plt.savefig("flow_completion_times_comparison.pdf", dpi=300)
    plt.savefig("flow_completion_times_comparison.png", dpi=300)
    plt.show()

def main():

    incast_avg_cc1 = 573.82848
    incast_avg_cc2 = 633.32624
    other_avg_cc1 = 114.09532999999999
    other_avg_cc2 = 111.07014000000001
    third_avg_cc1 = 221.26260151657917
    third_avg_cc2 = 106.71985462377171

    plot_averages(incast_avg_cc1, incast_avg_cc2, other_avg_cc1, other_avg_cc2, third_avg_cc1, third_avg_cc2)

if __name__ == "__main__":
    main()