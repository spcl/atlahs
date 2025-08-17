import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def extract_completion_times(filename):
    times = []
    # First pattern: "Completion Time Flow ... is <value>"
    pattern1 = re.compile(r"Completion Time Flow .*? is (\d+\.\d+)")
    # Second pattern: "OT <value>" (FCT from alternative log lines)
    pattern2 = re.compile(r"OT\s+(\d+(?:\.\d+)?)")
    
    with open(filename, 'r') as f:
        for line in f:
            match = pattern1.search(line)
            if match:
                times.append(float(match.group(1)))
            else:
                match2 = pattern2.search(line)
                if match2:
                    times.append(float(match2.group(1)))
    return times

def compute_stats(times):
    avg = np.mean(times)
    p99 = np.percentile(times, 99)
    return avg, p99

def plot_averages(incast_avg_cc1, incast_avg_cc2, other_avg_cc1, other_avg_cc2, third_avg_cc1=None, third_avg_cc2=None):
    # Determine the workloads to plot
    if third_avg_cc1 is not None and third_avg_cc2 is not None:
        categories = ['Incast\nMicrobenchmark', 'Permutation\nMicrobenchmark', 'LLAMA\nRealistic Workload']
        values_cc1 = [incast_avg_cc1, other_avg_cc1, third_avg_cc1]
        values_cc2 = [incast_avg_cc2, other_avg_cc2, third_avg_cc2]
    else:
        categories = ['Incast', 'Permutation']
        values_cc1 = [incast_avg_cc1, other_avg_cc1]
        values_cc2 = [incast_avg_cc2, other_avg_cc2]
    
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
    parser = argparse.ArgumentParser(
        description="Plot average flow completion times for two congestion control algorithms across workloads"
    )
    parser.add_argument("--file_incast_cc1", help="File for incast benchmark - congestion control algorithm 1", required=True)
    parser.add_argument("--file_incast_cc2", help="File for incast benchmark - congestion control algorithm 2", required=True)
    parser.add_argument("--file_perm_cc1", help="File for other benchmark - congestion control algorithm 1", required=True)
    parser.add_argument("--file_perm_cc2", help="File for other benchmark - congestion control algorithm 2", required=True)
    parser.add_argument("--file_llama_cc1", help="File for third benchmark - congestion control algorithm 1", required=False)
    parser.add_argument("--file_llama_cc2", help="File for third benchmark - congestion control algorithm 2", required=False)
    args = parser.parse_args()

    # Incast benchmark
    times_incast_cc1 = extract_completion_times(args.file_incast_cc1)
    times_incast_cc2 = extract_completion_times(args.file_incast_cc2)
    # Perm benchmark
    times_other_cc1 = extract_completion_times(args.file_perm_cc1)
    times_other_cc2 = extract_completion_times(args.file_perm_cc2)

    incast_stats_cc1 = compute_stats(times_incast_cc1)
    incast_stats_cc2 = compute_stats(times_incast_cc2)
    other_stats_cc1 = compute_stats(times_other_cc1)
    other_stats_cc2 = compute_stats(times_other_cc2)

    # Use only the average (first element) from the tuple
    incast_avg_cc1 = incast_stats_cc1[0]
    incast_avg_cc2 = incast_stats_cc2[0]
    other_avg_cc1 = other_stats_cc1[0]
    other_avg_cc2 = other_stats_cc2[0]
    
    # Check for llama workload
    third_avg_cc1 = None
    third_avg_cc2 = None
    if args.file_llama_cc1 and args.file_llama_cc2:
        times_third_cc1 = extract_completion_times(args.file_llama_cc1)
        times_third_cc2 = extract_completion_times(args.file_llama_cc2)
        if times_third_cc1 and times_third_cc2:
            third_stats_cc1 = compute_stats(times_third_cc1)
            third_stats_cc2 = compute_stats(times_third_cc2)
            third_avg_cc1 = third_stats_cc1[0]
            third_avg_cc2 = third_stats_cc2[0]
        else:
            print("Warning: Could not extract times for the third workload. Skipping third workload.")

    plot_averages(incast_avg_cc1, incast_avg_cc2, other_avg_cc1, other_avg_cc2, third_avg_cc1, third_avg_cc2)

if __name__ == "__main__":
    main()