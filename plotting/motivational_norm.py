import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def extract_completion_times(filename):
    times = []
    regex = re.compile(r"Completion Time Flow .*? is (\d+\.\d+)")
    with open(filename, 'r') as f:
        for line in f:
            match = regex.search(line)
            if match:
                times.append(float(match.group(1)))
    return times

def compute_stats(times):
    avg = np.mean(times)
    p99 = np.percentile(times, 99)
    return avg, p99

def plot_normalized_stats(incast_stats_cc1, incast_stats_cc2, other_stats_cc1, other_stats_cc2):
    # Compute percentage improvement: (CC1 - CC2) / CC1 * 100
    # (Assuming lower flow completion time is better.)
    incast_improve_avg = (incast_stats_cc1[0] - incast_stats_cc2[0]) / incast_stats_cc1[0] * 100
    incast_improve_p99 = (incast_stats_cc1[1] - incast_stats_cc2[1]) / incast_stats_cc1[1] * 100
    other_improve_avg = (other_stats_cc1[0] - other_stats_cc2[0]) / other_stats_cc1[0] * 100
    other_improve_p99 = (other_stats_cc1[1] - other_stats_cc2[1]) / other_stats_cc1[1] * 100

    categories = ['Incast Avg', 'Incast 99', 'Other Avg', 'Other 99']
    improvements = [incast_improve_avg, incast_improve_p99, other_improve_avg, other_improve_p99]
    
    x = np.arange(len(categories))
    width = 0.5

    fig, ax = plt.subplots()
    rects = ax.bar(x, improvements, width, color='skyblue')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Normalized Improvement: CC2 over CC1')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    # Annotate bars with improvement percentages
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # Offset the label above the bar
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot normalized improvement (percentage) in flow completion times between two congestion control algorithms"
    )
    parser.add_argument("--file_incast_cc1", help="File for incast benchmark - congestion control algorithm 1", required=True)
    parser.add_argument("--file_incast_cc2", help="File for incast benchmark - congestion control algorithm 2", required=True)
    parser.add_argument("--file_other_cc1", help="File for other benchmark - congestion control algorithm 1", required=True)
    parser.add_argument("--file_other_cc2", help="File for other benchmark - congestion control algorithm 2", required=True)
    args = parser.parse_args()

    # Incast benchmark
    times_incast_cc1 = extract_completion_times(args.file_incast_cc1)
    times_incast_cc2 = extract_completion_times(args.file_incast_cc2)
    # Other benchmark
    times_other_cc1 = extract_completion_times(args.file_other_cc1)
    times_other_cc2 = extract_completion_times(args.file_other_cc2)

    if not times_incast_cc1 or not times_incast_cc2 or not times_other_cc1 or not times_other_cc2:
        print("Error: Could not extract times from one or more files.")
        return

    incast_stats_cc1 = compute_stats(times_incast_cc1)
    incast_stats_cc2 = compute_stats(times_incast_cc2)
    other_stats_cc1 = compute_stats(times_other_cc1)
    other_stats_cc2 = compute_stats(times_other_cc2)

    plot_normalized_stats(incast_stats_cc1, incast_stats_cc2, other_stats_cc1, other_stats_cc2)

if __name__ == "__main__":
    main()