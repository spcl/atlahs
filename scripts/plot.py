import os
import argparse
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
from pathlib import Path
import re

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



FONT_SIZE = 13

def print_warning(message: str, flush: bool = True) -> None:
    """
    Print a warning message.
    """
    print(f"\033[93m{message}\033[0m", flush=flush)


def print_error(message: str, flush: bool = True) -> None:
    """
    Print an error message.
    """
    print(f"\033[91m{message}\033[0m", flush=flush)

def print_success(message: str, flush: bool = True) -> None:
    """
    Print a success message.
    """
    print(f"\033[92m{message}\033[0m", flush=flush)

def print_info(message: str, flush: bool = True) -> None:
    """
    Print an info message.
    """
    print(f"\033[94m{message}\033[0m", flush=flush)

def read_overall_times(filename):
    times = []
    regex = re.compile(r"OT\s+(\d+)")
    with open(filename, 'r') as f:
        for line in f:
            match = regex.search(line)
            if match:
                # Divide by 1000 to convert to microseconds.
                times.append(int(match.group(1)) / 1000)
    return times

def read_max_host_time(filename: str) -> float:
    """
    Parse a log file and return the maximum host completion time in seconds.
    """
    times = []
    regex = re.compile(r'Host\s+\d+:\s+(\d+)')
    with open(filename, 'r') as f:
        for line in f:
            m = regex.search(line)
            if m:
                times.append(int(m.group(1)))
    if not times:
        raise ValueError(f"No host times found in {filename}")
    # Convert from nanoseconds to seconds
    return max(times) / 1e9

def read_total_nacks(filename: str) -> int:
    """
    Parse a log file and return the total number of NACKS.
    """
    total = None
    regex = re.compile(r'Total\s+Number\s+of\s+NACKS\s+(\d+)')
    with open(filename, 'r') as f:
        for line in f:
            m = regex.search(line)
            if m:
                total = int(m.group(1))
                break
    if total is None:
        raise ValueError(f"No NACKS total found in {filename}")
    return total

def plot_four_violins(plots_dir, data1, data2, data3, data4,
                      label1="MPRDMA", label2="NDP"):
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Define positions for the four violins with less distance between groups.
    positions = [1, 2, 3, 4]
    violin_data = [data1, data2, data3, data4]
    
    # Create the violins without medians/extrema.
    vp = ax.violinplot(violin_data, positions=positions,
                         showmedians=False, showmeans=False, showextrema=False)
    
    # Customize colors for the violins: use lightcoral for MPRDMA and palegreen for NDP.
    colors = ['lightcoral', 'palegreen', 'lightcoral', 'palegreen']
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    
    # Compute max, 99th percentile and mean for each dataset.
    max_vals = [np.max(d) for d in violin_data]
    p99_vals = [np.percentile(d, 99) for d in violin_data]
    means   = [np.mean(d) for d in violin_data]
    
    # Overlay scatter markers (using s=20 for slightly smaller markers)
    # Group 1: positions 1 and 2.
    ax.scatter([positions[0]], [max_vals[0]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[0]], [p99_vals[0]], color='mediumseagreen', marker='x', s=20, zorder=3)
    ax.scatter([positions[1]], [max_vals[1]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[1]], [p99_vals[1]], color='mediumseagreen', marker='x', s=20, zorder=3)
    # Group 2: positions 3 and 4.
    ax.scatter([positions[2]], [max_vals[2]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[2]], [p99_vals[2]], color='mediumseagreen', marker='x', s=20, zorder=3)
    ax.scatter([positions[3]], [max_vals[3]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[3]], [p99_vals[3]], color='mediumseagreen', marker='x', s=20, zorder=3)
    
    # Add extra scatter markers for the mean (dodgerblue).
    for pos, mean in zip(positions, means):
        ax.scatter([pos], [mean], color='dodgerblue', marker='o', s=20, zorder=3)
    
    # Annotate each violin with max, 99th percentile, and mean values (fontsize increased).
    for i, pos in enumerate(positions):
        ax.text(pos + 0.05, max_vals[i], f"{max_vals[i]:.1f} µs",
                color='firebrick', fontsize=11, verticalalignment='center')
        ax.text(pos + 0.05, p99_vals[i], f"{p99_vals[i]:.1f} µs",
                color='mediumseagreen', fontsize=11, verticalalignment='center')
        ax.text(pos + 0.05, means[i] + 0.2, f"{means[i]:.1f} µs",
                color='dodgerblue', fontsize=11, verticalalignment='center')
    
    # Set x-axis ticks and labels (fontsize increased in tick_params).
    ax.set_xticks(positions)
    ax.set_xticklabels([label1, label2, label1, label2], fontsize=11)
    ax.set_ylabel("MCT (µs)", fontsize=13)
    
    # Use only horizontal grid lines.
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    
    # Custom legend for markers (with smaller marker size and increased font size in legend).
    custom_handles = [
        Line2D([], [], color='firebrick', marker='o', linestyle='None', markersize=6, label='Max'),
        Line2D([], [], color='mediumseagreen', marker='o', linestyle='None', markersize=6, label='99th Percentile'),
        Line2D([], [], color='dodgerblue', marker='o', linestyle='None', markersize=6, label='Mean')
    ]
    ax.legend(handles=custom_handles, fontsize=11)
    
    # Add oversubscription group labels below the x-axis labels (plain text, fontsize increased).
    ax.text(1.5, -0.16, "No Oversubscription", ha='center', va='center',
            transform=ax.get_xaxis_transform(), fontsize=13)
    ax.text(3.5, -0.16, "8:1 Oversubscription", ha='center', va='center',
            transform=ax.get_xaxis_transform(), fontsize=13)
    
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    case_study_3_dir = Path(plots_dir) / "case_studies" / "case_study_storage"
    filename_storage = "case_study_1.pdf"
    os.makedirs(case_study_3_dir, exist_ok=True)
    plt.savefig(case_study_3_dir / filename_storage, dpi=300)
    #plt.show()

def plot_storage_case_study(data_dir: str, plots_dir: str) -> None:

    case_study_3_dir = Path(data_dir) / "case_studies" / "case_study_storage"
    filename1 = "1os_mprdma.tmp"
    filename2 = "1os_eqds.tmp"
    filename3 = "8os_mprdma.tmp"
    filename4 = "8os_eqds.tmp"

    data1 = read_overall_times(case_study_3_dir / filename1)
    data2 = read_overall_times(case_study_3_dir / filename2)
    data3 = read_overall_times(case_study_3_dir / filename3)
    data4 = read_overall_times(case_study_3_dir / filename4)

    missing = False
    for f, d in [(filename1, data1), (filename2, data2),
                 (filename3, data3), (filename4, data4)]:
        if not d:
            print(f"No overall time data found in {f}.")
            missing = True
            
    if not missing:
        plot_four_violins(plots_dir, data1, data2, data3, data4,
                          label1="MPRDMA", label2="NDP")

def plot_job_allocation_case_study(data_dir: str, plots_dir: str) -> None:
    # Define paths for job‐allocation logs
    case_study_job_alloc_dir = Path(data_dir) / "case_studies" / "case_study_job_alloc"
    packed_llama_file  = case_study_job_alloc_dir / "llama_packed.tmp"
    packed_lulesh_file = case_study_job_alloc_dir / "lulesh_packed.tmp"
    random_llama_file  = case_study_job_alloc_dir / "llama_random.tmp"
    random_lulesh_file = case_study_job_alloc_dir / "lulesh_random.tmp"

    # Parse max host completion time (s) from each log
    llama_packed  = read_max_host_time(packed_llama_file)
    lulesh_packed = read_max_host_time(packed_lulesh_file)
    llama_random  = read_max_host_time(random_llama_file)
    lulesh_random = read_max_host_time(random_lulesh_file)

    runtime_data = {
        'Packed Allocation': {'Llama': llama_packed, 'LULESH': lulesh_packed},
        'Random Allocation': {'Llama': llama_random, 'LULESH': lulesh_random},
    }

    # Extract groups and job labels
    groups = list(runtime_data.keys())          # ['Packed Allocation', 'Random Allocation']
    jobs   = list(next(iter(runtime_data.values())).keys())  # ['Llama', 'LULESH']

    n_groups = len(groups)
    n_jobs   = len(jobs)
    bar_width = 0.35
    index = np.arange(n_groups)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--')

    # Plot each job as grouped bars
    for i, job in enumerate(jobs):
        positions = index + (i * bar_width) - ((n_jobs - 1) * bar_width / 2)
        values = [runtime_data[g][job] for g in groups]
 
        # pick a custom colour per job
        if job == 'Llama':
            color = "#2eaba9"
        elif job == 'LULESH':
            color = "#fb8574"
        else:
            color = None  # fallback to default
 
        ax.bar(positions, values, bar_width,
               label=job, color=color, zorder=2)

        # Annotate the Random Allocation bar with % increase
        if len(values) == 2:
            pct = (values[1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
            ax.text(positions[1], values[1] + 0.1, f"+{pct:.0f}%",
                    ha='center', va='bottom', color='red', fontsize=12)

    ax.set_xlabel('Allocation Type', fontsize=14)
    ax.set_ylabel('Simulated Runtime (s)', fontsize=12.5)
    ax.set_xticks(index)
    ax.set_xticklabels(groups, fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()

    # Save
    out_dir = Path(plots_dir) / "case_studies" / "case_study_job_alloc"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir / "case_study_3.pdf", dpi=300)

def plot_lgs_vs_htsim(data_dir: str, plots_dir: str) -> None:
    # Define groups and algorithm labels
    groups = ["No Oversubscription", "4:1 Oversubscription"]
    algorithms = ["ATLAHS LGS", "ATLAHS htsim"]

    case_study_2_dir = Path(data_dir) / "case_studies" / "case_study_lgs_vs_htsim"

    # Placeholder file paths for run time logs
    lgs_nos_file    = case_study_2_dir / "llama_lgs.tmp"
    lgs_4os_file    = case_study_2_dir / "llama_lgs.tmp"
    htsim_nos_file  = case_study_2_dir / "llama_no_os_htsim.tmp"
    htsim_4os_file  = case_study_2_dir / "llama_os_htsim.tmp"

    # Extract max host completion time (seconds) from each log
    run_time_data = {
        "ATLAHS LGS": [
            read_max_host_time(lgs_nos_file),
            read_max_host_time(lgs_4os_file),
        ],
        "ATLAHS htsim": [
            read_max_host_time(htsim_nos_file),
            read_max_host_time(htsim_4os_file),
        ],
    }

    dropped_packets_data = [292765, 206409686]

    # Setup the x locations for the groups
    x = np.arange(len(groups))
    width = 0.35  # bar width

    # Create subplots: left for run time, right for dropped packets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_axisbelow(True)  # Ensures grid lines appear behind the bars in ax1
    ax2.set_axisbelow(True)  # Ensures grid lines appear behind the bars in ax2

    # Left plot: grouped bar chart for Total Run Time
    ax1.bar(x - width/2, run_time_data["ATLAHS LGS"], width, label="ATLAHS LGS", color="#20b2aa")
    ax1.bar(x + width/2, run_time_data["ATLAHS htsim"], width, label="ATLAHS htsim", color="#40e0d0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=14)  # increased tick label size
    ax1.set_ylabel("Time For Training Iteration (s)", fontsize=14.5)
    ax1.set_xlabel("Topology Configuration", fontsize=14.5)
    ax1.legend(prop={'size': 14})

    # Add horizontal dotted grid lines
    ax1.yaxis.grid(True, linestyle=':', color='black', alpha=0.7)

    # Annotate percentage difference on top of the ATLAHS LGS bars
    for i in range(len(groups)):
        lgs_val = run_time_data["ATLAHS LGS"][i]
        htsim_val = run_time_data["ATLAHS htsim"][i]
        diff_percentage = ((lgs_val - htsim_val) / lgs_val) * 100 if lgs_val != 0 else 0
        annotation = f"{diff_percentage:+.1f}%"
        # Offset the annotation slightly above the LGS bar
        ax1.text(x[i] - width/2, lgs_val + 0.05 * lgs_val, annotation, ha="center", va="bottom", color="black", fontsize=11.5)

    # Right plot: bar chart for Total Dropped Packets (only ATLAHS htsim)
    right_width = 0.5  # Increased bar width to reduce spacing between bars
    ax2.bar(x, dropped_packets_data, right_width, color="#40e0d0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=14)
    ax2.set_ylabel("Total Packet Drops", fontsize=14.5)
    ax2.set_xlabel("Topology Configuration", fontsize=14.5)

    # Add horizontal dotted grid lines
    ax2.yaxis.grid(True, linestyle=':', color='black', alpha=0.7)

    plt.tight_layout()
    case_study_2_dir = Path(plots_dir) / "case_studies" / "case_study_lgs_vs_htsim"
    filename_storage = "case_study_2.pdf"
    os.makedirs(case_study_2_dir, exist_ok=True)
    plt.savefig(case_study_2_dir / filename_storage, dpi=300)

def plot_case_studies(data_dir: str, plots_dir: str) -> None:
    """
    Plot the results of the case studies.
    """
    print_info(f"Plotting case studies from {data_dir}...")

    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Plot storage case study
    plot_storage_case_study(data_dir, plots_dir)

    # Plot job allocation case study
    plot_job_allocation_case_study(data_dir, plots_dir)

    # Plot HPC validation experiment
    plot_lgs_vs_htsim(data_dir, plots_dir)


def plot_hpc_validation_exp(data_dir: str, plots_dir: str) -> None:
    """
    Plot the results of the HPC validation experiment.
    """
    print_info(f"Plotting HPC validation experiment results from {data_dir}...")
    assert os.path.exists(data_dir), f"HPC data directory {data_dir} does not exist."

    cols = ["workload", "runtime"]

    # Fetch the data for ATLAHS LGS
    atlahs_lgs_file = os.path.join(data_dir, "atlahs_lgs.csv")
    if not os.path.exists(atlahs_lgs_file):
        print_warning(f"ATLAHS LGS file {atlahs_lgs_file} does not exist.")
        # Create an empty dataframe
        print_info(f"Creating empty dataframe for ATLAHS LGS...")
        atlahs_lgs_df = pd.DataFrame(columns=cols)
    else:
        # Read the file into a pandas dataframe
        # Tell pandas that the first row is not the header
        atlahs_lgs_df = pd.read_csv(atlahs_lgs_file, header=None)
        # Add column names ["workload", "runtime"]
        atlahs_lgs_df.columns = ["workload", "runtime"]
        print_info(f"Read ATLAHS LGS data from {atlahs_lgs_file}")
    
    # Fetch data for ATLAHS htsim
    atlahs_htsim_file = os.path.join(data_dir, "atlahs_htsim.csv")
    if not os.path.exists(atlahs_htsim_file):
        print_warning(f"ATLAHS htsim file {atlahs_htsim_file} does not exist.")
        # Create an empty dataframe
        print_info(f"Creating empty dataframe for ATLAHS htsim...")
        atlahs_htsim_df = pd.DataFrame(columns=["workload", "runtime"])
    else:
        # Read the file into a pandas dataframe
        # Tell pandas that the first row is not the header
        atlahs_htsim_df = pd.read_csv(atlahs_htsim_file, header=None)
        # Add column names ["workload", "runtime"]
        atlahs_htsim_df.columns = ["workload", "runtime"]
        print_info(f"Read ATLAHS htsim data from {atlahs_htsim_file}")
    
    # Fetch data for measured runtime
    measured_runtime_file = os.path.join(data_dir, "measured.csv")
    if not os.path.exists(measured_runtime_file):
        print_warning(f"Measured runtime file {measured_runtime_file} does not exist.")
        # Create an empty dataframe
        print_info(f"Creating empty dataframe for measured runtime...")
        measured_runtime_df = pd.DataFrame(columns=cols)
    else:
        # Read the file into a pandas dataframe
        # Tell pandas that the first row is not the header
        measured_runtime_df = pd.read_csv(measured_runtime_file, header=None)
        # Add column names ["workload", "runtime"]
        measured_runtime_df.columns = ["workload", "runtime"]
        print_info(f"Read measured runtime data from {measured_runtime_file}")
    
    # Combine the dataframes into a single dataframe
    # The new dataframe should have the following columns: "workload",
    # "atlahs_lgs", "atlahs_htsim", "measured"
    # This should be done by aligning the dataframes on the "workload" column
    atlahs_lgs_df = atlahs_lgs_df.set_index("workload")
    atlahs_htsim_df = atlahs_htsim_df.set_index("workload")
    measured_runtime_df = measured_runtime_df.set_index("workload")
    df = pd.concat([atlahs_lgs_df, atlahs_htsim_df, measured_runtime_df], axis=1)
    df.columns = ["atlahs_lgs", "atlahs_htsim", "measured"]
    df = df.reset_index()
    print_info(f"Combined dataframes into a single dataframe with {len(df)} rows")
    print(df)

    # Plot the HPC results
    # Convert the runtime columns from ns to s
    df["atlahs_lgs"] = df["atlahs_lgs"] / 1e9
    df["atlahs_htsim"] = df["atlahs_htsim"] / 1e9
    df["measured"] = df["measured"] / 1e9
    
    df["error_lgs"] = (df["atlahs_lgs"] - df["measured"]) / df["measured"] * 100
    df["error_htsim"] = (df["atlahs_htsim"] - df["measured"]) / df["measured"] * 100
    

    # Check the number of rows in the dataframe
    num_rows = len(df)
    fig_size = (num_rows * 1.8, 5)

    # Plot the results
    fig, ax = plt.subplots(figsize=fig_size)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    positions = range(num_rows)

    width = 0.3
    # Plot the real runtimes
    ax.bar([p - width for p in positions], df["measured"], width=width, label="Measured", color="steelblue")
    # Plot the ATLAHS LGS runtimes
    ax.bar([p for p in positions], df["atlahs_lgs"], width=width, label="ATLAHS LGS", color="lightseagreen")
    # Plot the ATLAHS htsim runtimes
    ax.bar([p + width for p in positions], df["atlahs_htsim"], width=width, label="ATLAHS htsim", color="turquoise")
    
    # Labels the bars with the error percentage
    margin = 2.2
    for i, v in enumerate(df['error_lgs']):
        ax.text(i, df['atlahs_lgs'][i] + margin,
                f"{v:.1f}%", ha="center", va="center", color="red",
                weight="bold", fontsize=FONT_SIZE, rotation=60)
    
    for i, v in enumerate(df['error_htsim']):
        ax.text(i + width, df['atlahs_htsim'][i] + margin,
                f"{v:.1f}%", ha="center", va="center", color="red",
                weight="bold", fontsize=FONT_SIZE, rotation=60)
    
    # Set the x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(df["workload"], rotation=60, ha="center", fontsize=FONT_SIZE + 1)
    
    # Set the y-axis label
    ax.set_ylabel("Time (s)", fontsize=FONT_SIZE)
    ax.set_ylim(0, max(df["measured"]) * 1.1)
    
    # Set the title
    ax.set_title("HPC Validation Experiment", fontsize=FONT_SIZE + 2)
    ax.tick_params(axis="y", labelsize=FONT_SIZE + 3)

    ax.legend(prop={"size": FONT_SIZE + 1}, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "validation_hpc_runtime.pdf"), dpi=300)
    plt.close()
    print_success(f"Saved HPC validation experiment results to {os.path.join(plots_dir, 'validation_hpc_runtime.pdf')}")

    
    
def plot_ai_validation_exp(data_dir: str, plots_dir: str) -> None:
    """
    Plot the results of the AI validation experiment.
    """

    # Plot the results for the AI validation experiment

    # ================================
    # Plot the runtime data
    # ================================
    print_info(f"Plotting AI validation experiment results from {data_dir}...")
    assert os.path.exists(data_dir), f"AI data directory {data_dir} does not exist."
    
    cols = ["workload", "runtime"]
    # Fetch the data for ATLAHS LGS
    atlahs_lgs_file = os.path.join(data_dir, "atlahs_lgs.csv")
    if not os.path.exists(atlahs_lgs_file):
        print_warning(f"ATLAHS LGS file {atlahs_lgs_file} does not exist.")
        # Create an empty dataframe
        print_info(f"Creating empty dataframe for ATLAHS LGS...")
        atlahs_lgs_df = pd.DataFrame(columns=cols)
    else:
        # Read the file into a pandas dataframe
        # Tell pandas that the first row is not the header
        atlahs_lgs_df = pd.read_csv(atlahs_lgs_file, header=None)
        # Add column names ["workload", "runtime"]
        atlahs_lgs_df.columns = ["workload", "runtime"]
        print_info(f"Read ATLAHS LGS data from {atlahs_lgs_file}")
    
    # Fetch data for ATLAHS htsim
    atlahs_htsim_file = os.path.join(data_dir, "atlahs_htsim.csv")
    if not os.path.exists(atlahs_htsim_file):
        print_warning(f"ATLAHS htsim file {atlahs_htsim_file} does not exist.")
        # Create an empty dataframe
        print_info(f"Creating empty dataframe for ATLAHS htsim...")
        atlahs_htsim_df = pd.DataFrame(columns=cols)
    else:
        # Read the file into a pandas dataframe
        # Tell pandas that the first row is not the header
        atlahs_htsim_df = pd.read_csv(atlahs_htsim_file, header=None)
        # Add column names ["workload", "runtime"]
        atlahs_htsim_df.columns = ["workload", "runtime"]
        print_info(f"Read ATLAHS htsim data from {atlahs_htsim_file}")
    
    # Fetch data for AstraSim
    astra_sim_file = os.path.join(data_dir, "astra_sim.csv")
    if not os.path.exists(astra_sim_file):
        print_warning(f"AstraSim file {astra_sim_file} does not exist.")
        # Create an empty dataframe
        print_info(f"Creating empty dataframe for AstraSim...")
        astra_sim_df = pd.DataFrame(columns=cols)
    else:
        # Read the file into a pandas dataframe
        # Tell pandas that the first row is not the header
        astra_sim_df = pd.read_csv(astra_sim_file, header=None)
        # Add column names ["workload", "runtime"]
        astra_sim_df.columns = ["workload", "runtime"]
        print_info(f"Read AstraSim data from {astra_sim_file}")

    # Fetch data for measured runtime
    measured_runtime_file = os.path.join(data_dir, "measured.csv")
    if not os.path.exists(measured_runtime_file):
        print_warning(f"Measured runtime file {measured_runtime_file} does not exist.")
        # Create an empty dataframe
        print_info(f"Creating empty dataframe for measured runtime...")
        measured_runtime_df = pd.DataFrame(columns=cols)    
    else:
        # Read the file into a pandas dataframe
        # Tell pandas that the first row is not the header
        measured_runtime_df = pd.read_csv(measured_runtime_file, header=None)
        # Add column names ["workload", "runtime"]
        measured_runtime_df.columns = ["workload", "runtime"]
        print_info(f"Read measured runtime data from {measured_runtime_file}")


    # Combine the dataframes into a single dataframe
    atlahs_lgs_df = atlahs_lgs_df.set_index("workload")
    atlahs_htsim_df = atlahs_htsim_df.set_index("workload")
    astra_sim_df = astra_sim_df.set_index("workload")
    measured_runtime_df = measured_runtime_df.set_index("workload")
    df = pd.concat([atlahs_lgs_df, atlahs_htsim_df, astra_sim_df, measured_runtime_df], axis=1)
    df.columns = ["atlahs_lgs", "atlahs_htsim", "astra_sim", "measured"]
    df = df.reset_index()
    print_info(f"Combined dataframes into a single dataframe with {len(df)} rows")
    print(df)

    # Convert the runtime columns from ns to s per iteration
    df["atlahs_lgs"] = df["atlahs_lgs"] / 1e9 / 2
    df["atlahs_htsim"] = df["atlahs_htsim"] / 1e9 / 2
    df["astra_sim"] = df["astra_sim"] / 1e9 / 2
    df["measured"] = df["measured"] / 1e9 / 2

    df["error_lgs"] = (df["atlahs_lgs"] - df["measured"]) / df["measured"] * 100
    df["error_htsim"] = (df["atlahs_htsim"] - df["measured"]) / df["measured"] * 100
    df["error_astra"] = (df["astra_sim"] - df["measured"]) / df["measured"] * 100

    # Plot the results
    # Check the number of rows in the dataframe
    num_rows = len(df)
    fig_size = (num_rows * 4, 5)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    positions = range(num_rows)

    width = 0.2
    # Plot the real runtimes
    ax.bar([p - 1.5 * width for p in positions], df["measured"], width=width, label="Measured", color="steelblue")
    # Plot the ATLAHS LGS runtimes
    ax.bar([p - 0.5 * width for p in positions], df["atlahs_lgs"], width=width, label="ATLAHS LGS", color="lightseagreen")
    # Plot the ATLAHS htsim runtimes
    ax.bar([p + 0.5 * width for p in positions], df["atlahs_htsim"], width=width, label="ATLAHS htsim", color="turquoise")
    # Plot the AstraSim runtimes
    ax.bar([p + 1.5 * width for p in positions], df["astra_sim"], width=width, label="AstraSim", color="salmon")

    # Labels the bars with the error percentage
    margin = 0.05
    for i, v in enumerate(df['error_lgs']):
        p = positions[i] - 0.5 * width
        ax.text(p, df['atlahs_lgs'][i] + margin,
                f"{v:.1f}%", ha="center", va="center", color="red",
                weight="bold", fontsize=FONT_SIZE)

    for i, v in enumerate(df['error_htsim']):
        p = positions[i] + 0.5 * width
        ax.text(p, df['atlahs_htsim'][i] + margin,
                f"{v:.1f}%", ha="center", va="center", color="red",
                weight="bold", fontsize=FONT_SIZE)
    
    for i, v in enumerate(df['error_astra']):
        # Check if the error is a valid number if not, 
        # Set it to a string "Error"
        p = positions[i] + 1.5 * width
        is_error = pd.isna(v)
        rotation = 90 if is_error else 0
        y = 0.05 if is_error else df['astra_sim'][i] + margin
        ax.text(p, y,
                f"{v:.1f}%" if not is_error else "Error", ha="center", va="center", color="red",
                weight="bold", fontsize=FONT_SIZE, rotation=rotation)

    # Set the x-axis labels
    ax.set_xticks(positions)
    x_labels = []
    for w in df["workload"]:
        tokens = w.split("_")
        label = " ".join(tokens[:3]) + "\n" + " ".join(tokens[3:])
        x_labels.append(label)

    ax.set_xticklabels(x_labels, ha="center", fontsize=FONT_SIZE + 1)

    # Set the y-axis label
    ax.set_ylabel("Time / Training Iteration (s)", fontsize=FONT_SIZE + 1)
    # ax.set_ylim(0, max(df["measured"]) * 1.1)
    
    # Set the title
    ax.set_title("AI Validation Experiment", fontsize=FONT_SIZE + 2)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)

    ax.legend(prop={"size": FONT_SIZE + 1}, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "validation_ai_runtime.pdf"), dpi=300)
    plt.close()
    print_success(f"Saved AI validation experiment results to {os.path.join(plots_dir, 'validation_ai_runtime.pdf')}")


    # ================================
    # Plot the trace size data
    # ================================

    trace_size_file = os.path.join(data_dir, "trace_sizes.csv")
    assert os.path.exists(trace_size_file), f"Trace size file {trace_size_file} does not exist."

    trace_size_df = pd.read_csv(trace_size_file)
    assert len(trace_size_df) == len(df), f"Trace size dataframe has {len(trace_size_df)} rows, but AI dataframe has {len(df)} rows."
    print_info(f"Read trace size data from {trace_size_file}")
    print(trace_size_df)

    # Convert the trace size columns
    # "goal_size" and "chakra_size" from MiB to GiB
    trace_size_df["goal_size"] = trace_size_df["goal_size"] / 1024
    trace_size_df["chakra_size"] = trace_size_df["chakra_size"] / 1024

    # Plot the trace size data
    fig_size = (len(trace_size_df) * 3, 5)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    positions = range(len(trace_size_df))
    
    width = 0.3
    # Plot the goal size
    ax.bar([p - width / 2 for p in positions], trace_size_df["goal_size"], width=width, label="GOAL (ATLAHS)", color="lightseagreen")

    # Plot the chakra size
    ax.bar([p + width / 2 for p in positions], trace_size_df["chakra_size"], width=width, label="Chakra (ATLAHS)", color="salmon")

    margin = 1
    # Labels the bars with the trace size
    for i, v in enumerate(trace_size_df["goal_size"]):
        ax.text(i - width / 2, v + margin,
                f"{v:.1f}", ha="center", va="center", color="black",
                weight="bold", fontsize=FONT_SIZE)
    
    for i, v in enumerate(trace_size_df["chakra_size"]):
        ax.text(i + width / 2, v + margin,
                f"{v:.1f}", ha="center", va="center", color="black",
                weight="bold", fontsize=FONT_SIZE)
    
    # Set the x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, ha="center", fontsize=FONT_SIZE + 1)

    y_lim_max = max(trace_size_df["goal_size"].max(), trace_size_df["chakra_size"].max())
    ax.set_ylim(0, y_lim_max * 1.1)

    
    # Set the y-axis label
    ax.set_ylabel("Trace Size (GiB)", fontsize=FONT_SIZE + 1)

    # Set the title
    ax.set_title("Trace Size Comparison", fontsize=FONT_SIZE + 2)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)

    ax.legend(prop={"size": FONT_SIZE})

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "validation_ai_trace_size.pdf"), dpi=300)
    plt.close()
    print_success(f"Saved AI trace size comparison to {os.path.join(plots_dir, 'validation_ai_trace_size.pdf')}")


def plot_validation_exp(data_dir: str, plots_dir: str) -> None:
    """
    Plot the results of the validation experiment.
    """
    validation_data_dir = os.path.join(data_dir, "validation")
    assert os.path.exists(validation_data_dir), f"Validation data directory {validation_data_dir} does not exist."

    hpc_data_dir = os.path.join(validation_data_dir, "hpc")
    plot_hpc_validation_exp(hpc_data_dir, plots_dir)

    ai_data_dir = os.path.join(validation_data_dir, "ai")
    plot_ai_validation_exp(ai_data_dir, plots_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the results of the validation experiment.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory containing the results of the validation experiment.')
    args = parser.parse_args()

    print(f"Plotting results from {args.data_dir}...")
    # Check if the `plots` directory exists
    plots_dir = os.path.join(args.data_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot results for the validation experiment
    plot_validation_exp(args.data_dir, plots_dir)

    # Plot results for case studies
    plot_case_studies(args.data_dir, plots_dir)