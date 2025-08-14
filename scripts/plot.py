import os
import argparse
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib
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


def plot_case_studies(data_dir: str, plots_dir: str) -> None:
    # TODO Tommaso: Plot the results for case studies
    pass


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
        measured_runtime_df = pd.DataFrame(columns=["workload", "runtime"])
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