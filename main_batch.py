import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch
import torch.multiprocessing as mp

from main import main

df = pd.read_csv("./hyperparams/global_best_params.csv")

# Remove rows that have the following values in "repr_model" column
remove_values = ["HIMP (b)", "XIMP (a)", "XIMP (b)", "XIMP (c)"]
df = df[~df["repr_model"].isin(remove_values)]

# Remove unused columns
df = df.drop(columns=["mean_val_loss", "std_val_loss", "source_file", "mae_test_scaffold"])

# Cast radius from float -> int
df["radius"] = df["radius"].astype("Int64")

# Define seeds
seeds = (42, 7, 123, 2025, 99, 31415, 2718, 404, 1337, 8888)

# Duplicate each row in dataframe by len(seeds) and add those
df = df.loc[df.index.repeat(10)].reset_index(drop=True)
df["seed"] = seeds * (len(df) // len(seeds))

# Convert df to dictionary
dic = df.to_dict(orient="records")


# --- Your "work" function -----------------------------------------------------
def run_job(dic):
    """
    Replace this with your real work.
    It's executed in a separate process.
    """
    start = time.time()
    main(dic)
    return f"job done in {time.time() - start:.2f}s"


# --- Simple dispatcher --------------------------------------------------------
def dispatch(jobs, per_run_cpus=1):
    """
    Dispatch a list/iterable of jobs onto the machine.
    Each run is conceptually assigned `per_run_cpus` CPUs by limiting
    overall concurrency to os.cpu_count() // per_run_cpus.
    """
    total_cpus = os.cpu_count() or 1
    max_workers = max(1, total_cpus // per_run_cpus)

    print(
        f"Detected {total_cpus} CPUs → running up to {max_workers} jobs at once "
        f"(~{per_run_cpus} CPUs per job)."
    )

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_job, dic) for dic in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# --- Example usage ------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # Queue of individual runs (here: just job IDs)
    torch.set_num_threads(1)
    # job_queue = list(range(1, 11))  # 10 jobs
    job_queue = dic

    outputs = dispatch(job_queue, per_run_cpus=1)
    for line in outputs:
        print(line)
