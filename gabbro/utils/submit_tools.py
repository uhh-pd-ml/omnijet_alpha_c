"""Tools to help with submitting jobs to the cluster."""

import argparse
import itertools
import os
import re


def from_dict(dct):
    """Return a function that looks up keys in dct."""

    def lookup(match):
        key = match.group(1)
        return dct.get(key, f"<{key} not found>")

    return lookup


def convert_values_to_strings(dct):
    """Convert all values in dct to strings."""
    return {k: str(v) for k, v in dct.items()}


def replace_placeholders(file_in, file_out, subs):
    """Replace placeholders of the form @@<placeholder_name>@@ in file_in and write to file_out.

    Parameters
    ----------
    file_in : str
        Input file.
    file_out : str
        Output file.
    subs : dict
        Dictionary mapping placeholders to their replacements, i.e. `{"dummy": "foo"}
        will replace @@dummy@@ with foo.
    """
    with open(file_in) as f:
        text = f.read()
    with open(file_out, "w") as f:
        f.write(re.sub("@@(.*?)@@", from_dict(subs), text))


def calc_batches_per_node(config, mode="train"):
    # convert parameters to floats
    n_files_at_once = float(config[f"{mode}_n_files_at_once"])
    n_jets_per_file = float(config[f"{mode}_n_jets_per_file"])
    batch_size = float(config["batch_size"])
    num_gpus_per_node = float(config["num_gpus_per_node"])
    num_nodes = float(config["num_nodes"])

    batches_per_node = (
        n_files_at_once * n_jets_per_file / batch_size / num_gpus_per_node / num_nodes
    )
    config[f"limit_{mode}_batches"] = str(int(batches_per_node))


def create_job_scripts_from_template_and_submit(
    hparams_to_try,
    hparams_defaults,
    job_file_template="job_template.sh",
):
    """Create job scripts from a template and submit them to the cluster. This function also
    initialized as argument parser under the hood. I.e. the following command line arguments are
    available if this function is.

    used in your script:
    --dry_run: Don't actually submit the jobs.
    --print_run_script: Print the run script of the individual jobs to the console.
    --use_bash: Run the job script with bash instead of sbatch (for debugging on
        interactive nodes).


    Parameters
    ----------
    hparams_to_try : dict
        Dictionary mapping hyperparameters to lists of values to try.
        Those parameters have to appear in the job_file_template with the
        placeholders @@<parameter_name>@@.
    hparams_defaults : dict
        Dictionary mapping hyperparameters to default values.
    job_file_template : str
        Path to the template file.
    """

    parser = get_job_script_parser()
    args = parser.parse_args()

    for k, v in hparams_defaults.items():
        if k not in hparams_to_try:
            hparams_to_try[k] = v

    combinations = list(itertools.product(*hparams_to_try.values()))

    for i, combination in enumerate(combinations):
        subs = dict(zip(hparams_to_try.keys(), combination))
        subs = convert_values_to_strings(subs)
        print(100 * "-")
        print(f"Config {i+1}/{len(combinations)}:")
        # ----
        # check if it was requested to calculate the limit_train_batches or limit_val_batches
        limit_train_batches = subs.get("limit_train_batches")
        limit_val_batches = subs.get("limit_val_batches")
        if limit_train_batches is not None or limit_val_batches is not None:
            if isinstance(limit_train_batches, str):
                if limit_train_batches == "calculate":
                    print("Calculating limit_train_batches from other parameters.")
                    calc_batches_per_node(subs)
            if isinstance(limit_val_batches, str):
                if limit_val_batches == "calculate":
                    print("Calculating limit_val_batches from other parameters.")
                    calc_batches_per_node(subs, mode="val")
        # ----
        # print key-value pairs formatted as a table
        max_key_len = max(len(k) for k in subs.keys())
        for k, v in subs.items():
            print(f"{k:>{max_key_len}} : {v}")
        print(100 * "-")
        replace_placeholders(job_file_template, "run_tmp.sh", subs)

        # if "use_bash" is true, remove "srun " from the run script
        if args.use_bash:
            with open("run_tmp.sh") as f:
                run_script = f.read()
            run_script = run_script.replace("srun ", "")
            with open("run_tmp.sh", "w") as f:
                f.write(run_script)

        if args.print_run_script:
            print("Run script:")
            print("-----------")
            with open("run_tmp.sh") as f:
                print(f.read())
        if not args.dry_run:
            if args.use_bash:
                os.system("bash run_tmp.sh")  # nosec
            else:
                os.system("sbatch run_tmp.sh")  # nosec


def get_job_script_parser():
    """Return an argument parser for job scripts.

    Returns
    -------
    argparse.ArgumentParser
        Argument parser for job scripts with the following flags:
        --dry_run: Don't actually submit the jobs.
        --print_run_script: Print the run script of the individual jobs to the console.
        --use_bash: Run the job script with bash instead of sbatch (for debugging on interactive nodes).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Don't actually submit the jobs.")
    parser.add_argument(
        "--print_run_script",
        action="store_true",
        default=False,
        help="Print the run script of the individual jobs to the console.",
    )
    parser.add_argument(
        "--use_bash",
        action="store_true",
        default=False,
        help="Run the job script with bash instead of sbatch (for debugging on interactive nodes).",
    )
    return parser
