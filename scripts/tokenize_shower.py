import argparse
import logging
import shutil
from pathlib import Path

import vector
import yaml
from omegaconf import OmegaConf

from gabbro.data.data_tokenization import reconstruct_shower_file, tokenize_shower_file

# import gabbro.plotting as jplt

vector.register_awkward()

logger = logging.getLogger(__name__)


def copy_checkpoint(ckpt_path, directory):
    """Copies a checkpoint file to a specified directory."""
    ckpt_path = Path(ckpt_path)
    directory = Path(directory)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

    directory.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    new_ckpt_path = directory / "model.ckpt"  # Maintain the original filename

    try:
        shutil.copy2(ckpt_path, new_ckpt_path)  # Use shutil.copy2 to preserve metadata
    except Exception as e:
        raise RuntimeError(f"Error copying checkpoint file: {e}")

    print(f"Checkpoint file copied to: {new_ckpt_path}")


def load_config(config_file_path):
    """Loads the YAML configuration file and returns it as a dictionary."""
    config_path = Path(config_file_path)
    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in configuration file: {e}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config_file)

    network = config["data"]["network"]
    epoch = config["data"]["epoch"]
    tokenize = config["data"]["tokenize"]
    reconstruct = config["data"]["reconstruct"]
    save_config = config["data"]["save_config"]
    save_ckpt = config["data"]["save_ckpt"]
    train = config["data"]["train"]
    test = config["data"]["test"]
    val = config["data"]["val"]

    print(f"networkfile: {network}")
    print(f"epoch: {epoch}")
    # network = "2024-06-06_10-04-25_max-cmsg010_OutermostOsteopetrosis"
    # epoch = "epoch_013_loss_13236648851865075712.00000.ckpt"
    filename_in_train = "/beegfs/desy/user/korcariw/CaloClouds/dataset/showers/photons_10_100GeV_float32_sorted_train.h5"
    filename_in_test = "/beegfs/desy/user/korcariw/CaloClouds/dataset/showers/photons_10_100GeV_float32_sorted_test.h5"
    filename_in_val = "/beegfs/desy/user/korcariw/CaloClouds/dataset/showers/photons_10_100GeV_float32_sorted_val.h5"

    ckpt_path = (
        f"/beegfs/desy/user/rosehenn/gabbro_output/TokTrain/runs/{network}/checkpoints/{epoch}"
    )

    directory = f"/beegfs/desy/user/rosehenn/gabbro/compare/{network}"
    filename_out_test = (
        f"/beegfs/desy/user/rosehenn/gabbro/compare/{network}/tokenized_test.parquet"
    )
    filename_out_train = (
        f"/beegfs/desy/user/rosehenn/gabbro/compare/{network}/tokenized_train.parquet"
    )
    filename_out_val = f"/beegfs/desy/user/rosehenn/gabbro/compare/{network}/tokenized_val.parquet"

    filename_out_2_train = (
        f"/beegfs/desy/user/rosehenn/gabbro/compare/{network}/reconstructed_train.parquet"
    )
    filename_out_2_test = (
        f"/beegfs/desy/user/rosehenn/gabbro/compare/{network}/reconstructed_test.parquet"
    )
    filename_out_2_val = (
        f"/beegfs/desy/user/rosehenn/gabbro/compare/{network}/reconstructed_val.parquet"
    )
    config_path = f"/beegfs/desy/user/rosehenn/gabbro_output/TokTrain/runs/{network}/config.yaml"

    if tokenize:
        print("Tokenizing data...")
        if train:
            # this function will save the tokenized data to a parquet file in the desired location
            tokens_int, p4s_original, data_showers = tokenize_shower_file(
                filename_in=filename_in_train,
                model_ckpt_path=ckpt_path,
                filename_out=filename_out_train,
                add_start_end_tokens=True,
                energy_sorting=True,
                n_load=760000,
            )
            print("Tokenized training data saved to:", filename_out_train)

        if test:
            tokens_int, p4s_original, data_showers = tokenize_shower_file(
                filename_in=filename_in_test,
                model_ckpt_path=ckpt_path,
                filename_out=filename_out_test,
                add_start_end_tokens=True,
                energy_sorting=True,
                n_load=760000,
            )
            print("Tokenized testing data saved to:", filename_out_test)

        if val:
            tokens_int, p4s_original, data_showers = tokenize_shower_file(
                filename_in=filename_in_val,
                model_ckpt_path=ckpt_path,
                filename_out=filename_out_val,
                add_start_end_tokens=True,
                energy_sorting=True,
                n_load=760000,
            )
            print("Tokenized validation data saved to:", filename_out_val)

    if reconstruct:
        print("Reconstructing data...")
        if train:
            data, p4data = reconstruct_shower_file(
                filename_in=filename_out_train,
                model_ckpt_path=ckpt_path,
                config_path=config_path,
                filename_out=filename_out_2_train,
                start_token_included=True,
                end_token_included=True,
                shift_tokens_by_minus_one=True,
                print_model=False,
                device="cuda",
                merge_duplicates=True,
            )
            print("Reconstructed training data saved to:", filename_out_2_train)
        if test:
            data, p4data = reconstruct_shower_file(
                filename_in=filename_out_test,
                model_ckpt_path=ckpt_path,
                config_path=config_path,
                filename_out=filename_out_2_test,
                start_token_included=True,
                end_token_included=True,
                shift_tokens_by_minus_one=True,
                print_model=False,
                device="cuda",
                merge_duplicates=True,
            )
            print("Reconstructed testing data saved to:", filename_out_2_test)

        if val:
            data, p4data = reconstruct_shower_file(
                filename_in=filename_out_val,
                model_ckpt_path=ckpt_path,
                config_path=config_path,
                filename_out=filename_out_2_val,
                start_token_included=True,
                end_token_included=True,
                shift_tokens_by_minus_one=True,
                print_model=False,
                device="cuda",
                merge_duplicates=True,
            )
            print("Reconstructed validation data saved to:", filename_out_2_val)

    if save_config:
        output_dir = Path(directory)
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the output directory if it doesn't exist
        # Extract the original filename
        original_filename = Path(config_path).name
        # Construct the new file path within the output directory
        new_config_path = output_dir / original_filename

        with new_config_path.open("w") as f:
            config_of_network = load_config(config_path)
            OmegaConf.save(config_of_network, f)

        print(f"Modified configuration saved to: {new_config_path}")

    if save_ckpt:  # Or a Path object
        copy_checkpoint(ckpt_path, directory)


if __name__ == "__main__":
    main()
