# OmniJet-α Calo: Applying OJA to calorimeter data

<div style="text-align: center;">
Joschka Birk, Frank Gaede, Anna Hallin, Gregor Kasieczka, Martina Mozzanica, Henning Rose

[![arXiv](https://img.shields.io/badge/arXiv-2501.05534-red)](https://arxiv.org/abs/2501.05534)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.2.1-purple)](https://lightning.ai)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-blue)](https://hydra.cc)
</div>

This repository contains the code for the results presented in the paper [`OmniJet-α_C: Learning point cloud calorimeter simulations using generative transformers`](https://arxiv.org/abs/2501.05534)
The documentation of the original OmniJet-α can be found at [uhh-pd-ml/omnijet_alpha](https://github.com/uhh-pd-ml/omnijet_alpha)
**Abstract:**

```
We show the first use of generative transformers for generating calorimeter showers as point clouds
in a high-granularity calorimeter. Using the tokenizer and generative part of the OmniJet-α model,
we represent the hits in the detector as sequences of integers. This model allows variable-length
sequences, which means that it supports realistic shower development and does not need to be
conditioned on the number of hits. Since the tokenization represents the showers as point clouds,
the model learns the geometry of the showers without being restricted to any particular voxel grid.
```

## Table of Contents

- [How to run the code](#how-to-run-the-code)
- [Dataset](#dataset)
- [Installation](#installation)
- [Tokenization](#tokenization)
- [Generative training](#generative-training)
- [Transfer learning / Classifier training](#transfer-learning--classifier-training)
- [Citation](#citation)

## How to run the code

### Dataset

Instructions on how to download the dataset can be found in the repository [jet-universe/particle_transformer.](https://github.com/FLC-QU-hep/getting_high)

### Installation

The recommended (and by us tested) way of running the code is to use the provided docker image at jobirk/omnijet on [DockerHub](https://hub.docker.com/r/jobirk/omnijet). The requirements listed in docker/requirements.txt are installed in the conda environment base of the base image (official pytorch image). Thus, you have to make sure that the conda environment is activated when running the code, which can be done with `source /opt/conda/bin/activate`.

An interactive session inside a container can be started by running the following command:

#### On a machine with Singularity

```sh
singularity shell docker://jobirk/omnijet:latest  # start a shell in the container
source /opt/conda/bin/activate  # activate the conda environment in the container
```

#### On a machine with Docker

```sh
docker run -it --rm jobirk/omnijet:latest bash  # start a shell in the container
source /opt/conda/bin/activate  # activate the conda environment in the container
```

Alternatively, you can install the requirements from the `docker/requirements.txt` file, but you'll have to add pytorch to the list of requirements, since this is not included in the `requirements.txt` file (we use the official pytorch image as base image).

Furthermore, you'll have to add/create a `.env` file in the root of the project with the following content:

```sh
JETCLASS_DIR="<path to the jetclass dataset i.e. where the train_100M, val_5M, .. folders are>"
JETCLASS_DIR_TOKENIZED="<path to where you want to save the tokenized jetclass dataset>"

# stuff for hydra
LOG_DIR="<path to log dir>"
COMET_API_TOKEN="<your comet api token>"
HYDRA_FULL_ERROR=1
```

## Tokenization / Reconstruction

To play around with the already-trained VQ-VAE model, you can download the checkpoint (see `checkpoints/README.md` for instructions) and then have a look at the notebook `examples/notebooks/example_tokenize_and_reconstruct_jets.ipynb`.

You can run the training of the VQ-VAE model by running the following command:

```sh
python gabbro/train.py experiment=example_experiment_tokenization
```

To create the tokenized dataset, you can run the following command:

```sh
python python scripts/tokenize_shower_pipeline.py scripts/tokenize_shower.yaml
```

Make sure to adjust the settings in the `tokenize_shower.yaml` to your needs and declare the correct folder for your showers.

## Generative training

To play around with the already-trained generative model, you can download the checkpoint (see `checkpoints/README.md` for instructions) and then have a look at the notebook `examples/notebooks/example_generate_jets.ipynb`.

You can run the generative training of the backbone model by running the following command:

```sh
python gabbro/train.py experiment=example_experiment_backbone_generative
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{birk2025omnijetalphaclearningpoint,
      title = {OmniJet-${\alpha_{ C}}$: Learning point cloud calorimeter simulations using generative transformers},
      author = {Joschka Birk and Frank Gaede and Anna Hallin and Gregor Kasieczka and Martina Mozzanica and Henning Rose},
      year = {2025},
      eprint = {2501.05534},
      archivePrefix = {arXiv},
      primaryClass = {hep-ph},
      url = {https://arxiv.org/abs/2501.05534},
}
```
