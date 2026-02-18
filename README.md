## entropyestimators

### Overview

Code for the paper 'Know Your Limits: Entropy Estimation Modeling for Compression and Generalization', found here: https://arxiv.org/abs/2511.10618

### Quickstart

To run code using a GPU-accelerated node, spin up a venv, install dependencies via `uv pip install requirements.txt`, run the driver code in the `entropyestimators` directory. Note that the requirements expect a CUDA device capability of at least 7.0, Python >=3.10, CUDA runtime major version of 12 and driver of at least 535.xxx.xx

All driver code is compatible with using a GPU-accelerated server either via Distributed Data Parallel as follows,

```bash
$ torchrun --nproc_per_node {n_gpus} {training_script.py}
```

or Fully Sharded Data Parallel,

```bash
$ accelerate launch --config_file "configs/fsdp_config.yaml" {training_script.py}
```

or Deepspeed ZeRO stage 3

```bash
$ accelerate launch --config_file "configs/zero_config.yaml" {training_script.py}
```

### Format

Most code is divided into files containing model architectures (e.g. `entropyestimators/mixer_clm.py`) and files specifying model dimensions, datasets, and optimizations (e.g. `mixer_trainer_fineweb.py`). Datasets are typically pre-tokenized with a tokenizer trained for a particular domain in question.

If you want to train a tokenizer on a corpus, use `fineweb_tokenizer_trainer.py` and subsitute your corpus as necessary.
