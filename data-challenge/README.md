---
license: apache-2.0
language:
- en
tags:
- chemistry
- drug-discovery
- ADMET
- molecular-properties
- blind-challenge
- computational-chemistry

pretty_name: OpenADMET PXR Induction Blind Challenge
size_categories:
- 10K<n<100K
task_categories:
- tabular-regression
annotations_creators:
- expert-generated
source_datasets:
- original
configs:
- config_name: default
  data_files:
  - split: train
    path: pxr-challenge_TRAIN.csv
  - split: test
    path: pxr-challenge_TEST_BLINDED.csv
- config_name: counter_assay
  data_files:
  - split: train
    path: pxr-challenge_counter-assay_TRAIN.csv
- config_name: structure
  data_files:
  - split: test
    path: pxr-challenge_structure_TEST_BLINDED.csv
- config_name: single_concentration
  data_files:
  - split: train
    path: pxr-challenge_single_concentration_TRAIN.csv
---

# PXR Challenge Train/Test Dataset

A high-quality experimental dataset for predicting human Pregnane-X Receptor (PXR) induction, comprising over 11,000 compounds screened using a high-fidelity in-house assay. This is the largest publicly available PXR activity dataset, released as part of the [OpenADMET PXR Induction Blind Challenge](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/).

**Blog post:** [Announcing the Next OpenADMET Blind Challenge: Predicting PXR Induction](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)

**Challenge Space:** [openadmet/pxr-challenge](https://huggingface.co/spaces/openadmet/pxr-challenge)

**Challenge period:** April 1 – July 1, 2026

**Produced by:** OpenADMET

## CHANGELOG

* Updated 2026-04-09 dropping some compounds, fixing minor confidence interval issues and improving naming join. See [here](https://docs.google.com/document/d/14-2EL4Zk8g3NNO7bi33gA3fHz_ef98bBigunJMH3Ui4/edit?usp=sharing) for more details.  

### Dataset contents

| Config | Split | Description |
|---|---|---|
| `default` | `train` | Primary assay training set (pEC50, Emax) |
| `default` | `test` | 513-compound blinded test set |
| `counter_assay` | `train` | PXR-null counter-assay training data |
| `structure` | `test` | 78 fragment-sized molecules with X-ray crystal structures |
| `single_concentration` | `train` | Single-concentration screening data (log2 fold change) |

## Loading with Hugging Face `datasets`

```python
from datasets import load_dataset

# Default config (primary assay)
ds = load_dataset("openadmet/pxr-challenge-train-test")
train = ds["train"]
test  = ds["test"]

# Counter-assay config
ds_counter = load_dataset("openadmet/pxr-challenge-train-test", "counter_assay")
train_counter = ds_counter["train"]

# Structure config
ds_structure = load_dataset("openadmet/pxr-challenge-train-test", "structure")
test_structure = ds_structure["test"]

# Single-concentration config
ds_single = load_dataset("openadmet/pxr-challenge-train-test", "single_concentration")
train_single = ds_single["train"]
```

## Loading directly with pandas

```python
import pandas as pd

train         = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
test          = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")
train_counter  = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv")
test_structure = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_structure_TEST_BLINDED.csv")
train_single   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv")
```
