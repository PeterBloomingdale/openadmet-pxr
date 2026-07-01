---
license: apache-2.0
language:
- en
tags:
- chemistry
- biology
---

> [!WARNING]  
> Due to the sparsity of PXR pEC50s on ChEMBL, this is a pretty terrible model. To be used as a baseline ONLY.

In this repo, we have our baseline model. It is a **single task CheMeleon** model trained on **pEC50** data curated from ChEMBL for `PXR`.

It is a no split model, meaning it has been trained with no data allocated to validation and test sets and with just a training set of `1.0`.  


# Getting Started

## Downloading the model from Github

1. clone the model repo:
```
git clone https://huggingface.co/openadmet/pxr-chemeleon-baseline/
```
2. Change to the repo directory. Ensure you have `git lfs` installed for the repo and get the large model files:
```
git lfs install
git lfs pull
```
3. You are now ready to use the model!

### Running the model with your environment
We *highly* recommend you have the Anvil framework from `openadmet-models` installed in an environment (called `openadmet-models`) for ease of use and full utilization of OpenADMET's models. The installation instructions can be found [here](https://docs.openadmet.org/en/latest/installation.html).

Alternatively, you can also use Docker to spin up a containerized pre-installed environment to run `openadmet-models`. Just be sure you are mounting the correct folder (`./pxr-chemeleon-baseline`) where you've downloaded the model.

## Downloading & running the model with Docker

If you're using a gpu, run:
```
docker run -it --user=root --rm  \
    -v ./pxr-chemeleon-baseline:/home/mambauser/model:rw \
    --runtime=nvidia 
    --gpus 
    all ghcr.io/openadmet/openadmet-models:main 
```
Otherwise, for cpu only:
```
docker run -it --user=root --rm  \
    -v ./pxr-chemeleon-baseline:/home/mambauser/model:rw \
    all ghcr.io/openadmet/openadmet-models:main 
```


## Using the model
We will use this model for inference or, to predict the pIC50s of a set of molecular compounds unseen to the model.
You can do this either **inside the docker container** as per the instructions above, or if you have installed `openadmet-models` on your own computer, you can use the appropriate environment.

For demonstration purposes, we have provided a small subset of compounds from a [ZINC](https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/zinc_10K.smi) deck in the file `compounds_for_inference.csv`. 

The generic command to run our inference pipeline is:
```bash
    openadmet predict \
        --input-path <the path to the data to predict on> \
        --input-col <the column to of the data to predict on, often SMILES> \
        --model-dir <the anvil_training directory of the model to predict with> \
        --output-csv <the path to an output CSV to save the predictions to> \
        --accelerator <whether to use gpu or cpu, defaults to gpu>
```
You can run this directly in your command line, OR you can use the bash script we've provided, `run_model_inference.sh`.

For our working example, this command becomes:
```bash
openadmet predict \
    --input-path compounds_for_inference.csv \
    --input-col OPENADMET_CANONICAL_SMILES \
    --model-dir anvil_training/ \
    --output-csv predictions.csv \
    --accelerator cpu
```
You can easily substitute your own set of compounds, simply modify the `--input-path` and `--input-col` arguments for your specific dataset.
If you want to use a GPU (reccomended) substitute `accelerator gpu` in the above.

In our example, this outputs a file called `predictions.csv` which will have predicted (the `OADMET_PRED` columns) pEC50 values for the PXR target:
```
OADMET_PRED_chemprop-chembl_pchembl_value_mean,
OADMET_STD_chemprop-chembl_pchembl_value_mean
```
**NOTE** In this example, the standard deviation (`OADMET_STD`) columns are empty because uncertainty cannot be estimated unless training an ensemble of models. For further details, visit our [docs](https://demos.openadmet.org/en/latest/demos/04_Ensemble_Model_Training/04_Ensemble_Model_Training_Active_Learning.html).