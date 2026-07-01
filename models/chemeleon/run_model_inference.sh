#!/bin/bash
# Script to run OpenADMET predictions in a standalone shell

# Exit immediately if a command exits with a non-zero status
set -e

# --- Environment variables ---
export TABPFN_TELEMETRY_OPTOUT=1
export OADMET_NO_RICH_LOGGING=1

openadmet predict \
    --input-path compounds_for_inference.csv \
    --input-col OPENADMET_CANONICAL_SMILES \
    --model-dir anvil_training/ \
    --output-csv predictions.csv \
    --accelerator cpu