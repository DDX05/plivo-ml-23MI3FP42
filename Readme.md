LLM Speedrun 120

Overview

This repository contains the source code and configuration for a 1.96M parameter transformer model, developed within a 2,000-step training constraint. The project focuses on optimizing Bits Per Byte (BPB) through architectural efficiency, custom tokenization, and modern training heuristics on CPU-only infrastructure.

System Requirements

Ubuntu (Target environment)

Python 3.12

PyTorch

Project Structure

data/: Contains the training and evaluation corpus.

model.py: Transformer architecture implementing RMSNorm, Rotary Positional Embeddings (RoPE), and SwiGLU activations.

tokenizer.py: Custom Byte-Pair Encoding (BPE) implementation.

train.py: Training entry point with Cosine Annealing LR scheduling and gradient accumulation.

evaluate.py: Evaluation script for Bits Per Byte (BPB) calculation.

ckpt.pt: The finalized model checkpoint (generated post-training).

Setup and Installation

1. Environment Configuration

Create a virtual environment using Python 3.12 to maintain dependency isolation:

sudo apt update
sudo apt install python3.12-venv
python3.12 -m venv env
source env/bin/activate


2. Dependency Installation

Install the necessary requirements within the virtual environment:

pip install torch


Training

To train the model, ensure the training corpus is placed in data/train_corpus.txt. Execute the training script:

python train.py


Note: The script automatically performs BPE tokenizer training and generates the ckpt.pt output file.

Evaluation

To calculate the final BPB score, run the evaluation script against the held-out development set:

python evaluate.py -checkpoint ckpt.pt --text_file data/dev_eval.txt


Architectural Highlights

Parameter Efficiency: Implemented weight tying between the input embedding layer and the final LM head to maximize network depth within the 2,000,000 parameter constraint.

Tokenizer Optimization: Replaced the baseline byte-level tokenizer with a BPE tokenizer (Vocab=1024), reducing sequence length and improving semantic density for Hindi (Devanagari) characters.

Modern Primitives: Integrated RoPE for positional encoding and SwiGLU for MLP layers, ensuring faster convergence within the strict step budget.

Optimization Strategy: Utilized AdamW with a Cosine Annealing schedule and gradient accumulation (simulating a batch size of 64) to stabilize training on CPU.

Deliverables Checklist

[x] SUMMARY.html

[x] RUNLOG.md

[x] NOTES.md

[x] Model Source Code (model.py, tokenizer.py, train.py, evaluate.py)

[x] Output Checkpoint (ckpt.pt)