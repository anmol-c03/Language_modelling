
# GPT-2
# Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#Usage)
4. [Modules](#modules)
5. [References](#references)

## Project Overview

This project implements a GPT-2 model for text generation. It includes modules for preprocessing data, building the transformer model, and generating text using bigram language models.This implementation is purely for my own personal growth and learning purpose.This is inspired from Karpathey's nanoGPT.

For visualization, one can consult [this resource](https://bbycroft.net/llm).



## Installation

To use this project, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/anmol-c03/Language_modelling.git
```

2. Navigate to the project directory:
```bash
cd gpt_2
```


3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

This project can be used for training and generating text with GPT-2 models. Here's how to use it:

1. Preprocess your text data using the `dataprep` module.
2. Train the GPT-2 model using the preprocessed data.
3. Use the trained model to generate text based on input prompts.
4. Fine-tune the model for specific tasks or domains if needed.

For detailed usage instructions, refer to the references given below and example scripts provided in each module.

## Modules

### 1. Bigram Model (`bigram.py`)

The `bigram` is a character-level model implemented in PyTorch for predicting the next token in a text sequence based on the preceding token. It tokenizes input text data, encodes it into integers, and trains the model using the AdamW optimizer with cross-entropy loss. The model architecture comprises an embedding layer mapping tokens to embeddings. Training proceeds for a fixed number of steps, with periodic evaluation on training and validation data. 

### 2. Data Preprocessing (`dataprep`)

The `dataprep` module handles data preprocessing tasks such as tokenization, encoding, and batching. It prepares text data for training the GPT-2 model.

### 3. Transformer Model (`transformer_model`)

The `transformer_model` module implements the core components of the GPT-2 model, including the transformer architecture, attention mechanism, positional encoding, and decoding layers.


## References
Attention mechanism https://arxiv.org/pdf/1706.03762 by Vaswani et al.
weight tying concpets https://arxiv.org/pdf/1608.05859v3
nanoGPT https://github.com/karpathy/nanoGPT


