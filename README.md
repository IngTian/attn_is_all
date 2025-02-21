# Transformer Implementation (Attention Is All You Need)

This project implements the Transformer model as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## Overview

The Transformer architecture revolutionized natural language processing by introducing self-attention mechanisms and eliminating the need for recurrence and convolutions. This implementation aims to provide a clear and educational reproduction of the original paper's architecture.

## Key Components

- Multi-Head Attention
- Positional Encoding
- Feed-Forward Networks
- Layer Normalization
- Residual Connections

## Project Structure

```
transformer_implementation/
├── model/
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
├── utils/
│   ├── positional_encoding.py
│   └── mask.py
└── train.py
```

## Requirements

```
python==3.9.*
torch>=1.8.0
numpy>=1.19.2
```

## Environment Setup

```bash
# Create a new conda environment
conda create -n transformer python=3.9
conda activate transformer

# Install PyTorch (for CUDA 11.x - adjust based on your GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

## Model Architecture

The implementation follows the original paper's architecture:
- Encoder: 6 layers, each with multi-head attention and feed-forward networks
- Decoder: 6 layers, each with masked multi-head attention, encoder-decoder attention, and feed-forward networks
- Attention Heads: 8 heads per layer
- Model Dimension: 512
- Feed-Forward Dimension: 2048

## Implementation Details

- Uses PyTorch as the deep learning framework
- Implements scaled dot-product attention as described in the paper
- Includes both encoder and decoder stacks
- Supports masking for decoder self-attention

## Citation

```bibtex
@article{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

## License

MIT License
