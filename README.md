![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 
![Status](https://img.shields.io/badge/Status-Educational-yellow)
![KV Cache](https://img.shields.io/badge/KV_Cache-Enabled-success)
![RoPE](https://img.shields.io/badge/RoPE-Enabled-success)

# English to French Translation Transformer from Scratch

This repository implements a **full Transformer model from scratch** in PyTorch featuring **Key-Value (KV) Caching** for efficient inference and **Rotary Positional Embedding (RoPE)** to allow the transformer model to understand relative positions and not just absolute positions. It is trained on **character-level English → French machine translation**.

It includes:
- `models/transformer.py` – Complete encoder-decoder Transformer architecture
- `data/dataloader.py` – Data loading, character-level tokenization, padding, and batching
- `inference/generate.py` – Translating english sentence to french
- `train.py` – Training loop with checkpointing and simple inference testing
- `app.py` – Streamlit application for model demo

The model follows the original *"Attention is All You Need"* (Vaswani et al., 2017) architecture with modern improvements:
- *Pre-Normalization* (Pre-Norm) for stable training.
- *Key-Value Caching* for *O(1)* generation latency.
- *Rotary Positional Embedding* for understanding relative positions.
- *Sliding Window Context Management* to handle long-sequence generation without memory crashes.

## Model Architecture

**Transformer Encoder-Decoder** (seq2seq)

### Hyperparameters
- **Embedding Dimension**: `n_embd = 256`
- **Context Length**: `context_length = 64`
- **Number of Layers**: `n_layers = 6` (both encoder and decoder)
- **Number of Attention Heads**: `n_heads = 8` (head size = 256 / 8 = 32)
- **Feed-Forward Hidden Size**: 4 × 256 = 1024
- **Dropout**: `0.2`

### Key Components

1. **Token + Position Embeddings**
   - Learned token embedding table: `(vocab_size, n_emb)` for both src ids and tgt ids
   - Rotary Position Embedding: Parameter-free sinusoidal rotations on attention queries and keys

2. **Encoder** (6 layers)
   - Unmasked multi-head **self-attention**
   - Feed-forward network
   - Pre-layer normalization + residual connections
   - Supports source padding mask

3. **Decoder** (6 layers)
   - Masked multi-head **self-attention** (causal + padding)
   - Multi-head **cross-attention** (queries from decoder, keys/values from encoder)
   - Supports KV Caching for masked multi head self-attention
   - Feed-forward network
   - Pre-layer normalization + residual connections
   - Supports target padding mask and source padding mask

4. **Output Head**
   - Final LayerNorm → Linear projection to French vocabulary size

### Total Parameters

Approximately **~9–12 million** (exact count printed during training).

## Features

- Full support for **padding masks** (both source and target)
- **Key-Value (KV) Caching** for *O(1)* generation latency
- **Rotary Positional Embedding (RoPE)** which replaces absolute embeddings with relative positional encoding, allowing for better generalization on longer sequences
- **Causal masking** in decoder self-attention
- **Teacher forcing** during training
- **Autoregressive generation** with:
  - Greedy decoding (default)
  - Optional sampling with temperature and top-k
  - Early stopping on `<eos>`
- Label smoothing (0.1) in cross-entropy loss
- Best checkpoint saving based on validation loss
- Character-level tokenization (no subword/BPE)

## Dataset

Uses a CSV file: `eng_french.csv` with two columns:

```csv
English words/sentences,French words/sentences
"I am a student","Je suis étudiant"
...
```
Special tokens added:

- `<pad>` – padding
- `<unk>` – unknown character
- `<bos>` – beginning of sequence (decoder start)
- `<eos>` – end of sequence

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas
- CUDA-capable GPU recommended (falls back to CPU)

## Project Structure

```text
Transformer-from-scratch/
│
├── data/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── eng_french.csv
│   └── eng_to_french.zip
│
├── inference/
│   ├── __init__.py
│   └── generate.py
│
├── models/
│   ├── __init__.py
│   └── transformer.py
│
├── notebooks/
│   └── data_analysis.ipynb
│
├── saved_models/
│   └── best_checkpoint.pt
│
├── app.py
├── train.py
├── README.md
├── requirements.txt
└── .gitignore

```

# Usage

## Prepare your dataset

Place your parallel English-French sentences in `eng_french.csv`:

```csv
English words/sentences,French words/sentences
Hello,Salut
How are you?,Comment vas-tu ?
...
```

---

## Train the model

```bash
python train.py
```

The script will:

- Build character vocabularies for English and French
- Train for 15000 iterations
- Evaluate on train/val splits every 500 steps
- Save the best model as `saved_models/best_checkpoint.pt`
- After training, you can run `app.py` for demo of the model

---

## To translate your own sentences after training:

```bash
streamlit run app.py
```
Run this for demo of the model, in which you can provide your input for translation

---

## Example Output (after sufficient training)

```text
>> English: I am a student
>> French: je suis étudiant
>> English: I am going home
>> French: je rentre à la maison
```

---

## Customization

Edit hyperparameters in `models/transformer.py` at the top of the file:

```python
n_embd = 256
context_length = 64
dropout = 0.2
n_heads = 8
n_layers = 6
```

Edit hyperparameters in `train.py` at the top of the file:

```python
lr = 1.5e-4
max_iters = 15_000
warmup_steps = 1_000
eval_iters = 200
eval_interval = 500
```

Edit hyperparameters in `data/dataloader.py` at the top of the file:

```python
context_length = 64
batch_size = 32
```

---

## Notes

- This is a character-level model — it learns to translate letter by letter.
- Training from scratch on small datasets (~10k–100k sentences) will give reasonable results.
- For better performance: use larger dataset, longer training, bigger model, or switch to subword tokenization.
- The attention implementation is flexible and supports self-attention, cross-attention, and masking.

---

## References

1.  **Attention Is All You Need** (Vaswani et al., 2017)
    * [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
    * *The foundational paper that introduced the Transformer architecture and Self-Attention mechanism.*

2.  **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
    * [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
    * *The paper introducing RoPE, which I implemented to replace standard absolute positional embeddings.*

## Credits
Inspired by:

- Andrej Karpathy’s nanoGPT and *"[Let's build GPT](https://youtu.be/kCc8FmEb1nY)"* lecture
- PyTorch official Transformer tutorials