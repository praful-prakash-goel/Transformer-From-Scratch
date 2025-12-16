# English to French Translation Transformer from Scratch

This repository implements a **full Transformer model from scratch** in PyTorch for **character-level English → French machine translation**.

It includes:
- `models/transformer.py` – Complete encoder-decoder Transformer architecture
- `data/dataloader.py` – Data loading, character-level tokenization, padding, and batching
- `inference/generate.py` – Translating english sentence to french
- `train.py` – Training loop with checkpointing and simple inference testing
- `app.py` – Streamlit application for model demo

The model follows the original *"Attention is All You Need"* architecture with modern improvements (pre-norm, proper masking).

## Model Architecture

**Transformer Encoder-Decoder** (seq2seq)

### Hyperparameters
- **Embedding Dimension**: `n_embd = 256`
- **Context Length**: `context_length = 64`
- **Number of Layers**: `n_layers = 6` (both encoder and decoder)
- **Number of Attention Heads**: `n_heads = 6` (head size = 256 / 6 ≈ 42)
- **Feed-Forward Hidden Size**: 4 × 256 = 1024
- **Dropout**: `0.2`

### Key Components

1. **Shared Learned Positional Embeddings**
   Added to both source (English) and target (French) token embeddings.

2. **Encoder** (6 layers)
   - Unmasked multi-head **self-attention**
   - Feed-forward network
   - Pre-layer normalization + residual connections
   - Supports source padding mask

3. **Decoder** (6 layers)
   - Masked multi-head **self-attention** (causal + padding)
   - Multi-head **cross-attention** (queries from decoder, keys/values from encoder)
   - Feed-forward network
   - Pre-layer normalization + residual connections
   - Supports target padding mask and source padding mask

4. **Output Head**
   - Final LayerNorm → Linear projection to French vocabulary size

### Total Parameters

Approximately **~9–12 million** (exact count printed during training).

## Features

- Full support for **padding masks** (both source and target)
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
│   ├── dataset.py
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
- Train for 5000 iterations
- Evaluate on train/val splits every 500 steps
- Save the best model as `saved_models/best_checkpoint.pt`
- After training, test translation on two example sentences

---

## Example Output (after sufficient training)

```text
>> English: i am a student
>> French: je suis étudiant
>> English: I am going home
>> French: je rentre à la maison
```

---

## Customization

Edit hyperparameters in `models/transformer.py` or at the top of files:

```python
n_embd = 256
context_length = 64
n_layers = 6
n_heads = 6
dropout = 0.2
batch_size = 32
max_iters = 5_000
lr = 3e-4
```

---

## To translate your own sentences after training:

```python
sentence = "Your English sentence here"
src_ids = torch.tensor(
    eng_tokenizer.encode(sentence),
    dtype=torch.long
).unsqueeze(0).to(device)

src_mask = src_ids != eng_tokenizer.pad_idx

idx = torch.tensor([[fr_tokenizer.bos]], dtype=torch.long).to(device)

generated = model.generate(
    src_ids=src_ids,
    idx=idx,
    max_new_tokens=100,
    src_mask=src_mask,
    eos_token=fr_tokenizer.eos
)

print(fr_tokenizer.decode(generated[0].tolist()))
```

---

## Notes

- This is a character-level model — it learns to translate letter by letter.
- Training from scratch on small datasets (~10k–100k sentences) will give reasonable results.
- For better performance: use larger dataset, longer training, bigger model, or switch to subword tokenization.
- The attention implementation is flexible and supports self-attention, cross-attention, and masking.

---

## Credits

Inspired by:

- Original Transformer paper: *"[Attention is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)"* by Vaswani et al.
- Andrej Karpathy’s nanoGPT and *"[Let's build GPT](https://youtu.be/kCc8FmEb1nY)"* lecture
- PyTorch official Transformer tutorials
