import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

context_length = 1024
batch_size = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "eng_french.csv")
TOKENIZER_NAME = "Helsinki-NLP/opus-mt-en-fr"

class TranslationTokenizer:
    """Adapter exposing the interface expected by the existing project."""

    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer
        self.pad = hf_tokenizer.pad_token_id
        self.unk = hf_tokenizer.unk_token_id
        self.eos = hf_tokenizer.eos_token_id
        self.bos = hf_tokenizer.bos_token_id

        if self.pad is None or self.unk is None or self.eos is None or self.bos is None:
            raise ValueError(
                "Tokenizer must have pad, unk, eos, and bos token IDs. "
                "The dataloader sets a dedicated BOS token before creating this adapter."
            )

    def encode(self, text, add_bos=True, add_eos=True):
        # Do not let Hugging Face automatically add its own special tokens;
        # this adapter adds the exact BOS/EOS markers used by the model.
        ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]
        return ids

    def decode(self, ids, remove_specials=True):
        # Keep the old call signature; generated special tokens are omitted.
        return self.tokenizer.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=remove_specials,
            clean_up_tokenization_spaces=True,
        ).strip()

_hf_source_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
_hf_target_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Add the same dedicated BOS token to both copies. Do not reuse PAD or EOS.
_hf_source_tokenizer.add_special_tokens({"bos_token": "<bos>"})
_hf_target_tokenizer.add_special_tokens({"bos_token": "<bos>"})

# Pin the correct SentencePiece model for each side. Marian's decoder defaults
# to target-side decoding; the source instance is only used for input encoding.
if hasattr(_hf_source_tokenizer, "spm_source"):
    _hf_source_tokenizer.current_spm = _hf_source_tokenizer.spm_source
if hasattr(_hf_target_tokenizer, "spm_target"):
    _hf_target_tokenizer.current_spm = _hf_target_tokenizer.spm_target

eng_tokenizer = TranslationTokenizer(_hf_source_tokenizer)
fr_tokenizer = TranslationTokenizer(_hf_target_tokenizer)
vocab_size_eng = len(_hf_source_tokenizer)
vocab_size_fr = len(_hf_target_tokenizer)

if eng_tokenizer.bos != fr_tokenizer.bos:
    raise ValueError("English and French tokenizers produced different BOS token IDs.")


if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find dataset at {CSV_PATH}. Put eng_french.csv in the same "
        "directory as this dataloader.py."
    )

df = pd.read_csv(CSV_PATH)
eng_text = df['English words/sentences'].tolist()
french_text = df['French words/sentences'].tolist()

# Tokenize each paired sentence using the same pretrained subword vocabulary.
eng_data = [eng_tokenizer.encode(sentence) for sentence in eng_text]
fr_data = [fr_tokenizer.encode(sentence) for sentence in french_text]

indices = list(range(len(eng_data)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.1,
    random_state=42,
    shuffle=True,
)

eng_data_train = [eng_data[i] for i in train_idx]
eng_data_val = [eng_data[i] for i in val_idx]

fr_data_train = [fr_data[i] for i in train_idx]
fr_data_val = [fr_data[i] for i in val_idx]

def get_batch_and_padding(split='train'):
    if split == "train":
        source_data, target_data = eng_data_train, fr_data_train
    elif split == "val":
        source_data, target_data = eng_data_val, fr_data_val
    else:
        raise ValueError("split must be either 'train' or 'val'")

    if not source_data:
        raise ValueError(f"The {split} dataset split is empty.")

        # Sample paired rows with replacement, matching the original dataloader.
    indices = torch.randint(len(source_data), (batch_size,)).tolist()
    src_max_len = min(max(len(source_data[i]) for i in indices), context_length)
    tgt_max_len = min(max(len(target_data[i]) for i in indices), context_length)

    def pad(ids, max_len, pad_id, eos_id):
        # If truncated, retain EOS as the final token.
        if len(ids) > max_len:
            ids = ids[:max_len]
            if max_len > 0:
                ids[-1] = eos_id
        return ids + [pad_id] * (max_len - len(ids))

    src_batch = torch.tensor(
        [
            pad(source_data[i], src_max_len, eng_tokenizer.pad, eng_tokenizer.eos)
            for i in indices
        ],
        dtype=torch.long,
    )
    tgt_batch = torch.tensor(
        [
            pad(target_data[i], tgt_max_len, fr_tokenizer.pad, fr_tokenizer.eos)
            for i in indices
        ],
        dtype=torch.long,
    )

    src_mask = src_batch != eng_tokenizer.pad
    tgt_mask = tgt_batch != fr_tokenizer.pad
    return src_batch, tgt_batch, src_mask, tgt_mask
