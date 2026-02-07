import pandas as pd
import torch
import os

context_length = 1024
batch_size = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "eng_french.csv")

df = pd.read_csv(CSV_PATH)
eng_text = df['English words/sentences'].tolist()
french_text = df['French words/sentences'].tolist()

# building vocabulary
eng_chars = set()
for sent in eng_text:
    for ch in sent:
        if ch not in eng_chars:
            eng_chars.add(ch)

fr_chars = set()
for sent in french_text:
    for ch in sent:
        if ch not in fr_chars:
            fr_chars.add(ch)

eng_chars = sorted(list(eng_chars))
fr_chars = sorted(list(fr_chars))
vocab_size_eng = len(eng_chars) + 4
vocab_size_fr = len(fr_chars) + 4 # for special tokens

class CharTokenizer:
    '''A custom character tokenizer'''
    
    def __init__(self, chars, specials=('<pad>', '<unk>', '<bos>', '<eos>')):
        self.specials = list(specials)
        self.chars = self.specials + sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.pad = self.stoi['<pad>']
        self.unk = self.stoi['<unk>']
        self.bos = self.stoi['<bos>']
        self.eos = self.stoi['<eos>']

    def encode(self, s, add_bos=True, add_eos=True):
        ids = [self.stoi.get(ch, self.unk) for ch in s]
        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]
        return ids

    def decode(self, ids, remove_specials=True):
        if remove_specials:
            ids = [i for i in ids if i not in (self.pad, self.bos, self.eos)]
        return ''.join([self.itos[i] for i in ids])
    
eng_tokenizer = CharTokenizer(eng_chars)
fr_tokenizer = CharTokenizer(fr_chars)

eng_data = [eng_tokenizer.encode(sent) for sent in eng_text]
fr_data = [fr_tokenizer.encode(sent) for sent in french_text]

train_len = int(0.9 * len(eng_data))
eng_data_train, eng_data_val = eng_data[:train_len], eng_data[train_len:]
fr_data_train, fr_data_val = fr_data[:train_len], fr_data[train_len:]

def get_batch_and_padding(split='train'):
    eng_data = eng_data_train if split == 'train' else eng_data_val
    fr_data = fr_data_train if split == 'train' else fr_data_val

    def pad(ids, max_len, pad_id):
        ids = ids[:max_len]
        return ids + [pad_id] * (max_len - len(ids))

    # sample random idxs for batching
    idxs = torch.randint(len(eng_data), (batch_size,))
    src_max_len = min(max(len(eng_data[i]) for i in idxs), context_length)
    tgt_max_len = min(max(len(fr_data[i]) for i in idxs), context_length)
    
    # apply padding and create batch
    src_batch = torch.stack([
        torch.tensor(pad(eng_data[i], src_max_len, eng_tokenizer.pad), dtype=torch.long)
        for i in idxs
    ])
    tgt_batch = torch.stack([
        torch.tensor(pad(fr_data[i], tgt_max_len, fr_tokenizer.pad), dtype=torch.long)
        for i in idxs
    ])
    
    # create padding mask
    src_mask = src_batch != eng_tokenizer.pad
    tgt_mask = tgt_batch != fr_tokenizer.pad

    return src_batch, tgt_batch, src_mask, tgt_mask