import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from data.dataloader import vocab_size_eng, vocab_size_fr

n_embd = 256
context_length = 64
dropout = 0.2
n_heads = 8
n_layers = 6

class MultiHeadAttention(nn.Module):
    '''Multiple attention heads in parallel'''
    
    def __init__(self, n_heads, head_size, masked=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.masked = masked
        
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0
        
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0
    
    def forward(self, q_input: torch.Tensor, kv_input: Optional[torch.Tensor] = None, attn_mask: Optional[torch.BoolTensor] = None, use_cache: bool = False):
        """
        q_input: (B, Tq, C) # queries
        kv_input: (B, Tk, C) or None, if None -> self attention 
        attn_mask: optional attn_mask (padding / self mask), shapes: (Tk,), (Tq,Tk), (B,Tk), (B,1,Tk), (B,Tq,Tk)
        use_cache: boolean to denote whether to use kv cache or not
        returns (B, Tq, HS)
        """
        # if no kv_input is provided, its self attention
        if kv_input is None:
            kv_input = q_input
        
        B, Tq, C = q_input.shape
        _, Tk, _ = kv_input.shape
        large_neg = -1e9
        
        # compute q, k and v vectors seperately
        q = self.query(q_input)
        k = self.key(kv_input)
        v = self.value(kv_input)
        # reshape for multi-head attention
        q = q.view(B, Tq, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, Tk, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, Tk, self.n_heads, self.head_size).transpose(1, 2)
        
        if attn_mask is not None:
            # convert the data type to bool
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(torch.bool)
            
            if attn_mask.dim() == 1:
                # shape: (Tk,) -> (1, Tk)
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 2:
                # shape: (Tq, Tk) [Global] or (B, Tk) [Per Batch]
                if attn_mask.size(0) == B and attn_mask.size(1) == Tk:
                    # (B, Tk) -> (B, 1, Tk)
                    attn_mask = attn_mask.unsqueeze(1)
                else:
                    # (Tq, Tk) -> (1, Tq, Tk)
                    attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                # shape: (B, 1, Tk) or (B, Tq, Tk)
                pass
            else:
                raise ValueError("attn_mask must be (Tq, Tk), (B, Tk) or (B, 1, Tk) or (B, Tq, Tk)")
            
        if use_cache:
            # kv cache path
            if self.k_cache is None:
                # initalize cache
                self.k_cache = torch.zeros(
                    (B, self.n_heads, context_length, self.head_size),
                    dtype=q_input.dtype, device=q_input.device
                )
                self.v_cache = torch.zeros_like(self.k_cache)
                self.cache_pos = 0
            
            # if cache is full, then reset
            if self.cache_pos + Tk > context_length:
                self.k_cache.zero_()
                self.v_cache.zero_()
                self.cache_pos = 0
            
            old_k = self.k_cache[:, :, :self.cache_pos]
            old_v = self.v_cache[:, :, :self.cache_pos]
            # use the full cache for attention
            full_k = torch.cat([old_k, k], dim=2)
            full_v = torch.cat([old_v, v], dim=2)
            total_len = self.cache_pos + Tq
            
            weights = (q @ full_k.transpose(-2, -1)) * (self.head_size ** -0.5)
            # apply causal mask if required
            if self.masked:
                if Tq == 1:
                    causal_mask = torch.ones(Tq, total_len, device=q_input.device, dtype=torch.bool)
                else:
                    causal_mask = torch.tril(torch.ones(Tq, total_len, device=q_input.device, dtype=torch.bool))
                weights = weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), large_neg)
            
            # padding mask
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    broadcasting_mask = attn_mask.unsqueeze(1)
                else:
                    broadcasting_mask = attn_mask
                weights = weights.masked_fill(~broadcasting_mask, large_neg)
            
            weights = F.softmax(weights, dim=-1)
            weights = self.dropout(weights)
            out = weights @ full_v
            
            # store new K/V in cache
            self.k_cache[:, :, self.cache_pos:self.cache_pos + Tk] = k
            self.v_cache[:, :, self.cache_pos:self.cache_pos + Tk] = v
            self.cache_pos += Tk
        else:
            # standard path
            weights = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
            # apply causal mask if required
            if self.masked:
                causal_mask = torch.tril(torch.ones(Tq, Tk, device=q_input.device, dtype=torch.bool))
                weights = weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), large_neg)
        
            # padding mask
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    broadcasting_mask = attn_mask.unsqueeze(1)
                else:
                    broadcasting_mask = attn_mask
                weights = weights.masked_fill(~broadcasting_mask, large_neg)
                
            weights = F.softmax(weights, dim=-1)
            weights = self.dropout(weights)
            out = weights @ v # shape: (batch, n_heads, time, head_size)
        
        out = out.transpose(1, 2).contiguous().view(B, Tq, C) # shape: (batch, time, channel)
        return self.proj(out)
    
class FeedForwardNetwork(nn.Module):
    '''A simple ffwd network with linear layer followed by non-linear layer'''
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # this ffwd network is used to extract hidden information
        return self.net(x)

class EncoderBlock(nn.Module):
    '''Encoder Block with unmasked self attention'''
    
    def __init__(self, n_heads, n_embd):
        super().__init__()
        head_size = n_embd // n_heads
        # self attention for encoder
        self.self_attn = MultiHeadAttention(n_heads, head_size, masked=False)
        # ffwd network for encoder
        self.ffwd = FeedForwardNetwork()
        # layer norms for self attention and ffwd network
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x, src_key_padding_mask: Optional[torch.BoolTensor] = None):
        """
        x: position and token embedded input
        src_key_padding_mask: Mask for encoder self-attention
        """
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1)
        else:
            attn_mask = None
            
        x = x + self.self_attn(self.ln1(x), attn_mask=attn_mask, use_cache=False)  # residual connection with sa heads and pre-layer-normalization
        x = x + self.ffwd(self.ln2(x))  # residual connection with ffwd network and pre-layer-normalization
        return x

class DecoderBlock(nn.Module):
    '''Decoder Block with masked self attention and cross attention'''
    
    def __init__(self, n_heads, n_embd):
        super().__init__()
        head_size = n_embd // n_heads
        # self attention for decoder
        self.self_attn = MultiHeadAttention(n_heads, head_size, masked=True)
        # cross attention encoder -> decoder
        self.cross_attn = MultiHeadAttention(n_heads, head_size, masked=False)
        # ffwd network for decoder
        self.ffwd = FeedForwardNetwork()
        # layer norms for self attention, cross attention and ffwd network
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
    
    def forward(self, x, encoder_output, tgt_key_padding_mask: Optional[torch.BoolTensor] = None, memory_key_padding_mask: Optional[torch.BoolTensor] = None, use_cache=False):
        """
        x: position and token embedded input
        tgt_key_padding_mask: Mask for decoder self-attention
        memory_key_padding_mask: Mask for cross-attention
        use_cache: boolean to denote whether to use kv cache or not
        """
        if tgt_key_padding_mask is not None:
            tgt_mask = tgt_key_padding_mask.unsqueeze(1)
        else:
            tgt_mask = None
        
        if memory_key_padding_mask is not None:
            memory_mask = memory_key_padding_mask.unsqueeze(1)
        else:
            memory_mask = None
            
        # self attention uses kv caching
        x = x + self.self_attn(self.ln1(x), attn_mask=tgt_mask, use_cache=use_cache) # residual connection with self attention and pre-layer-normalization
        # cross attention does not use kv caching 
        x = x + self.cross_attn(self.ln2(x), kv_input=encoder_output, attn_mask=memory_mask, use_cache=False)  # residual connection with cross attention and pre-layer-normalization
        x = x + self.ffwd(self.ln3(x))  # residual connection with ffwd network and pre-layer-normalization
        return x
        
class Transformer(nn.Module):
    '''Full-fledged transformer with multiple encoder blocks followed by multiple decoder blocks'''
    
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        # each token directly reads the embedding for the next token from the look up table
        self.src_token_embedding_table = nn.Embedding(src_vocab_size, n_embd)
        self.tgt_token_embedding_table = nn.Embedding(tgt_vocab_size, n_embd)
        # position embedding table will capture positional information
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        # multiple encoder blocks
        self.encoder = nn.ModuleList([EncoderBlock(n_heads, n_embd) for _ in range(n_layers)])
        # multiple decoder blocks
        self.decoder = nn.ModuleList([DecoderBlock(n_heads, n_embd) for _ in range(n_layers)])
        # final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        # language model head for getting logits from embeddings
        self.lm_head = nn.Linear(n_embd, tgt_vocab_size)
    
    def forward(self, src_ids, tgt_ids, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None, target=None):
        '''
        src_ids: (B, T_src)
        tgt_ids: (B, T_tgt)
        src_mask: src padding mask
        tgt_mask: tgt padding mask
        '''
        B, T_src = src_ids.shape
        _, T_tgt = tgt_ids.shape
        device = src_ids.device
        
        src_pos = torch.arange(T_src, device=device)
        tgt_pos = torch.arange(T_tgt, device=device)
        # input contains both token embedding as well as position embedding
        src = self.src_token_embedding_table(src_ids) + self.position_embedding_table(src_pos)
        tgt = self.tgt_token_embedding_table(tgt_ids) + self.position_embedding_table(tgt_pos)
        
        enc = src
        for block in self.encoder:
            enc = block(enc, src_key_padding_mask=src_mask)
        # passing encoder outputs as inputs to the decoder cross-attention
        dec = tgt
        for block in self.decoder:
            dec = block(dec, encoder_output=enc, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask, use_cache=False)
        out = self.ln_f(dec)
        logits = self.lm_head(out)
        
        if target is None:
            # if target is not provided, then we cannot compute loss
            loss = None
        else:
            # if target is provided, compute loss
            B, T, C = logits.shape
            labels = target.clone()
            if tgt_mask is not None:
                if tgt_mask.dtype != torch.bool:
                    tgt_mask = tgt_mask.to(device).to(torch.bool)
                labels[~tgt_mask] = -100
            logits = logits.view(B*T, C) # shape: (N, C), where N = No. of samples
            labels = labels.view(B*T,) # shape: (N,)
            loss = F.cross_entropy(logits, labels, ignore_index=-100, label_smoothing=0.1)
        return logits, loss
        
    @torch.no_grad()        
    def generate(
        self,
        src_ids: torch.LongTensor,
        idx: torch.LongTensor,
        max_new_tokens: int,
        src_mask: Optional[torch.BoolTensor] = None,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        eos_token: Optional[int] = None
    ):
        '''
        src_ids: (B, T_src) source token ids
        idx: (B, T_start) initial decoder token ids (prompt)
        max_new_tokens: number of tokens to generate
        src_mask: optional (B, T_src) boolean mask where True = valid (encoder pads)
        temperature: sampling temperature (controls the creativity of the model)
        do_sample: if True sample, else greedy argmax 
        top_k: if sampling and top_k provided, restrict to top_k (controls the diversity of sampling)
        eos_token: if provided, stop per-sequence when eos is generated
        returns: Tensor (B, T_start + gen) with generated token ids
        '''
        device = src_ids.device
        B, T_src = src_ids.shape
        
        # generation is autoregressive: tgt tokens are produced one at a time using all the previously generated tokens
        # while training assumes that the entire tgt sequence is known in advance and processed in parallel
        # for fixed source input, the encoder output (memory) is invariant, so we run the encoder once and reuse
        # its output during decoding to avoid redundant computation
        src_pos = torch.arange(T_src, device=device)
        src_embd = self.src_token_embedding_table(src_ids) + self.position_embedding_table(src_pos)
        enc_output = src_embd
        for block in self.encoder:
            enc_output = block(enc_output, src_key_padding_mask=src_mask)
        
        generated = idx.clone().to(device)
        # it keeps track of when the sequence finishes
        finished = torch.zeros(B, dtype=torch.bool, device=device) if eos_token is not None else None # shape: (B,)
        
        # run decoder autoregressively for each token
        for _ in range(max_new_tokens):
            # crop idx upto context length
            idx_cond = generated[:, -context_length:] if generated.size(1) > context_length else generated
            T_cond = idx_cond.size(1)
            
            # pass the tokens generated till now into the decoder
            tgt_pos = torch.arange(T_cond, device=device)
            tgt_embd = self.tgt_token_embedding_table(idx_cond) + self.position_embedding_table(tgt_pos)
            dec_output = tgt_embd
            for block in self.decoder:
                dec_output = block(dec_output, encoder_output=enc_output, tgt_key_padding_mask=None, memory_key_padding_mask=src_mask)
            out = self.ln_f(dec_output)
            logits = self.lm_head(out)
            # take only the last logits
            last_logits = logits[:, -1, :]
            
            if temperature != 1.0 and temperature > 0.0:
                last_logits = last_logits / temperature
            
            if do_sample:
                probs = F.softmax(last_logits, dim=-1)
                if top_k is not None and top_k > 0:
                    # take the top k probs sorted in descending order
                    topk_vals, topk_idxs = probs.topk(top_k, dim=-1) # shape: (B, k)
                    # take the last prob (min)
                    min_topk = topk_vals[..., -1].unsqueeze(-1) # shape: (B, 1)
                    allowed_mask = probs >= min_topk
                    # allow only those probs which are greater than min
                    probs = probs * allowed_mask.to(probs.dtype)
                    # normalize probs
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # if sampling is not allowed than take the argmax of the logits
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
        
            if eos_token is not None:
                # check to see if a sequence is finished
                just_finished = next_token.unsqueeze(1) == eos_token
                finished = finished | just_finished if finished is not None else just_finished
                if finished.all():
                    break
        
        return generated

    def generate_with_cache(
        self,
        src_ids: torch.LongTensor,
        idx: torch.tensor,
        max_new_tokens: int,
        src_mask: Optional[torch.BoolTensor] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        eos_token: Optional[int] = None
    ):
        '''
        src_ids: (B, T_src) source token ids
        idx: (B, T_start) initial decoder token ids (prompt)
        max_new_tokens: number of tokens to generate
        src_mask: optional (B, T_src) boolean mask where True = valid (encoder pads)
        temperature: sampling temperature (controls the creativity of the model)
        do_sample: if True sample, else greedy argmax 
        top_k: if sampling and top_k provided, restrict to top_k (controls the diversity of sampling)
        eos_token: if provided, stop per-sequence when eos is generated
        returns: Tensor (B, T_start + gen) with generated token ids
        '''
        self.eval()
        # reset cache
        for block in self.decoder:
            block.self_attn.reset_cache()
            
        device = src_ids.device
        B, T_src = src_ids.shape
        
        # process encoder (once)
        src_pos = torch.arange(T_src, device=device)
        src_embd = self.src_token_embedding_table(src_ids) + self.position_embedding_table(src_pos)
        enc_output = src_embd
        for block in self.encoder:
            enc_output = block(enc_output, src_key_padding_mask=src_mask)
        
        # initial decoder fill
        # we assume that idx is the prompt (<sos>)
        generated = idx.clone().to(device)
        finished = torch.zeros(B, dtype=torch.bool, device=device) if eos_token is not None else None
        seq_len = idx.size(1)
        
        tgt_pos = torch.arange(seq_len, device=device)
        tgt_embd = self.tgt_token_embedding_table(idx) + self.position_embedding_table(tgt_pos)
        dec_output = tgt_embd
        for block in self.decoder:
            dec_output = block(dec_output, enc_output, tgt_key_padding_mask=None, memory_key_padding_mask=src_mask, use_cache=True)
        out = self.ln_f(dec_output)
        logits = self.lm_head(out)
        # take only the last logit
        last_logits = logits[:, -1, :]
        
        if temperature != 1.0 and temperature > 0.0:
            last_logits = last_logits / temperature
                
        # sample/argmax logic
        if do_sample:
            probs = F.softmax(last_logits, dim=-1)
            if top_k is not None and top_k > 0:
                topk_vals, topk_idxs = probs.topk(top_k, dim=-1)
                min_topk = topk_vals[..., -1].unsqueeze(-1)
                allowed_mask = probs >= min_topk
                probs = probs * allowed_mask.to(probs.dtype)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
        
        generated = torch.cat([generated, next_token], dim=1)
        idx = torch.cat([idx, next_token], dim=1)
        seq_len += 1
        
        # generation loop
        for _ in range(1, max_new_tokens):
            # chunking logic
            if seq_len >= context_length:
                # recover the last half of the tokens
                recover_len = context_length // 2
                context_window = generated[:, -recover_len:]
                
                # hard reset cache
                for block in self.decoder:
                    block.self_attn.reset_cache()
                    
                # re-fill cache with the window
                # we need to re-calculate the embeddings for the window
                # Note - we reset position ids to 0..recover_len to match the reset cache
                tgt_pos = torch.arange(recover_len, device=device)
                tgt_embd = self.tgt_token_embedding_table(context_window) + self.position_embedding_table(tgt_pos)
                dec_output = tgt_embd
                for block in self.decoder:
                    dec_output = block(dec_output, enc_output, tgt_key_padding_mask=None, memory_key_padding_mask=src_mask, use_cache=True)
                
                idx = context_window
                seq_len = recover_len
                    
            # we are feeding 1 token. Position is relative to current cache size
            last_token = idx[:, -1:]
            start_pos = seq_len - 1
            
            # calculate embedding for just this 1 token
            pos_id = torch.tensor([start_pos], device=device)
            tgt_embd = self.tgt_token_embedding_table(last_token) + self.position_embedding_table(pos_id)
            dec_output = tgt_embd
            for block in self.decoder:
                dec_output = block(dec_output, enc_output, tgt_key_padding_mask=None, memory_key_padding_mask=src_mask, use_cache=True)
            out = self.ln_f(dec_output)
            logits = self.lm_head(out)
            last_logits = logits[:, -1, :]
            
            if temperature != 1.0 and temperature > 0.0:
                last_logits = last_logits / temperature
            
            # sample/argmax logic
            if do_sample:
                probs = F.softmax(last_logits, dim=-1)
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idxs = probs.topk(top_k, dim=-1)
                    min_topk = topk_vals[..., -1].unsqueeze(-1)
                    allowed_mask = probs >= min_topk
                    probs = probs * allowed_mask.to(probs.dtype)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            idx = torch.cat([idx, next_token], dim=1)
            seq_len += 1
            
            # eos check
            if eos_token is not None:
                just_finished = next_token.unsqueeze(1) == eos_token
                finished = finished | just_finished if finished is not None else just_finished
                if finished.all():
                    break
        
        return generated
    
def build_model(device):
    model = Transformer(
        src_vocab_size=vocab_size_eng,
        tgt_vocab_size=vocab_size_fr
    )
    model.to(device=device)
    return model