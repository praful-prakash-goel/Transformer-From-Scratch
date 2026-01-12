from models.transformer import build_model
from data.dataloader import get_batch_and_padding, vocab_size_eng, vocab_size_fr, eng_tokenizer, fr_tokenizer
import torch
import os
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'saved_models/best_checkpoint.pt'
lr = 1.5e-4
max_iters = 15_000
warmup_steps = 1_000
eval_iters = 200
eval_interval = 500

@torch.no_grad()
def estimate_loss(model):
    outputs = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            src_ids, tgt_ids, src_mask, tgt_mask = get_batch_and_padding(split)
            
            # move everything to the same device as model
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            # tgt inputs is everything except the last idx
            tgt_inputs = tgt_ids[:, :-1]
            # tgt label is input shifted by 1
            tgt_labels = tgt_ids[:, 1:]
            tgt_label_mask = tgt_mask[:, 1:]
            
            _, loss = model(src_ids, tgt_inputs, src_mask, tgt_label_mask, tgt_labels)
            losses[iter] = loss.item()
        outputs[split] = losses.mean()
    model.train()
    return outputs

def get_lr(step, base_lr, warmup_steps, total_steps):
    """Learning rate warmup"""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
def train_model():
    # build model
    model = build_model(device=device)
    print(f">> {sum(p.numel() for p in model.parameters())/1e6}M Parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # load a pretrained model if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        print(f"Restored model from checkpoint with val loss: {checkpoint['val_loss']}")
    else:
        best_val_loss = float('inf')

    # training loop
    for iter in range(max_iters):
        # evaluate model after an interval
        if iter % eval_interval == 0 or iter == max_iters-1:
            losses = estimate_loss(model)
            train_loss, val_loss = losses['train'], losses['val']
            print(f">> Step {iter} - train_loss: {train_loss}, val_loss: {val_loss}")
            
            # if current loss is less than min loss, then save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at step {iter} - train_loss : {train_loss}, val_loss : {val_loss}")
        
        # sample a batch
        src_ids, tgt_ids, src_mask, tgt_mask = get_batch_and_padding('train')
        
        # move everything to the same device as model
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        # tgt inputs is everything except the last idx
        tgt_inputs = tgt_ids[:, :-1]
        tgt_ip_mask = tgt_mask[:, :-1]
        # tgt label is input shifted by 1
        tgt_labels = tgt_ids[:, 1:]
        
        _, loss = model(src_ids, tgt_inputs, src_mask, tgt_ip_mask, tgt_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # update lr
        lr_now = get_lr(step=iter, base_lr=lr, warmup_steps=warmup_steps, total_steps=max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        optimizer.step()

    print("\n----- Training Complete -----\n")
    
if __name__ == '__main__':
    train_model()