from model import Transformer
from dataset import get_batch_and_padding, vocab_size_eng, vocab_size_fr, eng_tokenizer, fr_tokenizer
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'best_checkpoint.pt'
lr = 3e-4
max_iters = 5_000
eval_iters = 200
eval_interval = 500

model = Transformer(src_vocab_size=vocab_size_eng, tgt_vocab_size=vocab_size_fr)
model.to(device=device)
print(f">> {sum(p.numel() for p in model.parameters())/1e6}M Parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
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

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['val_loss']
    print(f"Restored model from checkpoint with val loss: {checkpoint['val_loss']}")
    
else:
    best_val_loss = float('inf')

for iter in range(max_iters):
    # evaluate model after an interval
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        train_loss, val_loss = losses['train'], losses['val']
        print(f">> Step {iter} - train_loss: {train_loss}, val_loss: {val_loss}")
        
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
    
    logits, loss = model(src_ids, tgt_inputs, src_mask, tgt_ip_mask, tgt_labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n----- Training Complete -----\n")

print("Testing the model")
model.eval()

# test case 1
# english input
sentence = "i am a student"
src_ids = eng_tokenizer.encode(sentence)
src_ids = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
src_mask = src_ids != eng_tokenizer.pad
# french decoder's start token
idx = torch.tensor([[fr_tokenizer.bos]], dtype=torch.long).to(device)

generated = model.generate(
    src_ids=src_ids,
    idx=idx,
    max_new_tokens=100,
    src_mask=src_mask,
    temperature=1.0,
    do_sample=False,
    eos_token=fr_tokenizer.eos
)
translation = fr_tokenizer.decode(generated[0].tolist())

print(f">> English: {sentence}")
print(f">> French: {translation}")

# test case 2
sentence = "I am going home"
src_ids = eng_tokenizer.encode(sentence)
src_ids = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
src_mask = src_ids != eng_tokenizer.pad
idx = torch.tensor([[fr_tokenizer.bos]], dtype=torch.long).to(device)

generated = model.generate(
    src_ids=src_ids,
    idx=idx,
    max_new_tokens=100,
    src_mask=src_mask,
    temperature=1.0,
    do_sample=False,
    eos_token=fr_tokenizer.eos
)
translation = fr_tokenizer.decode(generated[0].tolist())

print(f">> English: {sentence}")
print(f">> French: {translation}")