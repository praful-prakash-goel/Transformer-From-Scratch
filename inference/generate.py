import torch

def translate(
    model,
    eng_ip,
    eng_tokenizer,
    fr_tokenizer,
    device,
    max_new_tokens=200
):
    model.eval()
    
    eng_ip = eng_ip.strip()
    if not eng_ip.endswith('.'):
        eng_ip += '.'
        
    # processing input sentence to feed into encoder
    src_ids = eng_tokenizer.encode(eng_ip)
    src_ids = torch.tensor(
        src_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)
    src_mask = src_ids != eng_tokenizer.pad
    # french decoder's start token
    idx = torch.tensor(
        [[fr_tokenizer.bos]],
        dtype=torch.long
    ).to(device)
    
    translated_op = model.generate_with_cache(
        src_ids=src_ids,
        idx=idx,
        max_new_tokens=max_new_tokens,
        src_mask=src_mask,
        eos_token=fr_tokenizer.eos
    )
    fr_op = fr_tokenizer.decode(translated_op[0].tolist())
    
    return fr_op