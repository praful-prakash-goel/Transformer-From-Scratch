import torch

def translate(model, eng_ip, eng_tokenizer, fr_tokenizer, device):
    model.eval()
    
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
    
    translated_op = model.generate(
        src_ids=src_ids,
        idx=idx,
        max_new_tokens=100,
        src_mask=src_mask,
        temperature=1.0,
        do_sample=False,
        eos_token=fr_tokenizer.eos
    )
    fr_op = fr_tokenizer.decode(translated_op[0].tolist())
    
    return fr_op