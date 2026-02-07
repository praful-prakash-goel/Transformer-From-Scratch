import torch
import time
from data.dataloader import eng_tokenizer, fr_tokenizer
from models.transformer import build_model
from train import checkpoint_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Benchmarking on: {DEVICE.upper()}\n")

def make_long_input(tokenizer, target_len=512):
    base = (
        "Neural machine translation models learn from parallel corpora "
        "containing aligned sentences across languages. "
    )
    ids = tokenizer.encode(base)
    out = ids
    while len(out) < target_len:
        out += ids
    return tokenizer.decode(out[:target_len])

# Defining english inputs for benchmarking
ENGLISH_INPUTS = [
    make_long_input(eng_tokenizer, 512),
    make_long_input(eng_tokenizer, 512),
    make_long_input(eng_tokenizer, 512),
]

def load_model(device=DEVICE):
    model = build_model(device=device)
    checkpoint = torch.load(checkpoint_path, mmap=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def calculate_tps(generate_func, method_name, device=DEVICE, english_inputs=ENGLISH_INPUTS):
    timings = []
    tokens_generated = []
    # french decoder's start token
    idx = torch.tensor(
        [[fr_tokenizer.bos]],
        dtype=torch.long
    ).to(device)
    
    # Warmup the model
    for _ in range(2):
        # Encode the input and create mask
        src_ids = eng_tokenizer.encode("This is a warmup sentence.")
        src_ids = torch.tensor(
            src_ids,
            dtype=torch.long
        ).unsqueeze(0).to(device)
        src_mask = src_ids != eng_tokenizer.pad
        
        _ = generate_func(src_ids, idx, max_new_tokens=512, src_mask=src_mask)
    
    # Actual test
    print(f"\n>> Running benchmark for {method_name}")
    for sent in english_inputs:
        # Encode the input and create mask
        src_ids = eng_tokenizer.encode(sent)
        src_ids = torch.tensor(
            src_ids,
            dtype=torch.long
        ).unsqueeze(0).to(device)
        src_mask = src_ids != eng_tokenizer.pad
        
        # Start timer
        if device == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        
        output_ids = generate_func(src_ids, idx, max_new_tokens=512, src_mask=src_mask)
        
        # End timer
        if device == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate time taken and number of tokens generated
        duration = end_time - start_time
        total_len = output_ids.shape[1]
        input_len = idx.shape[1]
        generated_tokens = total_len - input_len
        
        timings.append(duration)
        tokens_generated.append(generated_tokens)
    
    # Calculate avg tps
    total_time = sum(timings)
    total_tokens = sum(tokens_generated)
    avg_tps = total_tokens / total_time
    
    return avg_tps
    
if __name__ == '__main__':
    print("----- Running the benchmarks -----\n")
    
    # Load the model
    print(f">> Loading the model")
    model = load_model()
    
    if model:
        # Calculate avg tps without kv cache
        tps_without_cache = calculate_tps(generate_func=model.generate, method_name="No KV Cache")
        # Calculate avg tps with kv cache
        tps_with_cache = calculate_tps(generate_func=model.generate_with_cache, method_name="KV Cache")
        
        print(f"\n>> Average tokens per second without kv cache: {tps_without_cache}")
        print(f">> Average tokens per second with kv cache: {tps_with_cache}")
    else:
        print(f">> An error occured while loading model.")