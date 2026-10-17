import os
import random
import argparse

import pandas as pd
import torch
from sacrebleu import BLEU, CHRF

from data.dataloader import (
    fr_tokenizer, eng_tokenizer,
    french_text, eng_text,
    test_idx, context_length
)
from models.transformer import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "saved_models", "checkpoint_best.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}.")

    model = build_model(device=device)
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use the test split
    indices = test_idx
    random.Random(42).shuffle(indices)

    if args.limit > 0:
        indices = indices[:args.limit]

    print(f"Evaluating {len(indices)} test examples...")

    hypothesis = []
    references = []
    results = []

    for count, i in enumerate(indices, start=1):
        source = str(eng_text[i]).strip()
        reference = str(french_text[i]).strip()

        if not source or not reference:
            continue

        src_ids = eng_tokenizer.encode(source)
        if len(src_ids) > context_length:
            src_ids = src_ids[:context_length]
            src_ids[-1] = eng_tokenizer.eos
        src_ids = torch.tensor(
            [src_ids], dtype=torch.long, device=device
        )
        src_mask = src_ids != eng_tokenizer.pad

        decoded_start = torch.tensor(
            [[fr_tokenizer.bos]], dtype=torch.long, device=device
        )

        with torch.no_grad():
            output_ids = model.generate_with_cache(
                src_ids=src_ids,
                idx=decoded_start,
                max_new_tokens=args.max_new_tokens,
                src_mask=src_mask,
                eos_token=fr_tokenizer.eos
            )

        prediction = fr_tokenizer.decode(output_ids[0].tolist()).strip()

        hypothesis.append(prediction)
        references.append(reference)

        source_words = len(source.split())
        if source_words <= 10:
            length_group = "short (<=10 words)"
        elif source_words <= 20:
            length_group = "medium (11-20 words)"
        else:
            length_group = "long (>20 words)"

        results.append({
            "English": source,
            "French reference": reference,
            "French prediction": prediction,
            "Source words": source_words,
            "Length group": length_group
        })

        if count <= 10:
            print(f"\n[{count}] English: {source}")
            print("Reference:", reference)
            print("Prediction:", prediction)

        if count % 50 == 0:
            print(f"Processed {count}/{len(indices)}")

    if not hypothesis:
        print("No examples were evaluated.")
        return

    bleu = BLEU()
    chrf = CHRF()

    overall_bleu = bleu.corpus_score(hypothesis, [references]).score
    overall_chrf = chrf.corpus_score(hypothesis, [references]).score
    print("\n===== OVERALL RESULTS =====")
    print(f"Examples: {len(hypothesis)}")
    print(f"BLEU: {overall_bleu:.2f}")
    print(f"CHRF: {overall_chrf:.2f}")

    score_results = {
        "examples": len(hypothesis),
        "overall_bleu": overall_bleu,
        "overall_chrf": overall_chrf,
    }

    print("\n===== RESULTS BY SOURCE LENGTH =====")
    df = pd.DataFrame(results)

    for group in [
        "short (<=10 words)",
        "medium (11-20 words)",
        "long (>20 words)",
    ]:
        subset = df[df["Length group"] == group]

        if len(subset) == 0:
            continue

        preds = subset["French prediction"].tolist()
        refs = subset["French reference"].tolist()

        group_bleu = bleu.corpus_score(preds, [refs]).score
        group_chrf = chrf.corpus_score(preds, [refs]).score
        print(f"\n{group}: {len(subset)} examples")
        print(f"BLEU: {group_bleu:2f}")
        print(f"CHRF: {group_chrf:2f}")

        score_results.update({
            f"group_{group}_bleu": group_bleu,
            f"group_{group}_chrf": group_chrf,
        })

    output_path = os.path.join(RESULTS_DIR, "saved_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved results to {output_path}")

    score_output_path = os.path.join(RESULTS_DIR, "score_results.csv")
    score_df = pd.DataFrame([score_results])
    score_df.to_csv(score_output_path, index=False, encoding="utf-8")
    print(f"Saved results to {score_output_path}")

if __name__ == '__main__':
    main()