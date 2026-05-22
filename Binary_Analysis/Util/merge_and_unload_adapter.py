"""
Merge a PEFT adapter into a full seq2seq model and save the merged weights.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a full model.")
    parser.add_argument("--base_model_path", required=True, help="Local base-model directory.")
    parser.add_argument("--adapter_path", required=True, help="Adapter directory to merge.")
    parser.add_argument("--save_dir", required=True, help="Output directory for merged weights.")
    parser.add_argument("--device", default="cuda", help="Device string, for example cuda or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    os.makedirs(args.save_dir, exist_ok=True)

    print("1) Loading base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float32,
        device_map={"": args.device},
    )

    print("2) Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

    print("3) Attaching adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    print("4) Merging weights...")
    model = model.merge_and_unload()

    print("5) Saving pytorch_model.bin...")
    model.save_pretrained(args.save_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.save_dir)

    print(f"6) Done, pytorch_model.bin saved to {args.save_dir}")


if __name__ == "__main__":
    main()
