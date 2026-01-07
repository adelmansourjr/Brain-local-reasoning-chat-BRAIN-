#!/usr/bin/env python3
"""
Download an MLX model (default: Qwen2.5-7B-Instruct-4bit), save locally, and sanity-test with MLX.

Examples:
  python download_model.py
  python download_model.py --model mlx-community/Llama-3.1-8B-Instruct-4bit --out ./models/llama3_8b_4bit
  HF_HUB_ENABLE_HF_TRANSFER=1 python download_model.py   # faster downloads if 'hf_transfer' is installed
"""
import os
import argparse
from huggingface_hub import snapshot_download

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL_ID", DEFAULT_MODEL),
                   help="HF repo id (MLX format, e.g. mlx-community/Qwen2.5-7B-Instruct-4bit)")
    p.add_argument("--out", default=os.environ.get("MODEL_DIR", "./models/qwen25_7b_4bit"),
                   help="Local target directory")
    p.add_argument("--no-test", action="store_true", help="Skip MLX load/generation sanity test")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Downloading {args.model} → {args.out}")
    try:
        local_path = snapshot_download(
            repo_id=args.model,
            local_dir=args.out,
            local_dir_use_symlinks=False,  # real files (easier to move across disks)
            resume_download=True,
        )
    except Exception as e:
        print("Download failed:", repr(e))
        print("Tips: 1) `pip install hf_transfer` and set HF_HUB_ENABLE_HF_TRANSFER=1 for speed,"
              " 2) `huggingface-cli login` if the repo is gated, 3) re-run with --resume.")
        raise

    print("Downloaded to:", local_path)

    if args.no_test:
        print("Skipping test load (--no-test). Wrote path to model_path.txt")
        with open("model_path.txt", "w") as f:
            f.write(local_path)
        return

    # Test-load with MLX
    try:
        from mlx_lm import load, generate
    except Exception as e:
        print("MLX not available:", repr(e))
        print("Install with: pip install -U mlx-lm")
    else:
        try:
            print("Loading model with MLX…")
            model, tok = load(local_path)
            out = generate(model, tok, "Say 'ready' in one word.", max_tokens=4, temp=0.0, verbose=False)
            print("Generation OK:", out.strip())
        except Exception as e:
            print("Model downloaded, but test generation failed:", repr(e))
            print("You can still try in Python: from mlx_lm import load; load(local_path)")

    with open("model_path.txt", "w") as f:
        f.write(local_path)
    print("\nSaved path to model_path.txt")

if __name__ == "__main__":
    main()