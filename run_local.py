#!/usr/bin/env python3
"""Run the reasoning + (optional) web search pipeline using a LOCAL model path.

Usage:
  python run_local.py "Your question here"
  # if model_path.txt exists (written by download_model.py), it'll be used automatically.
  # Otherwise set MODEL_PATH env var:
  MODEL_PATH=./models/qwen25_7b_4bit python run_local.py "2+2?"
"""
import os, sys
from mlx_lm import load
from reasoning_decoder import reason, GenConfig, FINAL_CLOSE_TOKEN

def get_model_path():
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    try:
        with open("model_path.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "Explain binary search in one sentence."
    model_path = get_model_path()
    if not model_path:
        print("No local model path found. Run: python download_model.py")
        sys.exit(1)
    print("Loading local model:", model_path)
    model, tok = load(model_path)
    ans = reason(model, tok, q, gen_cfg=GenConfig(max_new_tokens=256, temperature=0.6, top_p=0.95, stop=(FINAL_CLOSE_TOKEN,)))
    print("\nFINAL:", ans)

if __name__ == "__main__":
    main()
