from mlx_lm import load
from reasoning_decoder import reason, GenConfig, FINAL_CLOSE_TOKEN
import sys

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # or "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "Add 27 + 35."
    print("Loading model:", MODEL_ID)
    model, tokenizer = load(MODEL_ID)
    print("Running reasoning...")
    ans = reason(
        model,
        tokenizer,
        question,
        gen_cfg=GenConfig(max_new_tokens=256, temperature=0.7, top_p=0.95, stop=(FINAL_CLOSE_TOKEN,)),
    )
    print("\nFINAL:", ans)

if __name__ == "__main__":
    main()
