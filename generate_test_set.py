#!/usr/bin/env python3
"""
generate_test_set.py
--------------------
Generates 10 diverse instruction-following prompts + gold answers using
DeepSeek-v4-flash via the OpenAI SDK.

Usage:
    export DEEPSEEK_API_KEY='sk-f4b003381bec44ce9f86192362b875e0'
    python generate_test_set.py

Output:
    test_set.json  – the evaluation benchmark for the entire pipeline
"""

import json
import os
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("Set the DEEPSEEK_API_KEY environment variable")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
N_PROMPTS = 10
OUT_FILE = "test_set.json"

# ---------------------------------------------------------------------------
# DEEPSEEK CLIENT
# ---------------------------------------------------------------------------
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)


def chat(messages, temperature=0.7, reasoning_effort="high"):
    """Simple helper to call DeepSeek Chat API.

    Uses reasoning_effort to control the depth of thinking.
    No max_tokens limit is set so the model can generate as much as needed.
    """
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/3): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("DeepSeek API failed after 3 retries")


# ---------------------------------------------------------------------------
# 1. GENERATE PROMPTS
# ---------------------------------------------------------------------------
def generate_prompts():
    system_msg = (
        "You are a helpful assistant that generates diverse evaluation prompts for a "
        "small language model. Produce exactly 10 distinct instruction-following prompts. "
        "Cover the categories: coding, science, math, history, creative writing, general knowledge. "
        "Each prompt should be specific and require a detailed, factual answer. "
        "Output ONLY a JSON array of strings, nothing else."
    )

    print("[1/3] Generating 10 prompts via DeepSeek-v4-flash ...")
    raw = chat([{"role": "system", "content": system_msg},
                  {"role": "user", "content": "Generate the 10 prompts now."}],
                 temperature=0.9)

    # Try to parse JSON array from response
    try:
        prompts = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract lines that look like prompts
        lines = [l.strip().strip('"').strip("'") for l in raw.splitlines() if l.strip()]
        prompts = [l for l in lines if len(l) > 10][:N_PROMPTS]

    if len(prompts) < N_PROMPTS:
        raise ValueError(f"Only got {len(prompts)} prompts, expected {N_PROMPTS}")

    prompts = prompts[:N_PROMPTS]
    print(f"      ✓ Got {len(prompts)} prompts")
    return prompts


# ---------------------------------------------------------------------------
# 2. GENERATE GOLD ANSWERS
# ---------------------------------------------------------------------------
def generate_answers(prompts):
    system_msg = (
        "You are an expert assistant. Answer the user's question clearly, accurately, "
        "and concisely in 3-6 sentences. Be factual and helpful."
    )

    print("[2/3] Generating gold answers via DeepSeek-v4-flash ...")
    test_set = []
    for i, prompt in enumerate(prompts, 1):
        print(f"      Prompt {i}/{len(prompts)}: {prompt[:60]}...")
        answer = chat([{"role": "system", "content": system_msg},
                         {"role": "user", "content": prompt}],
                        temperature=0.3)
        test_set.append({
            "id": i,
            "prompt": prompt,
            "gold_answer": answer,
        })
        time.sleep(0.5)  # polite rate-limiting

    print(f"      ✓ Generated {len(test_set)} gold answers")
    return test_set


# ---------------------------------------------------------------------------
# 3. SAVE
# ---------------------------------------------------------------------------
def save_test_set(test_set):
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
    print(f"[3/3] Saved test set to: {OUT_FILE}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Generating Evaluation Test Set")
    print("  Model:", MODEL)
    print("  Output:", OUT_FILE)
    print("=" * 60)

    prompts = generate_prompts()
    test_set = generate_answers(prompts)
    save_test_set(test_set)

    print("\n  Sample entry:")
    print(json.dumps(test_set[0], indent=2))
    print("\n  Done! You can now run:  python modal_train.py")


if __name__ == "__main__":
    main()
