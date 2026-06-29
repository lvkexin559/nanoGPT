"""
eval_cr.py — Evaluate a trained nanoGPT on the chickens-and-rabbits task.

What this script does:
  1. Loads a trained checkpoint + the matching char-level tokenizer
  2. Synthesizes ground-truth (H, F, c, r) test samples with a fixed seed
  3. Asks the model to complete `H={H} F={F}\\n` and reads back the answer
  4. Reports exact-match accuracy + digit-level accuracy + parse-failure rate
     + per-step accuracy (for fmt_B CoT we get 2H/D/r/c step-by-step)
  5. Auto-detects fmt_A vs fmt_B from meta.pkl — no manual --format flag

Why this lives as a script (instead of inline `python -c`):
  - Inline commands keep getting mangled by shell escapes (S5.a hit this)
  - Eval is format-sensitive: prompt template + parser must follow the data fmt
  - This is the §5 / S4 deliverable from notes/07_chickens_rabbits.md
  - Becomes diff-able / commit-able / re-runnable across all S5 experiments

Usage:
  python eval_cr.py --ckpt out-cr/ckpt.pt
  python eval_cr.py --ckpt out-cr-cot/ckpt.pt --n 500 --show-samples 5
  python eval_cr.py --ckpt out-cr-cot/ckpt.pt --split iid
"""
import argparse
import os
import pickle
import random
import re
import sys

import torch

# nanoGPT is a single-file repo, not a package, so we extend sys.path so
# `from model import ...` works regardless of where eval_cr.py is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT  # noqa: E402


# ---------------------------------------------------------------------------
# Section 1 · Format-aware logic
#
# fmt_A and fmt_B use the prompt template `H={H} F={F}\n` (raw numbers).
# fmt_C reverses + zero-pads every number (S5.c trick), so its prompt is
# `H={rev(H)} F={rev(F)}\n`. The model decides whether to continue with
# `c=...` (A/C) or `2H=...` (B). Per-format differences live in build_prompt
# and the parsers.
# ---------------------------------------------------------------------------

_REV_WIDTH = 3  # must match prepare.py's REV_WIDTH for fmt_C


def _rev_pad(n: int, width: int = _REV_WIDTH) -> str:
    """8 -> '008' -> '800'. Mirrors prepare.py.rev_pad."""
    return str(n).zfill(width)[::-1]


def _unrev_int(s: str) -> int:
    """Decode a reversed-and-padded digit string back to its integer value.
    '500' -> reverse -> '005' -> int -> 5. Tolerates any length: a 1-digit
    output like '5' is treated as the number 5 (no padding to undo)."""
    return int(s[::-1]) if s else 0


def build_prompt(H: int, F: int, fmt: str = "A") -> str:
    """Format-aware prompt builder. fmt_C requires reversed-padded H and F
    since that's how the training data was tokenized."""
    if fmt == "C":
        return f"H={_rev_pad(H)} F={_rev_pad(F)}\n"
    return f"H={H} F={F}\n"


def parse_fmt_A(first_line: str):
    """fmt_A answer shape: 'c=X r=Y'
    Returns {'c': int, 'r': int} on success, {} on parse fail.
    Defensive: accept negative numbers via `-?` since OOD models may hallucinate."""
    m_c = re.search(r"c=(-?\d+)", first_line)
    m_r = re.search(r"r=(-?\d+)", first_line)
    if m_c is None or m_r is None:
        return {}
    return {"c": int(m_c.group(1)), "r": int(m_r.group(1))}


def parse_fmt_B(first_line: str):
    """fmt_B CoT answer shape: '2H=A D=B r=X c=Y'
    Returns dict with any of {two_h, D, r, c} that were found. Requires AT LEAST
    r and c to be parsed (otherwise the final answer is missing → parse fail).
    Per-step parsing enables per-step accuracy reporting (Q2.3 evidence)."""
    out = {}
    for key, pattern in [
        ("two_h", r"2H=(-?\d+)"),
        ("D",     r"D=(-?\d+)"),
        ("r",     r"r=(-?\d+)"),
        ("c",     r"c=(-?\d+)"),
    ]:
        m = re.search(pattern, first_line)
        if m is not None:
            out[key] = int(m.group(1))
    # Require at least the final answer (r and c) to count as a successful parse
    if "r" not in out or "c" not in out:
        return {}
    return out


def parse_fmt_C(first_line: str):
    """fmt_C reversed-direct answer shape: 'c=PPP r=QQQ' where PPP and QQQ are
    zero-padded reversed digits (e.g. 'c=300 r=500' means c=3, r=5).
    Returns {'c': int, 'r': int} with values decoded back to original integers.
    No negative-sign support: padded reversed never produces a minus."""
    m_c = re.search(r"c=(\d+)", first_line)
    m_r = re.search(r"r=(\d+)", first_line)
    if m_c is None or m_r is None:
        return {}
    return {"c": _unrev_int(m_c.group(1)), "r": _unrev_int(m_r.group(1))}


PARSERS = {"A": parse_fmt_A, "B": parse_fmt_B, "C": parse_fmt_C}


# ---------------------------------------------------------------------------
# Section 2 · Sanity check the parsers (Mandatory Verification §9)
#
# Throw a mix of correct/malformed/edge inputs at each parser, assert behavior.
# Catches regressions silently in em statistics — esp. "regex too greedy" type.
# ---------------------------------------------------------------------------

PARSER_TESTS = [
    # (input, fmt, should_parse, expected_subset)
    ("c=3 r=5",                 "A", True,  {"c": 3, "r": 5}),
    ("c=10 r=4",                "A", True,  {"c": 10, "r": 4}),
    ("c=-2 r=5",                "A", True,  {"c": -2, "r": 5}),    # hallucinated negative
    ("c= r=5",                  "A", False, {}),                    # missing c value
    ("",                        "A", False, {}),                    # empty
    ("garbage no c or r",       "A", False, {}),
    ("2H=16 D=6 r=3 c=5",       "B", True,  {"two_h": 16, "D": 6, "r": 3, "c": 5}),
    ("2H=6 D=34 r=17 c=",       "B", False, {}),                    # S5.a OOD failure case
    ("r=10 c=2",                "B", True,  {"r": 10, "c": 2}),     # partial CoT (no 2H/D)
    ("2H=24 r=10",              "B", False, {}),                    # missing c → parse fail
    # fmt_C: reversed-padded direct answer. '300' reverses to '003' = 3
    ("c=300 r=500",             "C", True,  {"c": 3, "r": 5}),
    ("c=610 r=400",             "C", True,  {"c": 16, "r": 4}),     # two-digit values
    ("c=001 r=050",             "C", True,  {"c": 100, "r": 50}),   # three-digit (OOD shape)
    ("c=300 r=",                "C", False, {}),                    # truncated mid-emit
    ("",                        "C", False, {}),
]


def sanity_check_parsers() -> None:
    print("[sanity] running parser robustness check...")
    for inp, fmt, should_parse, expected in PARSER_TESTS:
        result = PARSERS[fmt](inp)
        parsed = len(result) > 0
        if parsed != should_parse:
            raise AssertionError(
                f"parser fmt_{fmt}({inp!r}): expected parse={should_parse}, got {result}"
            )
        for k, v in expected.items():
            if result.get(k) != v:
                raise AssertionError(
                    f"parser fmt_{fmt}({inp!r}): key {k} expected {v}, got {result.get(k)}"
                )
    print(f"[sanity] {len(PARSER_TESTS)} parser test cases OK")


# ---------------------------------------------------------------------------
# Section 3 · Loading model + tokenizer
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str):
    """Reconstruct the trained model from a checkpoint.
    Strips _orig_mod. prefix if torch.compile was used during training."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, ckpt


def load_tokenizer(meta_path: str):
    """Returns (encode, decode, meta). `meta` carries format + H ranges used
    during prepare.py so eval_cr.py can default --iid-h-range / --ood-h-range
    without magic numbers in this script."""
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    return encode, decode, meta


# ---------------------------------------------------------------------------
# Section 4 · Sample synthesis (mirrors prepare.py's gen_sample)
# ---------------------------------------------------------------------------

def gen_sample(h_min: int, h_max: int, rng: random.Random):
    """Uniform over (H, c). Returns (H, F, c, r) legal by construction."""
    H = rng.randint(h_min, h_max)
    c = rng.randint(0, H)
    r = H - c
    F = 2 * c + 4 * r
    return H, F, c, r


# ---------------------------------------------------------------------------
# Section 5 · Single-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_one(model, encode, decode, prompt: str, max_new_tokens: int,
                device: str, temperature: float, top_k: int) -> str:
    """Run model.generate once; return the *generated continuation only*."""
    x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(y[0].tolist())[len(prompt):]


# ---------------------------------------------------------------------------
# Section 6 · Batch evaluation on one split
# ---------------------------------------------------------------------------

def eval_split(model, encode, decode, fmt: str, h_min: int, h_max: int,
               n: int, seed: int, max_new_tokens: int, device: str,
               temperature: float, top_k: int, show_samples: int = 0,
               name: str = "split"):
    """Evaluate one split (IID or OOD). Returns a metrics dict.

    Metrics:
      em            — exact-match: c_pred == c_gt AND r_pred == r_gt
      digit_acc     — char-level match between predicted answer and GT answer
      parse_fail    — fraction of outputs where parser couldn't find r or c
      step_acc      — per-CoT-step accuracy (Q2.3-style "where did it break?")
    """
    rng = random.Random(seed)
    parser = PARSERS[fmt]
    em = 0
    parse_fail = 0
    digit_total = 0
    digit_correct = 0

    step_keys = ["two_h", "D", "r", "c"] if fmt == "B" else ["c", "r"]
    step_correct = {k: 0 for k in step_keys}

    samples_shown = 0
    for i in range(n):
        H, F, c_gt, r_gt = gen_sample(h_min, h_max, rng)
        prompt = build_prompt(H, F, fmt)
        gen = predict_one(model, encode, decode, prompt, max_new_tokens,
                          device, temperature, top_k)
        first_line = gen.split("\n")[0]

        if samples_shown < show_samples:
            print(f"  [{name} #{i+1}] GT: H={H} F={F} c={c_gt} r={r_gt}")
            print(f"               model: {first_line!r}")
            samples_shown += 1

        result = parser(first_line)
        if not result:
            parse_fail += 1
            digit_total += 1  # tiny penalty placeholder so the rate isn't 100% by default
            continue

        c_pred, r_pred = result["c"], result["r"]
        if c_pred == c_gt and r_pred == r_gt:
            em += 1

        # per-step accuracy (each step judged independently)
        if fmt == "B":
            if result.get("two_h") == 2 * H:    step_correct["two_h"] += 1
            if result.get("D") == F - 2 * H:    step_correct["D"] += 1
            if result.get("r") == r_gt:         step_correct["r"] += 1
            if result.get("c") == c_gt:         step_correct["c"] += 1
        else:
            if c_pred == c_gt: step_correct["c"] += 1
            if r_pred == r_gt: step_correct["r"] += 1

        # digit accuracy: char-by-char compare predicted answer to GT answer.
        # For fmt_C the GT is the reversed-padded string the model was trained to emit.
        if fmt == "A":
            gt_str = f"c={c_gt} r={r_gt}"
        elif fmt == "C":
            gt_str = f"c={_rev_pad(c_gt)} r={_rev_pad(r_gt)}"
        else:  # fmt == "B"
            gt_str = f"2H={2*H} D={F-2*H} r={r_gt} c={c_gt}"
        for a, b in zip(first_line.ljust(len(gt_str)), gt_str):
            digit_total += 1
            if a == b:
                digit_correct += 1

    em_pct = 100 * em / n
    dig_pct = 100 * digit_correct / max(1, digit_total)
    pf_pct = 100 * parse_fail / n
    step_pct = {k: 100 * v / n for k, v in step_correct.items()}

    print(f"  {name:8} n={n}  em={em_pct:5.1f}%  digit={dig_pct:5.1f}%  parse_fail={pf_pct:4.1f}%")
    step_line = ", ".join(f"{k}={v:.1f}%" for k, v in step_pct.items())
    print(f"           per-step: {step_line}")
    return {
        "em": em_pct, "digit": dig_pct, "parse_fail": pf_pct,
        "step": step_pct, "n": n, "h_range": [h_min, h_max], "fmt": fmt,
    }


# ---------------------------------------------------------------------------
# Section 7 · CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate a trained nanoGPT on the chickens-and-rabbits task."
    )
    p.add_argument("--ckpt", required=True, help="path to ckpt.pt (e.g. out-cr/ckpt.pt)")
    p.add_argument("--data-dir", default="data/chickens_rabbits",
                   help="folder containing meta.pkl (default: data/chickens_rabbits)")
    p.add_argument("--split", choices=["iid", "ood", "both"], default="both")
    p.add_argument("--n", type=int, default=200, help="samples per split (default 200)")
    p.add_argument("--seed", type=int, default=1001,
                   help="rng seed (iid uses seed, ood uses seed+1)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.01,
                   help="near-greedy; 0 not allowed (div by 0)")
    p.add_argument("--top-k", type=int, default=1,
                   help="strict greedy = 1; raise for stochastic sampling")
    p.add_argument("--show-samples", type=int, default=0,
                   help="print first K (prompt, model output) tuples per split")
    p.add_argument("--iid-h-min", type=int, default=None)
    p.add_argument("--iid-h-max", type=int, default=None)
    p.add_argument("--ood-h-min", type=int, default=None)
    p.add_argument("--ood-h-max", type=int, default=None)
    p.add_argument("--no-sanity", action="store_true",
                   help="skip parser sanity check (not recommended)")
    args = p.parse_args()

    if not args.no_sanity:
        sanity_check_parsers()

    meta_path = os.path.join(args.data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise SystemExit(f"meta.pkl not found at {meta_path}")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"ckpt not found at {args.ckpt}")

    print(f"[load] ckpt = {args.ckpt}")
    model, ckpt = load_model(args.ckpt, args.device)
    n_params = sum(pp.numel() for pp in model.parameters())
    iter_num = ckpt.get("iter_num", "?")
    bvl = ckpt.get("best_val_loss", None)
    bvl_str = f"{float(bvl):.4f}" if bvl is not None else "?"
    print(f"[load] model loaded ({n_params/1e6:.2f}M params, iter_num={iter_num}, "
          f"best_val_loss={bvl_str})")

    encode, decode, meta = load_tokenizer(meta_path)
    fmt = meta["format"]
    print(f"[load] tokenizer loaded (vocab_size={meta['vocab_size']}, format={fmt})")

    # Resolve H ranges: CLI override > meta defaults
    iid_h_min = args.iid_h_min if args.iid_h_min is not None else 2
    iid_h_max = args.iid_h_max if args.iid_h_max is not None else meta["h_max_train"]
    ood_h_min = args.ood_h_min if args.ood_h_min is not None else meta["h_min_ood"]
    ood_h_max = args.ood_h_max if args.ood_h_max is not None else meta["h_max_ood"]

    print()
    print(f"=== eval (fmt_{fmt}, n={args.n}/split, temperature={args.temperature}, "
          f"top_k={args.top_k}) ===")

    results = {}
    if args.split in ("iid", "both"):
        print(f"\n[iid]  H in [{iid_h_min}, {iid_h_max}]")
        results["iid"] = eval_split(
            model, encode, decode, fmt, iid_h_min, iid_h_max,
            args.n, args.seed,
            args.max_new_tokens, args.device, args.temperature, args.top_k,
            show_samples=args.show_samples, name="val_iid",
        )
    if args.split in ("ood", "both"):
        print(f"\n[ood]  H in [{ood_h_min}, {ood_h_max}]")
        results["ood"] = eval_split(
            model, encode, decode, fmt, ood_h_min, ood_h_max,
            args.n, args.seed + 1,
            args.max_new_tokens, args.device, args.temperature, args.top_k,
            show_samples=args.show_samples, name="val_ood",
        )

    print()
    print("=== summary ===")
    iid_em = results.get("iid", {}).get("em", float("nan"))
    ood_em = results.get("ood", {}).get("em", float("nan"))
    print(f"  fmt={fmt}  ckpt={args.ckpt}  IID em={iid_em:.1f}%  OOD em={ood_em:.1f}%")


if __name__ == "__main__":
    main()
