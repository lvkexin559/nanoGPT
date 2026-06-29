"""
Prepare the chickens-and-rabbits arithmetic dataset for nanoGPT.

Classic Chinese math problem: given heads H and feet F, find chickens c and
rabbits r such that
    c + r  = H
    2c + 4r = F
solution: r = (F - 2H) / 2,  c = H - r;  constraint: 2H <= F <= 4H, F even.

We synthesize samples by drawing (H, c) uniformly and deriving F = 2c + 4r,
which guarantees every sample is legal by construction.

Three prompt formats are supported (set via --format):
  A  direct answer:    "H=8 F=22\nc=3 r=5\n"
  B  chain-of-thought: "H=8 F=22\n2H=16 D=6 r=3 c=5\n"
                       (D = F - 2H = 2r, then c = H - r)
  C  reversed digits:  "H=800 F=220\nc=300 r=500\n"   (S5.c, "Teaching Arithmetic" trick)
                       every number is zero-padded to 3 digits then reversed,
                       so the model emits low-order digit first — same trick
                       human grade-school uses for column addition. Width 3
                       covers H<=50 + F<=200, with no surprises at OOD.

Three splits are produced:
  train.bin       train samples (H in [2, 20], allows duplicates)
  val.bin         in-distribution val (H in [2, 20])
  val_ood.bin     out-of-distribution val (H in [21, 50]) <- the interesting one

meta.pkl carries {vocab_size, stoi, itos, format, h_max_train} so eval_cr.py
later knows how the data was tokenized and what counts as OOD.

Usage:
    python data/chickens_rabbits/prepare.py            # default fmt_A
    python data/chickens_rabbits/prepare.py --format B # CoT
    python data/chickens_rabbits/prepare.py --format C --out-dir data/chickens_rabbits_rev/
"""
import argparse
import os
import pickle
import random

import numpy as np

# vocab is the union of all characters used by fmt_A and fmt_B.
# fmt_A needs:  0-9, H, F, c, r, =, space, newline
# fmt_B also uses 'D' (for D = F - 2H)
# We expose a single static vocab so the same meta.pkl works for both formats.
VOCAB_CHARS = list("0123456789HFcrD= \n")
VOCAB_SIZE = len(VOCAB_CHARS)  # 18

stoi = {ch: i for i, ch in enumerate(VOCAB_CHARS)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s: str) -> list[int]:
    """char-level encode; explicit loop (not list comprehension) so we can
    raise a precise error on the offending char."""
    out = []
    for ch in s:
        if ch not in stoi:
            raise ValueError(f"char {ch!r} not in vocab (size={VOCAB_SIZE})")
        out.append(stoi[ch])
    return out


def decode(ids) -> str:
    return "".join(itos[i] for i in ids)


def gen_sample(h_min: int, h_max: int, rng: random.Random):
    """Draw one legal (H, F, c, r) sample uniformly over (H, c)."""
    H = rng.randint(h_min, h_max)
    c = rng.randint(0, H)
    r = H - c
    F = 2 * c + 4 * r
    return H, F, c, r


def fmt_A(H: int, F: int, c: int, r: int) -> str:
    """Direct answer format."""
    return f"H={H} F={F}\nc={c} r={r}\n"


def fmt_B(H: int, F: int, c: int, r: int) -> str:
    """Chain-of-thought format with the intermediate D = F - 2H step.
    Order is (r, c) because the model can derive r directly from D/2, then c
    from H - r. Putting r before c gives the model a useful causal scaffold."""
    return f"H={H} F={F}\n2H={2*H} D={F - 2*H} r={r} c={c}\n"


REV_WIDTH = 3  # zero-pad width before reversing; covers H<=50, F<=200


def rev_pad(n: int, width: int = REV_WIDTH) -> str:
    """Zero-pad to fixed width then reverse char by char.
    Examples (width=3):
        8   -> '008' -> '800'
        16  -> '016' -> '610'
        100 -> '100' -> '001'
    Fixed width keeps the generated answer a constant number of tokens, which
    is one of the reasons reversal works in practice (« Teaching Arithmetic »)."""
    return str(n).zfill(width)[::-1]


def fmt_C(H: int, F: int, c: int, r: int) -> str:
    """Reversed-digit direct answer. The trick: model emits low-order digit
    first, mirroring how humans do column arithmetic from right to left.
    fmt_A says 'c=3 r=5', fmt_C says 'c=300 r=500' (both encode c=3, r=5)."""
    return f"H={rev_pad(H)} F={rev_pad(F)}\nc={rev_pad(c)} r={rev_pad(r)}\n"


FORMATTERS = {"A": fmt_A, "B": fmt_B, "C": fmt_C}


def build_split(n: int, h_min: int, h_max: int, formatter, seed: int) -> str:
    """Generate n samples and return them concatenated into one big string."""
    rng = random.Random(seed)
    chunks = []
    for _ in range(n):
        sample = gen_sample(h_min, h_max, rng)
        chunks.append(formatter(*sample))
    return "".join(chunks)


def sanity_check(format_key: str, formatter) -> None:
    """Self-tests we want to fail loud rather than silently produce bad data.
    Spec required by workspace rule "Mandatory Verification"."""
    # 1. tokenizer roundtrip on a representative string
    probe = "H=8 F=22\n2H=16 D=6 r=3 c=5\nH=20 F=72\nc=2 r=18\n"
    if decode(encode(probe)) != probe:
        raise AssertionError("tokenizer is not invertible on probe string")

    # 2. tokenizer roundtrip on every single-char vocab member
    for ch in VOCAB_CHARS:
        if decode(encode(ch)) != ch:
            raise AssertionError(f"single-char roundtrip failed for {ch!r}")

    # 3. mathematical consistency: 100 random samples must satisfy the system
    rng = random.Random(0)
    for _ in range(100):
        H, F, c, r = gen_sample(2, 50, rng)
        if not (c + r == H and 2 * c + 4 * r == F and c >= 0 and r >= 0):
            raise AssertionError(f"illegal sample generated: H={H} F={F} c={c} r={r}")

    # 4. formatter output stays within vocab (no surprise chars)
    sample_text = formatter(*gen_sample(2, 50, rng))
    for ch in sample_text:
        if ch not in stoi:
            raise AssertionError(
                f"formatter {format_key!r} emitted char {ch!r} not in vocab"
            )

    # 5. fmt_C-specific: rev_pad must be perfectly invertible across our number
    #    range, otherwise the eval parser will silently decode wrong integers
    if format_key == "C":
        for n in [0, 1, 5, 8, 10, 16, 22, 40, 50, 100, 168, 200]:
            padded_rev = rev_pad(n)
            decoded = int(padded_rev[::-1])
            if decoded != n:
                raise AssertionError(
                    f"rev_pad/unrev not invertible for n={n}: "
                    f"rev_pad={padded_rev!r}, decoded={decoded}"
                )
            if len(padded_rev) != REV_WIDTH:
                raise AssertionError(
                    f"rev_pad width mismatch for n={n}: got {len(padded_rev)}, want {REV_WIDTH}"
                )

    print("[sanity] tokenizer roundtrip + sample legality + format coverage: OK")


def report_unique_combos(text: str, formatter_key: str) -> None:
    """Eyeball how many *unique* (H,F,c,r) tuples landed in this split, so we
    notice when 100k samples collapse to a tiny set of repeats.
    For fmt_C we reverse-decode the (H,F) string back to its real value, so
    the count reflects unique *real* combinations, not reversed strings."""
    seen = set()
    for line in text.split("\n"):
        if not line.startswith("H="):
            continue
        try:
            head, tail = line.split(" F=")
            H_str, F_str = head[2:], tail
            if formatter_key == "C":
                H = int(H_str[::-1])
                F = int(F_str[::-1])
            else:
                H = int(H_str)
                F = int(F_str)
            seen.add((H, F))
        except (ValueError, IndexError):
            continue
    print(f"  unique (H,F) combos in this split: {len(seen)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=list(FORMATTERS), default="A",
                        help="prompt format: A=direct, B=chain-of-thought")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-val-ood", type=int, default=1_000)
    parser.add_argument("--h-max-train", type=int, default=20,
                        help="upper bound of H for train + val_iid")
    parser.add_argument("--h-min-ood", type=int, default=21,
                        help="lower bound of H for val_ood (must be > h-max-train)")
    parser.add_argument("--h-max-ood", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None,
                        help="output dir for {train,val,val_ood}.bin + meta.pkl. "
                             "Default: this file's own directory. Pass a fresh path "
                             "(e.g. data/chickens_rabbits_h40/) to keep prior runs.")
    args = parser.parse_args()

    if args.h_min_ood <= args.h_max_train:
        raise SystemExit(
            f"OOD split must be strictly above train H range: "
            f"h_min_ood={args.h_min_ood} <= h_max_train={args.h_max_train}"
        )

    formatter = FORMATTERS[args.format]
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(out_dir, exist_ok=True)

    sanity_check(args.format, formatter)

    print(f"[gen] format={args.format}  vocab_size={VOCAB_SIZE}  vocab={VOCAB_CHARS!r}")

    splits = {
        "train":   (args.n_train,   2,                  args.h_max_train, args.seed),
        "val":     (args.n_val,     2,                  args.h_max_train, args.seed + 1),
        "val_ood": (args.n_val_ood, args.h_min_ood,     args.h_max_ood,   args.seed + 2),
    }

    meta_sizes = {}
    for name, (n, h_lo, h_hi, seed) in splits.items():
        text = build_split(n, h_lo, h_hi, formatter, seed)
        ids = encode(text)
        arr = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(out_dir, f"{name}.bin")
        arr.tofile(out_path)
        print(f"[gen] {name}: n={n:>6}  H in [{h_lo},{h_hi}]  "
              f"tokens={len(arr):>9,}  -> {os.path.basename(out_path)}")
        report_unique_combos(text, args.format)
        meta_sizes[name] = {"n_samples": n, "n_tokens": int(len(arr)),
                            "h_range": [h_lo, h_hi]}

    # OOD safety check: no train sample should have H > h_max_train.
    # For fmt_C the on-disk string is reverse-padded, so we decode back before
    # comparing — otherwise H=8 ("800") would falsely look like H=800 > 20.
    train_arr = np.fromfile(os.path.join(out_dir, "train.bin"), dtype=np.uint16)
    train_text = decode(train_arr.tolist())
    leaked = 0
    for line in train_text.split("\n"):
        if line.startswith("H="):
            try:
                H_str = line.split(" F=")[0][2:]
                H = int(H_str[::-1]) if args.format == "C" else int(H_str)
                if H > args.h_max_train:
                    leaked += 1
            except (ValueError, IndexError):
                pass
    if leaked > 0:
        raise AssertionError(
            f"OOD leak: train.bin contains {leaked} samples with H > {args.h_max_train}"
        )
    print(f"[sanity] OOD isolation in train.bin: OK (0 leaks above H={args.h_max_train})")

    # Eyeball: print the first 3 decoded samples so a human can verify the
    # formula by hand right after running prepare.py.
    print("[eyeball] first 3 samples decoded from train.bin:")
    head = decode(train_arr[:200].tolist())
    for line in head.split("\n")[:6]:
        print(f"  | {line}")

    meta = {
        "vocab_size": VOCAB_SIZE,
        "stoi": stoi,
        "itos": itos,
        "format": args.format,
        "h_max_train": args.h_max_train,
        "h_min_ood": args.h_min_ood,
        "h_max_ood": args.h_max_ood,
        "splits": meta_sizes,
    }
    meta_path = os.path.join(out_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[meta] saved meta.pkl  vocab_size={VOCAB_SIZE}  format={args.format}")


if __name__ == "__main__":
    main()
