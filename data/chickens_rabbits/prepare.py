"""
Prepare the chickens-and-rabbits arithmetic dataset for nanoGPT.

Classic Chinese math problem: given heads H and feet F, find chickens c and
rabbits r such that
    c + r  = H
    2c + 4r = F
solution: r = (F - 2H) / 2,  c = H - r;  constraint: 2H <= F <= 4H, F even.

We synthesize samples by drawing (H, c) uniformly and deriving F = 2c + 4r,
which guarantees every sample is legal by construction.

Five prompt formats are supported (set via --format):
  A  direct answer:    "H=8 F=22\nc=3 r=5\n"
  B  chain-of-thought: "H=8 F=22\n2H=16 D=6 r=3 c=5\n"
                       (D = F - 2H = 2r, then c = H - r)
  C  reversed digits:  "H=800 F=220\nc=300 r=500\n"   (S5.c, "Teaching Arithmetic" trick)
                       every number is zero-padded to 3 digits then reversed,
                       so the model emits low-order digit first — same trick
                       human grade-school uses for column addition. Width 3
                       covers H<=50 + F<=200, with no surprises at OOD.
  D  reversed + CoT:   "H=800 F=220\n2H=610 D=600 r=300 c=500\n"   (S5.d)
                       fmt_B's CoT structure with fmt_C's reversed digits on
                       EVERY number (including the intermediate 2H, D). The
                       combination Lee et al. 2023 advocate; tests whether
                       reversed-low-order-emit synergizes with explicit
                       multi-step scaffolding.
  M  multi-task mix:   50%  fmt_D main task (chickens-and-rabbits, H in [2, h_max_train])
                       50%  auxiliary "multiply by 2" task (H in [2, h_max_aux]),
                            format: "H=730\n2H=470\n" (H=37 -> 2H=74, reversed+padded)
                       (S5.g) Aux task's H covers the OOD range of the main
                       task; tests whether learning "double" as an isolated
                       skill can transfer to the multi-step main task at OOD.
                       This is the "change training signal" v4 falsification
                       attempt.

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
    python data/chickens_rabbits/prepare.py --format D --out-dir data/chickens_rabbits_revcot/
    python data/chickens_rabbits/prepare.py --format M --out-dir data/chickens_rabbits_multitask/
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


def fmt_D(H: int, F: int, c: int, r: int) -> str:
    """Reversed-digit + CoT. Every number — including intermediate 2H, D —
    is rev_pad'd, so each step's low-order digit comes out first.
    fmt_B says '2H=16 D=6 r=3 c=5', fmt_D says '2H=610 D=600 r=300 c=500'.
    Tests the «reversal × scratchpad» synergy at the heart of the
    "Teaching Arithmetic" paper."""
    return (f"H={rev_pad(H)} F={rev_pad(F)}\n"
            f"2H={rev_pad(2*H)} D={rev_pad(F - 2*H)} "
            f"r={rev_pad(r)} c={rev_pad(c)}\n")


def fmt_mul2(H: int) -> str:
    """Auxiliary task (used inside fmt_M): "multiply H by 2" in isolation.
    Same reversed+padded encoding as fmt_D, single-step. Example: H=37 ->
    'H=730\\n2H=470\\n' (74 reversed-padded). This exposes the "double"
    subskill to the training signal *outside* the multi-step main task."""
    return f"H={rev_pad(H)}\n2H={rev_pad(2*H)}\n"


FORMATTERS = {"A": fmt_A, "B": fmt_B, "C": fmt_C, "D": fmt_D}


def build_split(n: int, h_min: int, h_max: int, formatter, seed: int) -> str:
    """Generate n samples and return them concatenated into one big string."""
    rng = random.Random(seed)
    chunks = []
    for _ in range(n):
        sample = gen_sample(h_min, h_max, rng)
        chunks.append(formatter(*sample))
    return "".join(chunks)


def build_split_multitask(n: int, h_min_main: int, h_max_main: int,
                          h_min_aux: int, h_max_aux: int,
                          seed: int, main_ratio: float = 0.5) -> str:
    """fmt_M mixed generator: each sample independently rolls a coin —
    with prob `main_ratio` emit a fmt_D chickens-and-rabbits sample
    (H in [h_min_main, h_max_main]); otherwise emit a fmt_mul2 aux
    'multiply by 2' sample (H in [h_min_aux, h_max_aux]).
    RNG is seeded, so shuffle order is deterministic."""
    rng = random.Random(seed)
    chunks = []
    n_main = 0
    n_aux = 0
    for _ in range(n):
        if rng.random() < main_ratio:
            sample = gen_sample(h_min_main, h_max_main, rng)
            chunks.append(fmt_D(*sample))
            n_main += 1
        else:
            H = rng.randint(h_min_aux, h_max_aux)
            chunks.append(fmt_mul2(H))
            n_aux += 1
    print(f"  [fmt_M] main={n_main} ({100*n_main/n:.1f}%)  "
          f"aux_mul2={n_aux} ({100*n_aux/n:.1f}%)")
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

    # 5. fmt_C/D/M rev_pad must be perfectly invertible across our number
    #    range, otherwise the eval parser will silently decode wrong integers
    if format_key in ("C", "D", "M"):
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

    # 6. fmt_M-specific: fmt_mul2 aux output stays in vocab across full aux range
    if format_key == "M":
        for h in [2, 8, 20, 37, 50]:
            aux_text = fmt_mul2(h)
            for ch in aux_text:
                if ch not in stoi:
                    raise AssertionError(
                        f"fmt_mul2({h}) emitted char {ch!r} not in vocab"
                    )

    print("[sanity] tokenizer roundtrip + sample legality + format coverage: OK")


def report_unique_combos(text: str, formatter_key: str) -> None:
    """Eyeball how many *unique* (H,F,c,r) tuples landed in this split, so we
    notice when 100k samples collapse to a tiny set of repeats.
    For fmt_C/D/M we reverse-decode the (H,F) string back to its real value,
    so the count reflects unique *real* combinations, not reversed strings.
    For fmt_M, aux "H=XXX\\n" lines lack " F=" and are silently skipped
    (caught by the ValueError branch below); the count reports only main-task
    (H,F) combos, which is what we care about."""
    seen = set()
    for line in text.split("\n"):
        if not line.startswith("H="):
            continue
        try:
            head, tail = line.split(" F=")
            H_str, F_str = head[2:], tail
            if formatter_key in ("C", "D", "M"):
                H = int(H_str[::-1])
                F = int(F_str[::-1])
            else:
                H = int(H_str)
                F = int(F_str)
            seen.add((H, F))
        except (ValueError, IndexError):
            continue
    print(f"  unique (H,F) main-task combos in this split: {len(seen)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=list(FORMATTERS) + ["M"], default="A",
                        help="prompt format: A=direct, B=CoT, C=reversed, D=rev+CoT, "
                             "M=multi-task mix (fmt_D main + fmt_mul2 aux)")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-val-ood", type=int, default=1_000)
    parser.add_argument("--h-max-train", type=int, default=20,
                        help="upper bound of H for train + val_iid")
    parser.add_argument("--h-min-ood", type=int, default=21,
                        help="lower bound of H for val_ood (must be > h-max-train)")
    parser.add_argument("--h-max-ood", type=int, default=50)
    parser.add_argument("--h-max-aux", type=int, default=50,
                        help="fmt_M only: upper bound of H for the aux 'multiply by 2' "
                             "task. Should cover the main task's OOD range so the model "
                             "sees big-H doubling in the aux stream. Default 50.")
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

    # fmt_M shares fmt_D as the main-task formatter (used for val/val_ood).
    # For sanity_check we probe with fmt_D as well since aux is single-arg.
    formatter = fmt_D if args.format == "M" else FORMATTERS[args.format]
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(out_dir, exist_ok=True)

    sanity_check(args.format, formatter)

    print(f"[gen] format={args.format}  vocab_size={VOCAB_SIZE}  vocab={VOCAB_CHARS!r}")
    if args.format == "M":
        print(f"[gen] fmt_M: main task H in [2, {args.h_max_train}], "
              f"aux mul2 H in [2, {args.h_max_aux}], 50/50 mix in train only")

    splits = {
        "train":   (args.n_train,   2,                  args.h_max_train, args.seed),
        "val":     (args.n_val,     2,                  args.h_max_train, args.seed + 1),
        "val_ood": (args.n_val_ood, args.h_min_ood,     args.h_max_ood,   args.seed + 2),
    }

    meta_sizes = {}
    for name, (n, h_lo, h_hi, seed) in splits.items():
        if args.format == "M" and name == "train":
            # fmt_M train mixes fmt_D main task (H in [h_lo,h_hi]) with
            # fmt_mul2 aux task (H in [2, h_max_aux]). val/val_ood stay
            # main-task-only so we always evaluate main-task performance.
            text = build_split_multitask(n, h_lo, h_hi, 2, args.h_max_aux, seed)
        else:
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

    # OOD safety check: no MAIN-TASK train sample should have H > h_max_train.
    # For fmt_C/D/M the on-disk H is reverse-padded, so we decode back before
    # comparing — otherwise H=8 ("800") would falsely look like H=800 > 20.
    # For fmt_M we only check main-task lines (which have " F=" in them);
    # aux "H=XXX\n" lines are allowed to have any H by design.
    train_arr = np.fromfile(os.path.join(out_dir, "train.bin"), dtype=np.uint16)
    train_text = decode(train_arr.tolist())
    leaked = 0
    for line in train_text.split("\n"):
        if line.startswith("H=") and " F=" in line:
            try:
                H_str = line.split(" F=")[0][2:]
                H = int(H_str[::-1]) if args.format in ("C", "D", "M") else int(H_str)
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
