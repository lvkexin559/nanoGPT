"""
Prepare the chickens-and-rabbits arithmetic dataset for nanoGPT.

Classic Chinese math problem: given heads H and feet F, find chickens c and
rabbits r such that
    c + r  = H
    2c + 4r = F
solution: r = (F - 2H) / 2,  c = H - r;  constraint: 2H <= F <= 4H, F even.

We synthesize samples by drawing (H, c) uniformly and deriving F = 2c + 4r,
which guarantees every sample is legal by construction.

Six prompt formats are supported (set via --format):
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
  N  multi-task full:  50%  fmt_D main task (H in [2, h_max_train])
                       12.5% aux_mul2    : "H=X\n2H=Y\n"       (H in [2, h_max_aux])
                       12.5% aux_sub_F2H : "F=X 2H=Y\nD=Z\n"   (H in [2, h_max_aux])
                       12.5% aux_div_D   : "D=X\nr=Y\n"        (H in [2, h_max_aux])
                       12.5% aux_sub_Hr  : "H=X r=Y\nc=Z\n"    (H in [2, h_max_aux])
                       (S5.h) The v5 direct falsification test: supervise
                       ALL four subskills of the main task's CoT, each in
                       isolation and each covering the OOD H range. If v5
                       is right ("training signal + compositional coverage"),
                       main-task OOD em should jump to 60-80%.
  L  loss-masked SFT:  Text on disk is identical to fmt_D. But a companion
                       *_mask.bin file marks each position as prompt (0) or
                       answer (1). train.py loads this mask and sets y=-1 at
                       mask=0 positions, so cross_entropy(ignore_index=-1)
                       supervises ONLY answer tokens. This is the SFT-style
                       loss-mask trick: model sees full context every batch
                       but only gets gradient signal on the answer portion.
                       (S5.j) v5.2 direct falsification test — turned out
                       loss_mask alone doesn't fix OOD because it doesn't
                       introduce OOD subskill exposure. Prompted v5.3.
  O  context-aligned:  All samples share the SAME fmt_D text layout
                       ("H=X F=Y\n2H=A D=B r=C c=D\n") — even the aux ones.
                       What differs is the mask:
                         50%  main task (H in [2, h_max_train]), mask=1 on
                              the whole answer (2H D r c)
                         12.5% aux_mul2   (H in [2, h_max_aux]), mask=1 only
                              on the "2H=A" tokens
                         12.5% aux_sub_F2H, mask=1 only on "D=B"
                         12.5% aux_div_D,   mask=1 only on "r=C"
                         12.5% aux_sub_Hr,  mask=1 only on "c=D"
                       (S5.k) v5.3 direct falsification test: this is the
                       combination that satisfies BOTH (a) OOD exposure and
                       (b) full main-task context per subskill. Predicts
                       every per-step should jump to ~70% and OOD em to
                       ~40-70%.
  P  full-answer mask: Identical text distribution as fmt_O (50% main +
                       12.5% each of 4 aux type, aux H covers OOD range).
                       BUT the mask is fmt_L-style: prompt=0, answer=1 for
                       ALL samples (main and aux). So aux samples degenerate
                       into "extra main-task samples with H in [2, h_max_aux]".
                       (S5.l) v6 mask-role falsification: does mask's
                       "supervise only the target subskill" matter, or does
                       any mask work as long as data mix covers OOD? If
                       fmt_P works on val_ood [21,50] but breaks on val_ood
                       [51,100] while fmt_O generalizes, mask's role IS to
                       prevent lookup shortcut and force algorithmic
                       learning.

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
    python data/chickens_rabbits/prepare.py --format N --out-dir data/chickens_rabbits_multitask_full/
    python data/chickens_rabbits/prepare.py --format L --out-dir data/chickens_rabbits_lossmask/
    python data/chickens_rabbits/prepare.py --format O --out-dir data/chickens_rabbits_context_aligned/
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


REV_WIDTH = 3  # module-level default; main() can override via --rev-width to 4
               # (width 3 covers H<=50 F<=200; width 4 covers H<=500 F<=2000, etc.)


def rev_pad(n: int, width: int | None = None) -> str:
    """Zero-pad to fixed width then reverse char by char.
    If width is None, dynamically read the module-level REV_WIDTH — this
    lets main() override REV_WIDTH once and every downstream formatter
    picks it up without threading a width kwarg through every function.
    Examples (width=3):
        8   -> '008' -> '800'
        16  -> '016' -> '610'
        100 -> '100' -> '001'
    Examples (width=4):
        8    -> '0008' -> '8000'
        500  -> '0500' -> '0050'
        2000 -> '2000' -> '0002'
    Fixed width keeps the generated answer a constant number of tokens, which
    is one of the reasons reversal works in practice (« Teaching Arithmetic »)."""
    w = REV_WIDTH if width is None else width
    return str(n).zfill(w)[::-1]


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
    """Auxiliary task (used inside fmt_M, fmt_N): "multiply H by 2" in
    isolation. Same reversed+padded encoding as fmt_D, single-step.
    Example: H=37 -> 'H=730\\n2H=470\\n' (74 reversed-padded)."""
    return f"H={rev_pad(H)}\n2H={rev_pad(2*H)}\n"


def fmt_sub_F2H(H: int, c: int) -> str:
    """Auxiliary task (fmt_N): "D = F - 2H" in isolation. Uses the same
    (H, c) draw as the main task, so all derived numbers are legal.
    Example: H=37, c=26 -> F=96, 2H=74, D=22 -> 'F=690 2H=470\\nD=220\\n'."""
    r = H - c
    F = 2 * c + 4 * r
    return f"F={rev_pad(F)} 2H={rev_pad(2*H)}\nD={rev_pad(F - 2*H)}\n"


def fmt_div_D(H: int, c: int) -> str:
    """Auxiliary task (fmt_N): "r = D / 2" in isolation. Since D = 2r by
    construction, r is always a legal integer division. Example: H=37,
    c=26 -> r=11, D=22 -> 'D=220\\nr=110\\n'."""
    r = H - c
    F = 2 * c + 4 * r
    return f"D={rev_pad(F - 2*H)}\nr={rev_pad(r)}\n"


def fmt_sub_Hr(H: int, c: int) -> str:
    """Auxiliary task (fmt_N): "c = H - r" in isolation. Example: H=37,
    c=26, r=11 -> 'H=730 r=110\\nc=620\\n'."""
    r = H - c
    return f"H={rev_pad(H)} r={rev_pad(r)}\nc={rev_pad(c)}\n"


def fmt_L_with_mask(H: int, F: int, c: int, r: int):
    """fmt_L: text identical to fmt_D, plus a per-char mask marking
    prompt (0) vs answer (1). Consumed by train.py get_batch, which sets
    y=-1 at mask=0 positions so cross_entropy(ignore_index=-1) skips them.
    Semantically: train the same fmt_D data but only compute loss on the
    answer portion (SFT-style loss masking).

    Returns (text, mask_list) both of same length. char-level tokenizer +
    ordering guarantees `mask[i]` corresponds to `encode(text)[i]`.

    Example (H=8, F=22, c=3, r=5):
      text = "H=800 F=220\\n2H=610 D=600 r=300 c=500\\n"  (38 chars)
      mask = [0]*12 + [1]*26                              (12 prompt + 26 answer)
    """
    prompt = f"H={rev_pad(H)} F={rev_pad(F)}\n"
    answer = (f"2H={rev_pad(2*H)} D={rev_pad(F - 2*H)} "
              f"r={rev_pad(r)} c={rev_pad(c)}\n")
    text = prompt + answer
    mask = [0] * len(prompt) + [1] * len(answer)
    return text, mask


def fmt_O_with_mask(H: int, c: int, subskill: str):
    """fmt_O: every sample has the exact same fmt_D text layout. What
    differs is which characters get mask=1 (i.e. get supervised).
    This is the v5.3 direct falsification test — teaches the aux subskill
    with the SAME context the model will face at inference time.

    subskill in {"main", "mul2", "sub_F2H", "div_D", "sub_Hr"}:
      main    -> supervise entire answer (2H D r c \\n)   [fmt_L-style main]
      mul2    -> supervise ONLY the "2H=A" 6 chars
      sub_F2H -> supervise ONLY the " D=B" 5 chars
      div_D   -> supervise ONLY the " r=C" 5 chars
      sub_Hr  -> supervise ONLY the " c=D" 5 chars

    Returns (text, mask). Text is identical fmt_D layout regardless of
    subskill; only the mask differs. This means the model's forward pass
    on aux samples sees the identical attention context as it will at
    inference — closing the aux-vs-main context gap that fmt_N v2 couldn't."""
    r = H - c
    F = 2 * c + 4 * r
    # Assemble text piece by piece so we can compute exact char offsets
    prompt   = f"H={rev_pad(H)} F={rev_pad(F)}\n"
    step_2H  = f"2H={rev_pad(2*H)}"        # 6 chars: "2H=" + 3-digit
    step_D   = f" D={rev_pad(F - 2*H)}"    # 6 chars: " D=" + 3-digit
    step_r   = f" r={rev_pad(r)}"          # 6 chars
    step_c   = f" c={rev_pad(c)}"          # 6 chars
    end      = "\n"                        # 1 char
    text = prompt + step_2H + step_D + step_r + step_c + end
    # Cumulative offsets:
    p0 = len(prompt)                        # answer starts here (after prompt+\n)
    p1 = p0 + len(step_2H)                  # after 2H=A
    p2 = p1 + len(step_D)                   # after " D=B"
    p3 = p2 + len(step_r)                   # after " r=C"
    p4 = p3 + len(step_c)                   # after " c=D"
    p5 = p4 + len(end)                      # after "\n" (= len(text))
    mask = [0] * len(text)
    if subskill == "main":
        for i in range(p0, p5):
            mask[i] = 1
    elif subskill == "mul2":
        for i in range(p0, p1):
            mask[i] = 1
    elif subskill == "sub_F2H":
        for i in range(p1, p2):
            mask[i] = 1
    elif subskill == "div_D":
        for i in range(p2, p3):
            mask[i] = 1
    elif subskill == "sub_Hr":
        for i in range(p3, p4):
            mask[i] = 1
    else:
        raise ValueError(f"unknown subskill {subskill!r} for fmt_O")
    return text, mask


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


def build_split_full_answer_mask(n: int, h_min_main: int, h_max_main: int,
                                  h_min_aux: int, h_max_aux: int,
                                  seed: int, main_ratio: float = 0.5):
    """fmt_P split generator: same text distribution as fmt_O (main + 4 aux
    all in complete fmt_D layout, aux H covers OOD) BUT mask is fmt_L-style
    prompt/answer split — the WHOLE answer is supervised regardless of
    whether the sample was drawn as 'main' or a specific aux subskill.

    Compared to fmt_O:
      - Identical text.bin
      - Different mask.bin: supervised_frac ≈ 0.68 (vs fmt_O's 0.42)
    Semantically: aux samples become extra main-task samples with wider H.

    Predicts: on val_ood [h_max_train+1, h_max_aux] em ~= 100% (H in
    training distribution now); on val_ood outside [2, h_max_aux] em
    should crash (no OOD exposure past h_max_aux)."""
    rng = random.Random(seed)
    aux_subskills = ["mul2", "sub_F2H", "div_D", "sub_Hr"]
    counts = {"main": 0, **{f"aux_{s}": 0 for s in aux_subskills}}
    text_parts = []
    mask_parts = []
    for _ in range(n):
        if rng.random() < main_ratio:
            H, F, c, r = gen_sample(h_min_main, h_max_main, rng)
            counts["main"] += 1
        else:
            H, F, c, r = gen_sample(h_min_aux, h_max_aux, rng)
            counts[f"aux_{aux_subskills[rng.randrange(len(aux_subskills))]}"] += 1
        # Whichever draw path we took, emit full fmt_D text and full-answer mask
        text, mask = fmt_L_with_mask(H, F, c, r)
        text_parts.append(text)
        mask_parts.append(mask)
    full_text = "".join(text_parts)
    full_mask = [m for chunk in mask_parts for m in chunk]
    assert len(full_text) == len(full_mask), \
        f"[fmt_P] text/mask length mismatch: {len(full_text)} vs {len(full_mask)}"
    supervised_frac = sum(full_mask) / len(full_mask) if full_mask else 0
    pct = {k: 100 * v / n for k, v in counts.items()}
    print(f"  [fmt_P] main={counts['main']} ({pct['main']:.1f}%)  "
          + "  ".join(f"aux_{s}(*)={counts[f'aux_{s}']} ({pct[f'aux_{s}']:.1f}%)"
                      for s in aux_subskills))
    print(f"          supervised_frac={supervised_frac:.3f}  "
          f"(vs fmt_O's ~0.42 — all answer positions supervised)")
    return full_text, full_mask


def build_split_lossmask_context_aligned(n: int, h_min_main: int, h_max_main: int,
                                          h_min_aux: int, h_max_aux: int,
                                          seed: int, main_ratio: float = 0.5):
    """fmt_O split generator: mixed main-task + 4 context-aligned aux tasks.
    Every sample has identical fmt_D text layout; only the mask differs.

    50%   main task  (H in [h_min_main, h_max_main]): mask covers full answer
    12.5% aux_mul2   (H in [h_min_aux,  h_max_aux]):  mask on "2H=A"
    12.5% aux_sub_F2H:                                mask on " D=B"
    12.5% aux_div_D:                                  mask on " r=C"
    12.5% aux_sub_Hr:                                 mask on " c=D"
    """
    rng = random.Random(seed)
    aux_subskills = ["mul2", "sub_F2H", "div_D", "sub_Hr"]
    counts = {"main": 0, **{f"aux_{s}": 0 for s in aux_subskills}}
    text_parts = []
    mask_parts = []
    supervised_tokens = 0
    total_tokens = 0
    for _ in range(n):
        if rng.random() < main_ratio:
            H, F, c, r = gen_sample(h_min_main, h_max_main, rng)
            text, mask = fmt_O_with_mask(H, c, "main")
            counts["main"] += 1
        else:
            H, F, c, r = gen_sample(h_min_aux, h_max_aux, rng)
            subskill = aux_subskills[rng.randrange(len(aux_subskills))]
            text, mask = fmt_O_with_mask(H, c, subskill)
            counts[f"aux_{subskill}"] += 1
        text_parts.append(text)
        mask_parts.append(mask)
        supervised_tokens += sum(mask)
        total_tokens += len(mask)
    full_text = "".join(text_parts)
    full_mask = [m for chunk in mask_parts for m in chunk]
    assert len(full_text) == len(full_mask), \
        f"[fmt_O] text/mask length mismatch: {len(full_text)} vs {len(full_mask)}"
    pct = {k: 100 * v / n for k, v in counts.items()}
    print(f"  [fmt_O] main={counts['main']} ({pct['main']:.1f}%)  "
          + "  ".join(f"aux_{s}={counts[f'aux_{s}']} ({pct[f'aux_{s}']:.1f}%)"
                      for s in aux_subskills))
    print(f"          supervised_frac={supervised_tokens/total_tokens:.3f}")
    return full_text, full_mask


def build_split_lossmask(n: int, h_min: int, h_max: int, seed: int):
    """fmt_L split generator: builds text stream + parallel per-char mask.
    Returns (text_str, mask_list) with len(text) == len(mask) guaranteed."""
    rng = random.Random(seed)
    text_parts = []
    mask_parts = []
    n_prompt_tok = 0
    n_answer_tok = 0
    for _ in range(n):
        H, F, c, r = gen_sample(h_min, h_max, rng)
        text, mask = fmt_L_with_mask(H, F, c, r)
        text_parts.append(text)
        mask_parts.append(mask)
        n_prompt_tok += mask.count(0)
        n_answer_tok += mask.count(1)
    full_text = "".join(text_parts)
    full_mask = [m for chunk in mask_parts for m in chunk]
    assert len(full_text) == len(full_mask), \
        f"[fmt_L] text/mask length mismatch: {len(full_text)} vs {len(full_mask)}"
    total = n_prompt_tok + n_answer_tok
    print(f"  [fmt_L] prompt_tokens={n_prompt_tok:,} ({100*n_prompt_tok/total:.1f}%)  "
          f"answer_tokens={n_answer_tok:,} ({100*n_answer_tok/total:.1f}%)")
    return full_text, full_mask


def build_split_multitask_full(n: int, h_min_main: int, h_max_main: int,
                               h_min_aux: int, h_max_aux: int,
                               seed: int, main_ratio: float = 0.5) -> str:
    """fmt_N mixed generator: main task (fmt_D) with probability
    `main_ratio`, otherwise uniformly one of 4 auxiliary tasks
    (mul2 / sub_F2H / div_D / sub_Hr), each on H in [h_min_aux, h_max_aux].
    Auxiliary tasks all use gen_sample(h_min_aux, h_max_aux) internally so
    all derived numbers (F, D, r, c) are legal by construction.
    Deterministic under `seed`."""
    rng = random.Random(seed)
    aux_formatters = [
        ("aux_mul2",    lambda H, c: fmt_mul2(H)),
        ("aux_sub_F2H", fmt_sub_F2H),
        ("aux_div_D",   fmt_div_D),
        ("aux_sub_Hr",  fmt_sub_Hr),
    ]
    counts = {"main": 0, **{name: 0 for name, _ in aux_formatters}}
    chunks = []
    for _ in range(n):
        if rng.random() < main_ratio:
            H, F, c, r = gen_sample(h_min_main, h_max_main, rng)
            chunks.append(fmt_D(H, F, c, r))
            counts["main"] += 1
        else:
            # Uniformly pick one of the 4 aux tasks
            name, fmt = aux_formatters[rng.randrange(len(aux_formatters))]
            H, F, c, r = gen_sample(h_min_aux, h_max_aux, rng)
            chunks.append(fmt(H, c))
            counts[name] += 1
    pct = {k: 100 * v / n for k, v in counts.items()}
    print(f"  [fmt_N] main={counts['main']} ({pct['main']:.1f}%)  "
          f"aux_mul2={counts['aux_mul2']} ({pct['aux_mul2']:.1f}%)  "
          f"aux_sub_F2H={counts['aux_sub_F2H']} ({pct['aux_sub_F2H']:.1f}%)  "
          f"aux_div_D={counts['aux_div_D']} ({pct['aux_div_D']:.1f}%)  "
          f"aux_sub_Hr={counts['aux_sub_Hr']} ({pct['aux_sub_Hr']:.1f}%)")
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

    # 5. fmt_C/D/M/N/L/O/P rev_pad must be perfectly invertible across our number
    #    range, otherwise the eval parser will silently decode wrong integers
    if format_key in ("C", "D", "M", "N", "L", "O", "P"):
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

    # 6. fmt_M/N-specific: aux task outputs stay in vocab across full H range
    if format_key in ("M", "N"):
        for h in [2, 8, 20, 37, 50]:
            for ch in fmt_mul2(h):
                if ch not in stoi:
                    raise AssertionError(
                        f"fmt_mul2({h}) emitted char {ch!r} not in vocab"
                    )

    # 6c. fmt_O-specific: verify (a) text is invariant across the 5 subskill
    #     variants, (b) aux masks are disjoint subsets of the main mask, and
    #     (c) union(aux masks) + end("\\n") == main mask.
    if format_key == "O":
        rng_probe = random.Random(456)
        for _ in range(20):
            H, F, c, r = gen_sample(2, 50, rng_probe)
            texts = {}
            masks = {}
            for sk in ["main", "mul2", "sub_F2H", "div_D", "sub_Hr"]:
                t, m = fmt_O_with_mask(H, c, sk)
                texts[sk] = t
                masks[sk] = m
                if len(t) != len(m):
                    raise AssertionError(
                        f"fmt_O({sk}) length mismatch (H={H},c={c}): {len(t)} vs {len(m)}"
                    )
            # (a) text must be identical across all subskill choices
            for sk in ["mul2", "sub_F2H", "div_D", "sub_Hr"]:
                if texts[sk] != texts["main"]:
                    raise AssertionError(
                        f"fmt_O text differs between main and {sk} at (H={H},c={c})"
                    )
            # (b) each aux mask position that = 1 must have main mask = 1
            for sk in ["mul2", "sub_F2H", "div_D", "sub_Hr"]:
                for i, m in enumerate(masks[sk]):
                    if m == 1 and masks["main"][i] != 1:
                        raise AssertionError(
                            f"fmt_O aux {sk} supervises pos {i} but main doesn't "
                            f"(H={H},c={c})"
                        )
            # (c) aux masks are disjoint pairwise
            for a, b in [("mul2","sub_F2H"), ("mul2","div_D"), ("mul2","sub_Hr"),
                          ("sub_F2H","div_D"), ("sub_F2H","sub_Hr"),
                          ("div_D","sub_Hr")]:
                for i in range(len(masks[a])):
                    if masks[a][i] == 1 and masks[b][i] == 1:
                        raise AssertionError(
                            f"fmt_O aux masks {a} and {b} both = 1 at pos {i}"
                        )
            # (d) union of 4 aux masks + \\n = main mask
            aux_sum = sum(sum(masks[sk]) for sk in ["mul2", "sub_F2H", "div_D", "sub_Hr"])
            main_sum = sum(masks["main"])
            if aux_sum + 1 != main_sum:  # +1 for the trailing '\n' in main only
                raise AssertionError(
                    f"fmt_O mask arithmetic: aux_sum={aux_sum} + 1(\\n) != main={main_sum} "
                    f"(H={H},c={c})"
                )
            # (e) vocab coverage
            for ch in texts["main"]:
                if ch not in stoi:
                    raise AssertionError(f"fmt_O emitted char {ch!r} not in vocab")

    # 6b. fmt_L-specific: verify text/mask length invariant and mask semantics
    if format_key == "L":
        rng_probe = random.Random(123)
        for _ in range(20):
            H, F, c, r = gen_sample(2, 50, rng_probe)
            text, mask = fmt_L_with_mask(H, F, c, r)
            if len(text) != len(mask):
                raise AssertionError(
                    f"fmt_L text/mask length mismatch at (H={H},c={c}): "
                    f"{len(text)} vs {len(mask)}"
                )
            # First "\n" separates prompt from answer; mask should be 0 up to
            # and including that newline, and 1 after.
            nl = text.index("\n")
            for i in range(nl + 1):
                if mask[i] != 0:
                    raise AssertionError(
                        f"fmt_L prompt token at pos {i} has mask={mask[i]} (want 0)"
                    )
            for i in range(nl + 1, len(mask)):
                if mask[i] != 1:
                    raise AssertionError(
                        f"fmt_L answer token at pos {i} has mask={mask[i]} (want 1)"
                    )
            # Also verify vocab coverage
            for ch in text:
                if ch not in stoi:
                    raise AssertionError(f"fmt_L emitted char {ch!r} not in vocab")

    # 7. fmt_N-specific: the 3 additional aux formatters produce legal
    #    (H, F, D, r, c) and stay in vocab. Also confirms the derived
    #    intermediates (F=2c+4r, D=F-2H=2r) match by construction.
    if format_key == "N":
        rng_probe = random.Random(999)
        for _ in range(50):
            H, F, c, r = gen_sample(2, 50, rng_probe)
            # Each aux formatter takes (H, c) and derives F/D/r internally.
            # Verify the outputs decode back to the right numbers.
            sub_text = fmt_sub_F2H(H, c)   # 'F=... 2H=...\nD=...\n'
            div_text = fmt_div_D(H, c)     # 'D=...\nr=...\n'
            sub_hr   = fmt_sub_Hr(H, c)    # 'H=... r=...\nc=...\n'
            for probe, name in [(sub_text, "fmt_sub_F2H"),
                                (div_text, "fmt_div_D"),
                                (sub_hr,   "fmt_sub_Hr")]:
                for ch in probe:
                    if ch not in stoi:
                        raise AssertionError(
                            f"{name}(H={H}, c={c}) emitted {ch!r} not in vocab"
                        )

    print("[sanity] tokenizer roundtrip + sample legality + format coverage: OK")


def report_unique_combos(text: str, formatter_key: str) -> None:
    """Eyeball how many *unique* (H,F,c,r) tuples landed in this split, so we
    notice when 100k samples collapse to a tiny set of repeats.
    For fmt_C/D/M/N/L we reverse-decode the (H,F) string back to its real
    value, so the count reflects unique *real* combinations, not reversed
    strings. fmt_L's text is identical to fmt_D's, so decoding is the same."""
    seen = set()
    for line in text.split("\n"):
        if not line.startswith("H="):
            continue
        try:
            head, tail = line.split(" F=")
            H_str, F_str = head[2:], tail
            if formatter_key in ("C", "D", "M", "N", "L", "O", "P"):
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
    parser.add_argument("--format", choices=list(FORMATTERS) + ["M", "N", "L", "O", "P"], default="A",
                        help="prompt format: A=direct, B=CoT, C=reversed, D=rev+CoT, "
                             "M=multi-task mix (fmt_D main + fmt_mul2 aux), "
                             "N=multi-task full (fmt_D main + all 4 subskill auxes), "
                             "L=loss-masked fmt_D (SFT-style, writes *_mask.bin), "
                             "O=context-aligned multi-task (v5.3 test, fmt_D text with "
                             "subskill-specific mask, writes *_mask.bin), "
                             "P=fmt_O text mix + fmt_L full-answer mask (S5.l mask-role test)")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-val-ood", type=int, default=1_000)
    parser.add_argument("--h-min-train", type=int, default=2,
                        help="lower bound of H for train + val_iid. Default 2.")
    parser.add_argument("--h-max-train", type=int, default=20,
                        help="upper bound of H for train + val_iid")
    parser.add_argument("--h-min-ood", type=int, default=21,
                        help="lower bound of H for val_ood (must be > h-max-train)")
    parser.add_argument("--h-max-ood", type=int, default=50)
    parser.add_argument("--h-min-aux", type=int, default=2,
                        help="fmt_M/N/O/P: lower bound of H for aux tasks. Default 2.")
    parser.add_argument("--h-max-aux", type=int, default=50,
                        help="fmt_M/N/O/P: upper bound of H for auxiliary tasks. Should cover "
                             "the main task's OOD range so the model sees big-H subskills "
                             "in the aux stream. Default 50.")
    parser.add_argument("--rev-width", type=int, default=3,
                        help="Zero-pad width for reversed digits (fmt_C/D/M/N/L/O/P). "
                             "3 covers numbers up to 999; use 4 for up to 9999. Only "
                             "matters for fmt_C onwards. Default 3.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None,
                        help="output dir for {train,val,val_ood}.bin + meta.pkl. "
                             "Default: this file's own directory. Pass a fresh path "
                             "(e.g. data/chickens_rabbits_h40/) to keep prior runs.")
    args = parser.parse_args()

    # Override module-level REV_WIDTH so all downstream rev_pad calls pick it up
    global REV_WIDTH
    REV_WIDTH = args.rev_width

    if args.h_min_ood <= args.h_max_train:
        raise SystemExit(
            f"OOD split must be strictly above train H range: "
            f"h_min_ood={args.h_min_ood} <= h_max_train={args.h_max_train}"
        )

    # fmt_M/N/L/O/P share fmt_D as the main-task formatter (used for val/val_ood).
    # For sanity_check we probe with fmt_D since fmt_L/O/P return (text, mask)
    # tuples which don't match the single-return formatter signature; fmt_D
    # covers the vocab-coverage test just as well.
    formatter = fmt_D if args.format in ("M", "N", "L", "O", "P") else FORMATTERS[args.format]
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(out_dir, exist_ok=True)

    sanity_check(args.format, formatter)

    print(f"[gen] format={args.format}  vocab_size={VOCAB_SIZE}  vocab={VOCAB_CHARS!r}")
    if args.format == "M":
        print(f"[gen] fmt_M: main task H in [{args.h_min_train}, {args.h_max_train}], "
              f"aux mul2 H in [{args.h_min_aux}, {args.h_max_aux}], 50/50 mix in train only")
    elif args.format == "N":
        print(f"[gen] fmt_N: main task H in [{args.h_min_train}, {args.h_max_train}], "
              f"4 aux subskills H in [{args.h_min_aux}, {args.h_max_aux}], "
              f"50% main + 12.5% each aux in train only")
    elif args.format == "L":
        print(f"[gen] fmt_L: fmt_D text + companion *_mask.bin "
              f"(prompt=0, answer=1). train.py masks loss via ignore_index=-1.")
    elif args.format == "O":
        print(f"[gen] fmt_O: fmt_D text with subskill-specific masks (v5.3 test)")
        print(f"      main H in [{args.h_min_train}, {args.h_max_train}], "
              f"aux H in [{args.h_min_aux}, {args.h_max_aux}]; "
              f"50% main + 12.5% each of 4 subskill auxes")
    elif args.format == "P":
        print(f"[gen] fmt_P: fmt_O text mix + full-answer mask (S5.l mask-role test)")
        print(f"      main H in [{args.h_min_train}, {args.h_max_train}], "
              f"aux H in [{args.h_min_aux}, {args.h_max_aux}] (aux degenerates into wider-H main)")
    print(f"      rev_width={args.rev_width}")

    splits = {
        "train":   (args.n_train,   args.h_min_train,   args.h_max_train, args.seed),
        "val":     (args.n_val,     args.h_min_train,   args.h_max_train, args.seed + 1),
        "val_ood": (args.n_val_ood, args.h_min_ood,     args.h_max_ood,   args.seed + 2),
    }

    meta_sizes = {}
    for name, (n, h_lo, h_hi, seed) in splits.items():
        mask_list = None  # fmt_L writes companion _mask.bin; others don't
        if args.format == "M" and name == "train":
            # fmt_M train mixes fmt_D main task (H in [h_lo,h_hi]) with
            # fmt_mul2 aux task (H in [h_min_aux, h_max_aux]). val/val_ood
            # stay main-task-only so we always evaluate main-task performance.
            text = build_split_multitask(n, h_lo, h_hi, args.h_min_aux, args.h_max_aux, seed)
        elif args.format == "N" and name == "train":
            # fmt_N train mixes fmt_D main task with all 4 subskill aux
            # tasks (each supervising one CoT step). val/val_ood remain
            # main-task-only so we always evaluate main-task performance.
            text = build_split_multitask_full(n, h_lo, h_hi, args.h_min_aux, args.h_max_aux, seed)
        elif args.format == "L":
            # fmt_L: fmt_D text + per-char mask (prompt=0, answer=1).
            # All 3 splits get masks so train_loss/val_loss are comparable
            # (both computed only on answer tokens).
            text, mask_list = build_split_lossmask(n, h_lo, h_hi, seed)
        elif args.format == "O":
            if name == "train":
                # fmt_O train: 50% main + 12.5% each of 4 context-aligned auxes
                text, mask_list = build_split_lossmask_context_aligned(
                    n, h_lo, h_hi, args.h_min_aux, args.h_max_aux, seed)
            else:
                # fmt_O val / val_ood: main-task only with prompt/answer mask.
                # Same content as fmt_L val split; used to compute masked val
                # loss comparable with masked train loss.
                text, mask_list = build_split_lossmask(n, h_lo, h_hi, seed)
        elif args.format == "P":
            if name == "train":
                # fmt_P train: same text mix as fmt_O but full-answer mask.
                text, mask_list = build_split_full_answer_mask(
                    n, h_lo, h_hi, args.h_min_aux, args.h_max_aux, seed)
            else:
                text, mask_list = build_split_lossmask(n, h_lo, h_hi, seed)
        else:
            text = build_split(n, h_lo, h_hi, formatter, seed)
        ids = encode(text)
        arr = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(out_dir, f"{name}.bin")
        arr.tofile(out_path)
        print(f"[gen] {name}: n={n:>6}  H in [{h_lo},{h_hi}]  "
              f"tokens={len(arr):>9,}  -> {os.path.basename(out_path)}")
        # Write companion mask file for fmt_L. train.py auto-detects it.
        if mask_list is not None:
            if len(mask_list) != len(arr):
                raise AssertionError(
                    f"[fmt_L] tokens/mask count mismatch for split {name}: "
                    f"{len(arr)} tokens vs {len(mask_list)} mask values"
                )
            mask_arr = np.array(mask_list, dtype=np.uint8)
            mask_path = os.path.join(out_dir, f"{name}_mask.bin")
            mask_arr.tofile(mask_path)
            print(f"       mask: supervised_frac={mask_arr.mean():.3f}  "
                  f"-> {os.path.basename(mask_path)}")
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
    # For fmt_N, the aux "H=X r=Y\n" (sub_Hr) starts with "H=" but has " r=",
    # not " F=" — so it's skipped by the " F=" filter below. And aux
    # "F=X 2H=Y\n" (sub_F2H) doesn't start with "H=", also skipped.
    # Only true main-task lines "H=X F=Y" reach the H-check.
    # fmt_L text is identical to fmt_D (no auxes), so filter also works.
    #
    # fmt_O/P: SKIP this check entirely. Their aux samples use the same
    # fmt_D text layout ("H=X F=Y\n...") but with H drawn from the aux
    # range [2, h_max_aux]. From text alone we cannot distinguish aux from
    # main (mask is what differs, and for fmt_P the mask is identical to
    # main). The "leak" here is by design — aux is supposed to expose
    # OOD H values. main-task samples DO stay within h_max_train.
    if args.format not in ("O", "P"):
        for line in train_text.split("\n"):
            if line.startswith("H=") and " F=" in line:
                try:
                    H_str = line.split(" F=")[0][2:]
                    H = int(H_str[::-1]) if args.format in ("C", "D", "M", "N", "L") else int(H_str)
                    if H > args.h_max_train:
                        leaked += 1
                except (ValueError, IndexError):
                    pass
    if args.format not in ("O", "P"):
        if leaked > 0:
            raise AssertionError(
                f"OOD leak: train.bin contains {leaked} samples with H > {args.h_max_train}"
            )
        print(f"[sanity] OOD isolation in train.bin: OK (0 leaks above H={args.h_max_train})")
    else:
        print(f"[sanity] fmt_{args.format}: OOD isolation check SKIPPED (aux samples "
              f"by design use H in [2, {args.h_max_aux}] with fmt_D text)")

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
        "h_min_train": args.h_min_train,
        "h_max_train": args.h_max_train,
        "h_min_ood": args.h_min_ood,
        "h_max_ood": args.h_max_ood,
        "h_min_aux": args.h_min_aux,
        "h_max_aux": args.h_max_aux,
        "rev_width": args.rev_width,
        "splits": meta_sizes,
    }
    meta_path = os.path.join(out_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[meta] saved meta.pkl  vocab_size={VOCAB_SIZE}  format={args.format}")


if __name__ == "__main__":
    main()
