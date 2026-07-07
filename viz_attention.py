"""viz_attention.py — Attention pattern visualization (ASCII bars) for FAR OOD.

Purpose (Q from §5.22 postmortem):
  On FAR (aux-uncovered OOD), is model's attention DIFFUSE (doesn't know
  where to look → PE / architecture is the fix) or FOCUSED on H tokens
  (looks right but computes wrong → RL / arch fix)?

Method:
  1. Load ckpt, build a legal (H, F) FAR prompt "H=<rev>{H} F=<rev>{F}\\n"
  2. Manual forward pass (bypass flash attention) to capture per-layer
     attention weights, shape [n_layer][1, n_head, T, T]
  3. Focus on LAST prompt position — attention row at T-1 tells us where
     model "looks" when about to emit 2H digit 1
  4. ASCII bar chart per layer (avg across heads) — no matplotlib needed

Reads meta.pkl for rev_width to highlight H digit positions.
"""
import os, sys, math, pickle, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT


def load_model_and_meta(ckpt_path, meta_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta


def encode_prompt(prompt, stoi):
    return [stoi[c] for c in prompt]


def get_attention_maps(model, input_ids):
    """Manual forward pass — capture per-layer attention weights.

    Returns list[n_layer] of Tensor(1, n_head, T, T) on CPU.
    Also returns final hidden states (for optional downstream use).

    Bypasses flash attention (which doesn't expose the softmax matrix)
    by manually computing attention scores via einsum-equivalent."""
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(input_ids)
    pos_emb = model.transformer.wpe(pos)
    h = model.transformer.drop(tok_emb + pos_emb)

    causal_mask = torch.tril(torch.ones(t, t, device=device)).view(1, 1, t, t)

    attention_maps = []
    for block in model.transformer.h:
        # Manual attention path
        x_normed = block.ln_1(h)
        attn = block.attn
        B, T, C = x_normed.size()
        q, k, v = attn.c_attn(x_normed).split(attn.n_embd, dim=2)
        n_head = attn.n_head
        head_dim = C // n_head
        k_ = k.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q_ = q.view(B, T, n_head, head_dim).transpose(1, 2)
        v_ = v.view(B, T, n_head, head_dim).transpose(1, 2)

        att_scores = (q_ @ k_.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att_scores = att_scores.masked_fill(causal_mask == 0, float("-inf"))
        att = torch.softmax(att_scores, dim=-1)  # (B, nh, T, T)
        attention_maps.append(att.detach().cpu())

        # Continue block forward for next layer
        y = att @ v_
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = attn.c_proj(y)
        h = h + y
        h = h + block.mlp(block.ln_2(h))

    return attention_maps


def ascii_bar(value, width=40, max_val=1.0):
    """Render a value in [0, max_val] as an ASCII bar of length `width`.
    Uses gradient chars for fine granularity."""
    filled = int((value / max_val) * width)
    frac = ((value / max_val) * width) - filled
    grad_chars = " ▏▎▍▌▋▊▉█"
    grad_idx = min(int(frac * len(grad_chars)), len(grad_chars) - 1)
    return "█" * filled + grad_chars[grad_idx] + " " * max(0, width - filled - 1)


def compute_diffuseness(att_row):
    """Return (uniform_baseline, actual_uniform_ratio):
        uniform_baseline = 1 / seq_len (what uniform attn would look like)
        max attention peak / uniform_baseline
        If ratio ~1: attention is diffuse (uniform-like)
        If ratio >>1: attention is focused
    Also return normalized entropy (0 = perfect focus, 1 = perfect uniform).
    """
    import math as _m
    T = len(att_row)
    uniform = 1.0 / T
    peak = float(att_row.max())
    focus_ratio = peak / uniform
    # normalized entropy
    p = att_row[att_row > 1e-12]
    ent = -(p * (p + 1e-30).log()).sum().item() if hasattr(p, "log") else -sum(
        pi * _m.log(pi + 1e-30) for pi in p.tolist()
    )
    max_ent = _m.log(T)
    ent_norm = ent / max_ent if max_ent > 0 else 0.0
    return uniform, focus_ratio, ent_norm


def visualize(ckpt_path, data_dir, prompts, title_suffix):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_model_and_meta(
        ckpt_path, os.path.join(data_dir, "meta.pkl"), device
    )
    stoi = meta["stoi"]
    rev_width = meta.get("rev_width", 3)
    n_layer = len(model.transformer.h)

    print(f"\n{'='*80}")
    print(f"  {title_suffix}")
    print(f"  ckpt: {ckpt_path}  |  n_layer={n_layer}  |  rev_width={rev_width}")
    print(f"{'='*80}")

    for prompt in prompts:
        ids = encode_prompt(prompt, stoi)
        chars = list(prompt)
        display_chars = [c if c != "\n" else "↵" for c in chars]

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            att_maps = get_attention_maps(model, input_ids)

        print(f"\n────── Prompt: {prompt!r}  (T={len(ids)}) ──────")
        # Header row: prompt chars with H-digit positions marked with *
        h_start, h_end = 2, 2 + rev_width  # H digit indices
        pos_marks = []
        for i in range(len(chars)):
            if h_start <= i < h_end:
                pos_marks.append(f"[{display_chars[i]}]")  # H digits
            else:
                pos_marks.append(f" {display_chars[i]} ")
        print("Position markers ([]=H digit): " + " ".join(pos_marks))

        for l, att in enumerate(att_maps):
            avg = att[0].mean(dim=0)  # (T, T)
            last_row = avg[-1, :]  # (T,) — attention from last position
            _, focus, ent_norm = compute_diffuseness(last_row)
            # Diffuseness verdict
            if ent_norm > 0.85:
                verdict = "DIFFUSE (uniform-like)"
            elif ent_norm > 0.6:
                verdict = "somewhat diffuse"
            elif ent_norm > 0.35:
                verdict = "moderately focused"
            else:
                verdict = "FOCUSED"

            # Weight on H digit positions
            h_weight = float(last_row[h_start:h_end].sum())

            print(f"\n  Layer {l}:  entropy_norm={ent_norm:.3f} "
                  f"[{verdict}]  peak/uniform={focus:.2f}×  "
                  f"H_digit_mass={h_weight:.3f}")
            max_val = float(last_row.max())
            for i, w in enumerate(last_row.tolist()):
                marker = "  ←H" if h_start <= i < h_end else "    "
                bar = ascii_bar(w, width=40, max_val=max_val)
                print(f"    {i:2d} {display_chars[i]!s:<3s} "
                      f"{w:.4f} |{bar}|{marker}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--prompts", nargs="+", required=True,
                   help="list of already-formatted prompts. Use \\n for newline.")
    args = p.parse_args()
    prompts = [p.replace("\\n", "\n") for p in args.prompts]
    visualize(args.ckpt, args.data_dir, prompts, args.title)


if __name__ == "__main__":
    main()
