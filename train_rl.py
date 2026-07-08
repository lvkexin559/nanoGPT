"""S5.ac · RL fine-tune on RoPE+Grok stack ckpt (§5.29 base).

Simple REINFORCE + KL(policy || frozen reference) with running-mean baseline.

Design (kept minimal on purpose):

  - Rollout: sample K completions per prompt with temperature/top-k;
    full CoT ("2H=XXXX D=XXXX r=XXXX c=XXXX\\n") is generated in one shot.
  - Reward: constraint verifier (c + r = H and 2c + 4r = F).  Because the
    verifier is a bijection to (c, r), passing verifier ⇔ exact match.
  - Advantage: reward − baseline, baseline = EMA over recent rewards.
  - Log-prob: teacher-forced forward pass on the sampled sequence, with
    grad enabled; loss mask covers only the generated CoT tokens (not
    the prompt, not the answer's constants like "2H=" letters).
  - KL: keep policy close to frozen ref, per-token forward KL, weight β.
  - Focus:  H in [201, 500]  (FAR range where §5.29 greedy is 90% and
    §5.31 bucket diagnosis shows [301, 500] is only ~82%).

Prints periodic diagnostic:  rolling reward, rolling KL, and full-FAR
greedy em (n=100) so we can watch the ceiling move.

Usage:
    python train_rl.py config/train_rl_rope_grok.py

Config keys required:
    base_ckpt     : path to base ckpt (RoPE+Grok stack, §5.29)
    dataset_dir   : path to data folder (needs meta.pkl)
    out_dir       : where to save fine-tuned ckpt
    max_steps     : total RL updates
    batch_size    : prompts per update
    k_samples     : completions per prompt (K in GRPO-ish grouping)
    temperature   : rollout temperature
    top_k         : rollout top-k
    lr            : optimizer lr (small, this is a fine-tune)
    kl_beta       : KL penalty coefficient
    baseline_ema  : decay for running-mean baseline
    h_min/h_max   : sampling range for rollout prompts (default [201, 500])
    eval_interval : run FAR eval every N steps
    eval_n        : samples per FAR eval
    device        : cuda/cpu
"""
import os
import sys
import time
import math
import pickle
import random
from dataclasses import dataclass

import torch
# NB: use ``Fn`` for torch.nn.functional because we rebind ``F`` at module
# scope (F = 2c + 4r) later — using name ``F`` would shadow the import.
import torch.nn.functional as Fn

from model import GPTConfig, GPT

# ------------------------------------------------------------
# Config (defaults; overridable via configurator.py or CLI file)
# ------------------------------------------------------------
base_ckpt      = "out-cr-wide-5m-O-v4-rope-grok/ckpt.pt"
dataset_dir    = "data/chickens_rabbits_wide_O_v4"
out_dir        = "out-cr-rl-rope-grok"
max_steps      = 1000
batch_size     = 32          # prompts per update
k_samples      = 4           # completions per prompt
temperature    = 0.7
top_k          = 10
lr             = 1e-5
kl_beta        = 0.02        # KL penalty vs ref
baseline_ema   = 0.9
h_min          = 201
h_max          = 500
max_new_tokens = 32
eval_interval  = 100
eval_n         = 100
log_interval   = 20
seed           = 42
device         = "cuda"

# Load config file if given as CLI arg (nanoGPT pattern: exec-based override)
config_keys = [k for k, v in globals().items()
               if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
if len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
    print(f"[cfg] loading overrides from {sys.argv[1]}")
    exec(open(sys.argv[1]).read())
config = {k: globals()[k] for k in config_keys}
print("[cfg] final config:")
for k in sorted(config.keys()):
    print(f"  {k} = {config[k]!r}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
random.seed(seed)


# ------------------------------------------------------------
# Tokenizer / helpers  (mirror eval_cr.py to keep behavior identical)
# ------------------------------------------------------------
with open(os.path.join(dataset_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta["vocab_size"]
rev_width = meta.get("rev_width", 3)


def encode(s: str):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


def rev_pad(n: int) -> str:
    return str(n).zfill(rev_width)[::-1]


def build_prompt(H: int, F: int) -> str:
    return f"H={rev_pad(H)} F={rev_pad(F)}\n"


def sample_hcr(rng, lo, hi):
    """Uniform over (H, c) with r = H - c, F = 2c + 4r."""
    H = rng.randint(lo, hi)
    c = rng.randint(0, H)
    r = H - c
    F = 2 * c + 4 * r
    return H, F, c, r


# ------------------------------------------------------------
# Load base model + frozen reference
# ------------------------------------------------------------
print(f"[load] base ckpt = {base_ckpt}")
ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
cfg = GPTConfig(**ckpt["model_args"])
assert cfg.vocab_size == vocab_size, f"vocab mismatch: {cfg.vocab_size} vs {vocab_size}"

model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.train()

ref_model = GPT(cfg).to(device)
ref_model.load_state_dict(ckpt["model"])
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in model.parameters())
print(f"[load] model loaded ({n_params/1e6:.2f}M params, "
      f"pe_type={cfg.pe_type}, base val_loss={ckpt.get('best_val_loss','?'):.4f})")


# ------------------------------------------------------------
# Sampling: generate K completions per prompt (batched)
# ------------------------------------------------------------
@torch.no_grad()
def batched_generate(model, prompt_ids_list, max_new, temperature, top_k):
    """
    prompt_ids_list: list of list[int]; all prompts must have equal length
                     (they do in fmt_O: 14 chars each).
    Returns:
        full_ids: (B, T)  tensor of prompt+generated tokens
        gen_mask: (B, T)  1 on generated positions (prompt=0)
    """
    B = len(prompt_ids_list)
    T_prompt = len(prompt_ids_list[0])
    ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=device)  # (B, Tp)
    gen_mask_list = []
    for _ in range(max_new):
        idx_cond = ids if ids.size(1) <= model.config.block_size \
            else ids[:, -model.config.block_size:]
        logits, _ = model(idx_cond)              # (B, 1, V)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = Fn.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        ids = torch.cat([ids, next_id], dim=1)
        gen_mask_list.append(True)
    T = ids.size(1)
    gen_mask = torch.zeros(B, T, dtype=torch.float, device=device)
    gen_mask[:, T_prompt:] = 1.0
    return ids, gen_mask


# ------------------------------------------------------------
# Reward: constraint verifier on parsed (c, r)
# ------------------------------------------------------------
import re

_RE_CR = re.compile(r"c=(\d+)\s+r=(\d+)|2H=(\d+)\s+D=(\d+)\s+r=(\d+)\s+c=(\d+)")


def _unrev_int(s: str) -> int:
    return int(s[::-1]) if s else 0


def parse_completion(gen_text: str):
    """Parse fmt_O generated line. Returns dict with c, r or {} on fail."""
    line = gen_text.split("\n")[0]
    out = {}
    for key, pat in [
        ("two_h", r"2H=(\d+)"),
        ("D",     r"D=(\d+)"),
        ("r",     r"r=(\d+)"),
        ("c",     r"c=(\d+)"),
    ]:
        m = re.search(pat, line)
        if m is not None:
            out[key] = _unrev_int(m.group(1))
    return out


def compute_reward(prompt_ids, gen_ids, H, F, c_gt, r_gt):
    """Return (reward: float, parsed: dict, verifier_pass: bool)."""
    text = decode(gen_ids.tolist())
    parsed = parse_completion(text)
    if "c" not in parsed or "r" not in parsed:
        return 0.0, parsed, False
    c_p, r_p = parsed["c"], parsed["r"]
    verifier_pass = (c_p + r_p == H) and (2 * c_p + 4 * r_p == F)
    reward = 1.0 if verifier_pass else 0.0
    return reward, parsed, verifier_pass


# ------------------------------------------------------------
# Loss:  REINFORCE (advantage-weighted logprob)  +  KL vs ref
# ------------------------------------------------------------
def compute_loss(model, ref_model, ids, gen_mask, advantages, kl_beta):
    """
    ids:         (B, T)      full sequence (prompt + gen)
    gen_mask:    (B, T)      1 on generated positions
    advantages:  (B,)        per-sample scalar advantage
    Returns:  loss, pg_loss, kl_loss, mean_logp
    """
    # Model logits at position t predict token at t+1 → shift target by 1.
    # NB: nanoGPT forward only returns last-position logits unless
    # ``targets`` is passed. We pass targets to force full-sequence
    # lm_head compute; the returned loss is discarded.
    # Slices are non-contiguous; the ``.view(-1)`` inside GPT.forward's
    # cross_entropy needs contiguous inputs.
    inp = ids[:, :-1].contiguous()
    targets = ids[:, 1:].contiguous()
    logits, _ = model(inp, targets=targets)          # (B, T-1, V)
    logp = Fn.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    with torch.no_grad():
        ref_logits, _ = ref_model(inp, targets=targets)
        ref_logp = Fn.log_softmax(ref_logits, dim=-1)

    # Mask covers positions t whose target (at t+1) is a generated token.
    # gen_mask marks generated positions t+1; shift by 1 → position t.
    loss_mask = gen_mask[:, 1:]                     # (B, T-1)
    n_tok = loss_mask.sum().clamp(min=1)

    # Sequence log-prob under current policy (only generated tokens)
    seq_logp = (tok_logp * loss_mask).sum(dim=-1)   # (B,)
    pg_loss = -(advantages.detach() * seq_logp).mean()

    # KL(policy || ref) per token, only on generated positions
    # KL(p || q) = sum_v p(v) [log p(v) - log q(v)]
    probs = logp.exp()                              # (B, T-1, V)
    kl = (probs * (logp - ref_logp)).sum(dim=-1)    # (B, T-1)
    kl_loss = (kl * loss_mask).sum() / n_tok

    loss = pg_loss + kl_beta * kl_loss
    return loss, pg_loss.detach(), kl_loss.detach(), (tok_logp * loss_mask).sum() / n_tok


# ------------------------------------------------------------
# Eval helper: greedy em on a range
# ------------------------------------------------------------
@torch.no_grad()
def eval_range(model, lo, hi, n, seed_):
    rng = random.Random(seed_)
    model.eval()
    em = 0
    for _ in range(n):
        H, F, c_gt, r_gt = sample_hcr(rng, lo, hi)
        prompt = build_prompt(H, F)
        pids = encode(prompt)
        pt = torch.tensor([pids], dtype=torch.long, device=device)
        out = model.generate(pt, max_new_tokens=max_new_tokens,
                             temperature=0.01, top_k=1)
        text = decode(out[0].tolist())[len(prompt):]
        parsed = parse_completion(text)
        if parsed.get("c") == c_gt and parsed.get("r") == r_gt:
            em += 1
    model.train()
    return em / n


# ------------------------------------------------------------
# RL training loop
# ------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99),
                              weight_decay=0.0)
rng = random.Random(seed)
baseline = 0.5
best_far_em = 0.0
recent_rewards = []
recent_kls = []
recent_pg = []

# Baseline eval before RL
print(f"\n[baseline eval] before RL:")
em_before = eval_range(model, h_min, h_max, eval_n, 5001)
em_iid_before = eval_range(model, 5, 100, eval_n, 5002)
print(f"  FAR [{h_min},{h_max}] em = {em_before*100:.1f}%   "
      f"IID [5,100] em = {em_iid_before*100:.1f}%")

t0 = time.time()
for step in range(1, max_steps + 1):
    # -------- Rollout: build B*K prompts, sample completions ----------
    prompt_ids_batch = []
    gts = []            # list of (H, F, c_gt, r_gt) for each row
    for _ in range(batch_size):
        H, F, c_gt, r_gt = sample_hcr(rng, h_min, h_max)
        prompt = build_prompt(H, F)
        pids = encode(prompt)
        for _k in range(k_samples):
            prompt_ids_batch.append(pids)
            gts.append((H, F, c_gt, r_gt))

    ids, gen_mask = batched_generate(model, prompt_ids_batch, max_new_tokens,
                                     temperature, top_k)

    # -------- Reward per row ----------
    T_prompt = len(prompt_ids_batch[0])
    rewards = torch.zeros(ids.size(0), device=device)
    for i, (H, F, c_gt, r_gt) in enumerate(gts):
        gen_ids = ids[i, T_prompt:]
        r, _, _ = compute_reward(prompt_ids_batch[i], gen_ids, H, F, c_gt, r_gt)
        rewards[i] = r

    # -------- Advantages ----------
    # Group-relative baseline within each prompt (GRPO-lite):
    r_view = rewards.view(batch_size, k_samples)                  # (B, K)
    group_mean = r_view.mean(dim=1, keepdim=True)                # (B, 1)
    advantages = (r_view - group_mean).view(-1)                  # (B*K,)
    # Also update running-mean baseline for logging
    baseline = baseline_ema * baseline + (1 - baseline_ema) * rewards.mean().item()

    # -------- Loss & optimizer step ----------
    loss, pg_loss, kl_loss, mean_logp = compute_loss(
        model, ref_model, ids, gen_mask, advantages, kl_beta
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    recent_rewards.append(rewards.mean().item())
    recent_kls.append(kl_loss.item())
    recent_pg.append(pg_loss.item())

    # -------- Logging ----------
    if step % log_interval == 0:
        rr = sum(recent_rewards[-log_interval:]) / log_interval
        rk = sum(recent_kls[-log_interval:]) / log_interval
        rp = sum(recent_pg[-log_interval:]) / log_interval
        dt = time.time() - t0
        print(f"[step {step:5d}/{max_steps}]  reward={rr:.3f}  "
              f"baseline={baseline:.3f}  KL={rk:.4f}  pg_loss={rp:+.4f}  "
              f"loss={loss.item():+.4f}  elapsed={dt:.1f}s")

    # -------- Periodic eval ----------
    if step % eval_interval == 0 or step == max_steps:
        em_far = eval_range(model, h_min, h_max, eval_n, 5001)
        em_iid = eval_range(model, 5, 100, eval_n, 5002)
        print(f"[eval @ step {step}]  FAR em = {em_far*100:.1f}%  "
              f"IID em = {em_iid*100:.1f}%")
        if em_far > best_far_em:
            best_far_em = em_far
            save_path = os.path.join(out_dir, "ckpt.pt")
            torch.save({
                "model": model.state_dict(),
                "model_args": ckpt["model_args"],
                "iter_num": step,
                "best_far_em": em_far,
                "iid_em": em_iid,
                "config": config,
            }, save_path)
            print(f"  ↑ saved (FAR em {em_far*100:.1f}% new best) → {save_path}")

print("\n=== RL done ===")
print(f"best FAR em (over training) = {best_far_em*100:.1f}%")
