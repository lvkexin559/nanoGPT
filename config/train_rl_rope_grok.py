# S5.ac RL fine-tune config: REINFORCE + KL penalty on RoPE+Grok stack ckpt.
#
# Base ckpt (§5.29): FAR [201,500] greedy em = 90.0%.
# §5.31 bucket diagnosis: [201,300]=96.5%, [301,400]=81.0%, [401,500]=84.5%.
# Target: use verifier reward + policy gradient to close the ~10% FAR gap
# without regressing IID / near-range performance.
#
# Design (see train_rl.py header for rationale):
#   - Focus rollouts on H in [201, 500] where the model has room to grow.
#   - K=4 samples per prompt for GRPO-style group-mean baseline (variance
#     reduction without a critic).
#   - Small lr (1e-5) and KL penalty vs frozen base to prevent forgetting.
#   - Periodic greedy eval on both FAR (target) and IID [5,100] (guard).

base_ckpt      = 'out-cr-wide-5m-O-v4-rope-grok/ckpt.pt'
dataset_dir    = 'data/chickens_rabbits_wide_O_v4'
out_dir        = 'out-cr-rl-rope-grok'

max_steps      = 600         # ~30 min at ~3 s/step
batch_size     = 32          # prompts per update
k_samples      = 4           # completions per prompt (GRPO group)
temperature    = 0.7
top_k          = 10
lr             = 1e-5
kl_beta        = 0.02
baseline_ema   = 0.9
h_min          = 201
h_max          = 500
max_new_tokens = 32

eval_interval  = 100
eval_n         = 100
log_interval   = 20
seed           = 42
device         = 'cuda'
