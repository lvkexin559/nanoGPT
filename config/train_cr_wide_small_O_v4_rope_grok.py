# S5.z Small + RoPE + Grokking triple stack (§5.29 successor):
# - small architecture (0.79M) from §5.16 wide-O
# - RoPE (§5.27)
# - grokking params (§5.21 audited, §5.29)
# - same wide_O v4 data as §5.29 (n_train=200k, aux [2,200])
#
# §5.28 revealed small model (0.79M, learned PE, 5k iter, wide_O v1 data)
# already reaches FAR 26% — 7.4× baseline. §5.29 showed RoPE + grok stack
# on 5M reaches FAR 90%. This experiment tests whether the two "small"
# advantages (capacity regularization + RoPE + long training) stack to
# 90%+ FAR at 0.15× params of 5M model.
#
# Predictions:
#   Bull:    FAR 90%+ (small has intrinsic reg + RoPE + grok all stack)
#   Neutral: FAR 60-80% (capacity slightly bottlenecks vs 5M)
#   Bear:    FAR 40-60% (small can't fit the full algorithm in 100k)

out_dir = 'out-cr-wide-small-O-v4-rope-grok'
eval_interval = 2500
eval_iters = 100
log_interval = 500
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-small-O-v4-rope-grok'
dataset = 'chickens_rabbits_wide_O_v4'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64
# Small architecture (~0.79M) — matches §5.16 wide-O
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
compile = False

# ---- RoPE ----
pe_type = 'rope'
rope_theta = 10000.0
use_pos_emb = False

# ---- Grokking ----
weight_decay = 0.5
