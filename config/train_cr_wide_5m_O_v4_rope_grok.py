# S5.y Rope+Grok stack: wide_O v4 with BOTH RoPE (§5.27) and grokking
# params (§5.21 audited: wd=0.5, 100k iter). Tests whether the two
# independent FAR-lifters (§5.28.d) stack.
#
# Independent baselines on v4 (post-audit, §5.28):
#   baseline (learned PE, wd=0.1, 20k):   FAR 3.5%
#   + grokking only (wd=0.5, 100k):       FAR 17.0%   (4.9×)
#   + RoPE only    (wd=0.1, 20k):         FAR 24.5%   (7×)
#
# Stack prediction:
#   Bull (multiplicative): FAR 40-60%   (RoPE 7× × grok 4.9× vs base, capped by data coverage)
#   Neutral (additive):    FAR 35-45%   (both push same direction)
#   Bear (redundant):      FAR ~25%     (mechanisms overlap, one dominates)
#
# Base rate: additive is most likely — mechanisms are orthogonal
# (RoPE = relative position, grok = long-training regularization).

out_dir = 'out-cr-wide-5m-O-v4-rope-grok'
eval_interval = 2500
eval_iters = 100
log_interval = 500
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-O-v4-rope-grok'
dataset = 'chickens_rabbits_wide_O_v4'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64
n_layer = 6
n_head = 8
n_embd = 256
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
