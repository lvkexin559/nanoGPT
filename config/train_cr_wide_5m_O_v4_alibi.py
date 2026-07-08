# S5.27 ALiBi ablation: same as wide_5m_O_v4 (v6.5 best), but with
# learned absolute PE replaced by ALiBi (Attention with Linear Bias).
#
# Baseline (v4, learned PE): IID 100 / BELOW 100 / NEAR 100 / FAR 6.5
# NoPE (§5.26):              IID  99.5 / BELOW 100 / NEAR  92 / FAR 1.5
#
# ALiBi question: does the bias-based form help FAR (length extrapolation)
# or match NoPE?
#
# ALiBi slopes for n_head=8: [0.5, 0.25, 0.125, ..., 1/256] (geometric).

out_dir = 'out-cr-wide-5m-O-v4-alibi'
eval_interval = 500
eval_iters = 100
log_interval = 100
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-O-v4-alibi'
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
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
compile = False

# ---- PE type ----
pe_type = 'alibi'
use_pos_emb = False  # legacy; not used since pe_type != 'learned'
