# S5.26 NoPE ablation: same as wide_5m_O_v4 (v6.5 best), but with
# positional embedding disabled (use_pos_emb = False).
#
# Baseline (v4, with wpe): IID 100 / BELOW 100 / NEAR 100 / FAR 6.5
#
# Question: does removing PE break IID, NEAR (both in aux range),
# or leave the model unchanged (causal mask alone carries position)?

out_dir = 'out-cr-wide-5m-O-v4-nope'
eval_interval = 500
eval_iters = 100
log_interval = 100
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-O-v4-nope'
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

# ---- NoPE flag ----
use_pos_emb = False
