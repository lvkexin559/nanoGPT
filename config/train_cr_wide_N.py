# S5.m fmt_N wide (H_train=[5,100], H_aux=[2,200], rev_width=4).
# 3-way middle: aux OOD exposure but simplified context.
# Aux depth per H: 12.5k / 199 = ~63 samples/H. Not deep, expect
# per-step 20-40% at best.

out_dir = 'out-cr-wide-N'
eval_interval = 250
eval_iters = 100
log_interval = 50
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-N'
dataset = 'chickens_rabbits_wide_N'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
compile = False
