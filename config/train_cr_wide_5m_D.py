# S5.m fmt_D wide + 5M scale (H_train=[5,100], rev_width=4).
# v6.2 test: does capacity scale up recover IID from 30% to ~100%
# while NEAR/FAR remain at their respective boundaries?

out_dir = 'out-cr-wide-5m-D'
eval_interval = 500
eval_iters = 100
log_interval = 100
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-D'
dataset = 'chickens_rabbits_wide_D'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64
# 5M architecture (matches S5.e/f baseline scale-up)
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
