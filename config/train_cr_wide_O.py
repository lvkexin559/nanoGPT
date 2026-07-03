# S5.m fmt_O wide (H_train=[5,100], H_aux=[2,200], rev_width=4).
# 3-way winner candidate: v6 recipe scaled 5x.
# Predicts high em on splits within aux coverage [2,200], collapse on
# [201,500] — direct verification of v6.1 "aux H range decides subskill
# lookup boundary" claim.

out_dir = 'out-cr-wide-O'
eval_interval = 250
eval_iters = 100
log_interval = 50
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-O'
dataset = 'chickens_rabbits_wide_O'
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
