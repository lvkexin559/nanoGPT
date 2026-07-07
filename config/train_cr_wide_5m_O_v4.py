# S5.p wide_O v4: 125 uniq × 44 rep = 5500 total exposure per H.
# Same total-exposure as v2/v3, but skewed toward MORE REPETITION and
# FEWER UNIQUE samples. Isolates "repetition vs uniqueness" trade-off.
#
# Predictions:
#   If repetitions dominate: v4 > v2 > v3 (higher accuracy at higher rep)
#   If uniqueness dominates: v4 < v2 (fewer unique = worse)
#   If total exposure dominates: v4 ≈ v2 ≈ v3

out_dir = 'out-cr-wide-5m-O-v4'
eval_interval = 500
eval_iters = 100
log_interval = 100
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-O-v4'
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
