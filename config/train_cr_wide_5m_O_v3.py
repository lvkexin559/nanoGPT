# S5.o wide_O v3 depth 500/H (n_train=800k).
# Fourth data point on the v6.4 subskill saturation curve.
#
# depth per H: 800k * 0.125 / 200 H_aux = 500 samples/H
#   (vs 250/H in wide_O_v2, 62/H in wide_O)
#
# Predictions:
#   IID:    100% (unchanged)
#   BELOW:  100% (unchanged)
#   NEAR:   98-99% (Bull: curve saturates smoothly)
#           OR ~96-97% (Bear: cap in place, saturation curve is flat past 250)
#   FAR:    ~9% (unchanged noise floor)

out_dir = 'out-cr-wide-5m-O-v3'
eval_interval = 500
eval_iters = 100
log_interval = 100
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-O-v3'
dataset = 'chickens_rabbits_wide_O_v3'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64
# 5M architecture (same as wide_5m_O / v2)
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
