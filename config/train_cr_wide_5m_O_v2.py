# S5.n wide_O 5M + 4x depth (n_train=400k).
# v6.3 -> v6.4 direct falsification test: is aux depth per H the hidden
# parameter that caps wide_O 5M NEAR at 86% instead of 100%?
#
# depth per H: 400k * 0.125 / 200 H_aux = 250 samples/H (vs 62 in wide_O)
# Everything else identical to wide_5m_O.
#
# Predictions:
#   IID [5, 100]:    100% (unchanged, capacity gate ON)
#   BELOW [2, 4]:    100% (unchanged, aux covers)
#   NEAR [101, 200]: 95-100% (Bull v6.4: depth-per-H matters)
#                    OR 86% ± noise (v6.4 falsified, other bottleneck)
#   FAR [201, 500]:  ~4% (aux doesn't cover, no change expected)

out_dir = 'out-cr-wide-5m-O-v2'
eval_interval = 500
eval_iters = 100
log_interval = 100
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-O-v2'
dataset = 'chickens_rabbits_wide_O_v2'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64
# 5M architecture (same as wide_5m_O)
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
