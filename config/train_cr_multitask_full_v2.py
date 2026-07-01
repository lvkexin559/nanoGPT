# S5.i fmt_N v2: v5.1 direct depth test.
#
# fmt_N v1 (S5.h) exposed the depth-vs-coverage tradeoff — each aux got
# only 12.5k samples (100k * 0.125), and per-step accuracy on OOD landed
# at 22% instead of the 70% we saw in fmt_M with 50k aux samples. v5.1
# hypothesis: depth is the missing gate; if we scale n_train 4x (100k
# -> 400k) so each aux gets 50k samples (matching fmt_M), per-step
# should climb to fmt_M-like 60-70% and OOD em jump to 30-60%.
#
# ONLY variable changed vs S5.h: dataset name / out_dir. All hyperparams,
# architecture, iters, learning rate are identical. n_train is a
# prepare.py flag, not a train.py flag — data is already generated at
# 400k samples in data/chickens_rabbits_multitask_full_v2/.
#
# Predictions committed at S5.i kickoff (see notes §5.12):
#
#   train_loss:  0.30-0.34 (slight drop from v1's 0.335 as more samples
#                let the model resolve the per-task distributions better)
#   IID em:      100%
#   OOD em:      30-60% (Bull case for v5.1)
#                20-40% (Neutral, depth helps but not fully)
#                <20%   (Bear, depth isn't the only gate)
#   OOD 2H/D/r/c: expect 50-70% each if v5.1 holds
#   parse_fail:  15-25% (pattern confusion should ease with more
#                data per aux, but likely won't vanish)

out_dir = 'out-cr-multitask-full-v2'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'multitask-full-v2-depth'

dataset = 'chickens_rabbits_multitask_full_v2'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# Baseline 0.79M — deliberately NOT scale up; we're isolating "depth".
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
