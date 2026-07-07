# S5.q Grok-A: wide_O v4 baseline + Grokking parameters.
# Everything (data, model, seed) identical to train_cr_wide_5m_O_v4.py
# except weight_decay (0.1 → 0.5) and max_iters (20k → 100k).
#
# Motivation: v4 already saturates IID/BELOW/NEAR to 100% but FAR
# [201, 500] stays at 6.5% because aux only covers [2, 200]. If Grokking
# (Power et al. 2022) triggers, model may phase-transition to a real
# subskill algorithm that extrapolates past the aux boundary. FAR is
# the only "improvement space" left.
#
# Predictions:
#   IID [5,100]:      100%     (unchanged, saturated)
#   BELOW [2,4]:      100%     (unchanged, saturated)
#   NEAR [101,200]:   100%     (unchanged, saturated)
#   FAR [201,500]:
#     * Bull (grokking triggered): 60%+
#     * Neutral (partial phase transition): 20-40%
#     * Bear (no grokking): <15%
#
# Base rate: bear-leaning (~60%). Grokking is well-documented on
# single-step algebraic tasks but rarely on multi-step CoT tasks.
# But wide_O_v4's subskill decomposition is close enough to
# "single step per position" that we might catch a partial transition.

out_dir = 'out-cr-wide-5m-O-grok-a'
eval_interval = 2500       # 40 eval points across the run
eval_iters = 100
log_interval = 500
always_save_checkpoint = False  # only save best val loss (final ckpt likely best)
wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'wide-5m-O-grok-a'
dataset = 'chickens_rabbits_wide_O_v4'  # reuse v4 data (n_train=200k, rev_width=4)
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64
# Same 5M architecture as v4
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.0
bias = False
learning_rate = 1e-3
max_iters = 100000         # 5× longer than v4's 20k (grokking key ingredient)
lr_decay_iters = 100000    # smooth lr decay across full run
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
# The two grokking ingredients:
weight_decay = 0.5         # 5× stronger than nanoGPT default (0.1)
compile = False
