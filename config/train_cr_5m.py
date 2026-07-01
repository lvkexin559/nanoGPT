# Train a ~5M-param GPT on the reversed-digit + CoT chickens-and-rabbits dataset (fmt_D).
# Phase 5+ S5.e experiment: scale-up axis. Everything except model size is
# identical to S5.d (config/train_cr_revcot.py): same dataset, same iters, same
# seed. Only n_layer / n_head / n_embd change.
#
# Motivation:
#   4 previous experiments (fmt_A/B/C/D + expand-H) all hit OOD em <= 3.5% on
#   0.79M params. §5.7 upshot v3 declared "data-format trick exhausted;
#   the next axis to vary is model scale." This is that experiment.
#
# Predictions committed at S5.e kickoff (see notes §5.8):
#   train_loss:  0.18-0.22 (0.79M was 0.22; hard to fall much further since
#                answer tokens already near 0 loss on IID)
#   IID em:      100% (lookup capacity scales up trivially)
#   OOD em:      5-15% (base-rate pessimistic — multi-step system is harder
#                than the paper's pure addition; 6x scale probably won't
#                fundamentally flip "lookup vs algorithm")
#   OOD 2H:      10-30% (the key diagnostic; if 2H jumps to 50%+ then the
#                model has actually started learning "multiply by 2")
#   OOD digit:   80-90% (already saturated by reversal, little room to grow)
#
# Param count math: params ≈ 12 * n_layer * n_embd^2 = 12 * 6 * 256^2 ≈ 4.72M
#   (Same order of magnitude as Lee et al. 2023's arithmetic scratch models.)

out_dir = 'out-cr-5m'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = '5m-fmt-D'

dataset = 'chickens_rabbits_revcot'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# ~4.72M params: 6 layers of 256d, 8 heads (n_embd must be divisible by
# n_head; 256/8=32 per-head-dim). 6x bigger than 0.79M baseline, aligned
# with the Lee et al. 2023 "small transformer arithmetic" regime.
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.0
bias = False

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

compile = False
