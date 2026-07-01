# Train a ~14M-param GPT on fmt_D data — evidence-hardening for v4 "scale
# doesn't help". Direct scaling of S5.e (5M) by another 3x.
#
# Predictions committed at S5.f kickoff (see notes §5.9):
#   train_loss:  0.21-0.22 (0.79M and 5M both plateaued at 0.22; IID
#                objective is already saturated, more params can't push lower)
#   IID em:      100% (lookup capacity trivially scales)
#   OOD em:      2-8% (base rate: v4 holds, still noise floor)
#   OOD 2H:      2-10% (base rate: v4 holds, "multiply by 2" not learned)
#
# Verdict logic:
#   OOD em <= 10% AND 2H <= 15% -> v4 hardened (5M + 14M same story)
#   OOD em >  20% OR  2H >  30% -> v4 falsified, retreat to v3 "scale
#                                    was the bottleneck, 5M just wasn't
#                                    enough". Would require re-thinking
#                                    the "training signal is the real
#                                    bottleneck" claim.
#
# Param count math: 12 * n_layer * n_embd^2 = 12 * 8 * 384^2 ≈ 14.16M
#   (384/8=48 per-head-dim; divides cleanly, no assert crash unlike S5.e's
#   initial n_head=6 attempt at 256d.)

out_dir = 'out-cr-14m'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = '14m-fmt-D'

dataset = 'chickens_rabbits_revcot'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# ~14.16M params: 8 layers of 384d, 8 heads (48 per-head-dim).
# 3x bigger than S5.e's 5M, 18x bigger than the 0.79M baseline.
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.0
bias = False

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

compile = False
