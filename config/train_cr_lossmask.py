# S5.j fmt_L loss-mask: v5.2 direct falsification test.
#
# Data: fmt_D text (100% main task) + companion train_mask.bin marking
# prompt (mask=0) vs answer (mask=1). train.py get_batch loads mask and
# sets y=-1 at prompt positions, so cross_entropy(ignore_index=-1) only
# supervises answer tokens. Model sees full main-task context at every
# batch but only gets gradient on 2H/D/r/c positions.
#
# ONLY variable changed vs baseline fmt_D: presence of loss mask.
# Model, iters, arch, seed, all data-generation params: identical.
#
# v5.2 hypothesis being tested:
#   fmt_N v2 taught c with 50k aux samples in SIMPLIFIED context ("H=X r=Y\n")
#   -> c per-step 21% (P(c aux transfer efficacy) ≈ 23%). Context distribution
#   shift ate 77% of the learned skill.
#
#   fmt_L teaches c with real main-task context (H=X F=Y\n2H=A D=B r=Y c=?)
#   because loss_mask supervises c IN CONTEXT. Predicts:
#     - c per-step: 60-80% (matches other 3 subskills in fmt_N v2)
#     - OOD em: 40-70% (bull case for v5.2)
#
# Predictions committed at S5.j kickoff (see notes §5.13):
#
#   train_loss (masked): 0.05-0.15 (only answer tokens counted; prompt
#                        tokens are much harder to predict so their
#                        exclusion drops loss significantly)
#   val_loss (masked):   similar to train_loss
#   IID em:              100% (lookup capacity unchanged)
#
#   * Bull (v5.2 hardened):
#       OOD em > 40%,  c per-step > 60%
#       -> context alignment IS the c fix; fmt_L is the complete recipe
#          for extrapolation on multi-step tasks in <1M params
#
#   * Neutral:
#       OOD em 20-40%, c per-step 40-60%
#       -> loss mask helps but doesn't fully resolve; v5.2 needs more nuance
#
#   * Bear (v5.2 partially falsified):
#       OOD em < 20% OR c per-step < 30%
#       -> context alignment isn't the only gate; v5.2 -> v5.3 refinement

out_dir = 'out-cr-lossmask'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'lossmask-fmt-L'

dataset = 'chickens_rabbits_lossmask'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# Baseline 4L/4H/128d — deliberately NOT scale up; we're isolating "loss mask".
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
