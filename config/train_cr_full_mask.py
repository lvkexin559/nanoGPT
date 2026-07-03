# S5.l fmt_P full-answer mask: v6 mask-role falsification test.
#
# Same text distribution as fmt_O (50% main H in [2,20] + 50% aux H in [2,50],
# all in complete fmt_D layout). Difference: mask supervises the WHOLE answer
# (fmt_L-style) instead of only the target subskill. So aux samples degenerate
# into "extra main-task samples with wider H".
#
# The test: does the specific "supervise only one subskill" mask design in
# fmt_O matter, or does any mask+OOD-exposure combo work?
#
# Predictions committed at S5.l kickoff:
#   train_loss (masked): 0.05-0.10 (67% supervised, similar to fmt_L)
#   IID em (H in [2,20]): 100% (main lookup)
#
#   val_ood H in [21,50]:
#     * fmt_P: ~100% (H is IN training distribution now, this is lookup on
#       the wider training range)
#     * fmt_O: 100% (already known)
#
#   val_ood_extreme H in [51,100] (via eval_cr.py --ood-h-min 51):
#     * fmt_P: 0-20% (lookup breaks at truly unseen H)
#     * fmt_O: 60-100% (algorithm should extrapolate, unless rev_pad width
#       or model size hits some hidden cap)
#
# If fmt_P works on [21,50] but crashes on [51,100] while fmt_O generalizes,
# then mask's role IS "prevent lookup shortcut, force algorithmic learning".

out_dir = 'out-cr-full-mask'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'full-mask-fmt-P'

dataset = 'chickens_rabbits_full_mask'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# Baseline 4L/4H/128d — isolate the mask-role variable.
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
