# Train a tiny GPT on the reversed-digit + CoT chickens-and-rabbits dataset (fmt_D).
# Phase 5+ S5.d experiment: the 4th cell of the (forward vs reversed) × (direct vs CoT)
# grid. Same H range [2,20], same OOD [21,50], same model as A/B/C — only the
# answer encoding changes.
#
# Hypothesis (Lee et al. 2023, "Teaching Arithmetic to Small Transformers"):
# CoT and reversed digits each help a bit individually, but their combination
# is what unlocks length generalization on arithmetic. CoT exposes the per-step
# computation explicitly, and reversal aligns each step's digit emit order with
# its position-wise dependency.
#
# Caveat for our task: 3 of 4 CoT steps (2H, D=F-2H, c=H-r) are addition/
# multiplication-like and should benefit from reversal. The 4th step
# (r = D/2) is division, where humans actually go high-order first — so
# reversal is partially anti-aligned. We expect 2H and D to show the biggest
# per-step OOD jumps; r may stay near the noise floor.
#
# Predictions committed at S5.d kickoff (see notes §5.7):
#   train_loss: 0.22-0.25 (between fmt_B 0.26 and fmt_C 0.28, both effects)
#   IID em:     100% (lookup IID has been 100% across all four experiments)
#   OOD em:     5-25% (optimistic synergy) or <5% (lookup verdict sealed)
#   OOD 2H:     30-70% (the key diagnostic — multiplication is reversal's
#                       sweet spot)
#   OOD r:      0-15% (division is the bottleneck)

out_dir = 'out-cr-revcot'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'revcot-fmt-D'

dataset = 'chickens_rabbits_revcot'
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
