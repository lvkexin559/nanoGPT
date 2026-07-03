# S5.k fmt_O context-aligned multi-task: v5.3 direct falsification test.
#
# All samples share the SAME fmt_D text layout ("H=X F=Y\n2H=A D=B r=C c=D\n").
# Difference is in the loss mask:
#   50%   main task, H in [2, h_max_train] — mask=1 on full answer
#   12.5% aux_mul2,     H in [2, h_max_aux] — mask=1 only on "2H=A"
#   12.5% aux_sub_F2H,  H in [2, h_max_aux] — mask=1 only on " D=B"
#   12.5% aux_div_D,    H in [2, h_max_aux] — mask=1 only on " r=C"
#   12.5% aux_sub_Hr,   H in [2, h_max_aux] — mask=1 only on " c=D"
#
# v5.3 hypothesis: OOD subskill transfer requires BOTH
#   (a) aux exposes OOD H values, AND
#   (b) aux uses full main-task context (not simplified prompts)
#
# fmt_N v2 had (a) not (b) — c per-step stuck at 21%.
# fmt_L had (b) not (a) — OOD didn't move.
# fmt_O has BOTH — predicts all 4 per-step around 70%, OOD em 40-70%.
#
# ONLY variable changed vs fmt_N v2: aux text is now full fmt_D (identical
# to main task), so aux training context matches inference context exactly.
#
# Predictions committed at S5.k kickoff (see notes §5.14):
#
#   train_loss (masked): 0.05-0.15 (same order as fmt_L)
#   IID em:              100%
#
#   * Bull (v5.3 hardened):
#       OOD em > 40%, all per-step > 60%
#       -> context alignment × OOD exposure IS the complete recipe;
#          fmt_O is how you unlock multi-step OOD in <1M params
#
#   * Neutral (v5.3 mostly right):
#       OOD em 20-40%, per-step 40-60%
#       -> some improvement but not full unlock; missing nuance
#
#   * Bear (v5.3 partially falsified):
#       OOD em < 20% OR some per-step < 30%
#       -> even (a)+(b) isn't enough; there's a v5.4 in there

out_dir = 'out-cr-context-aligned'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'context-aligned-fmt-O'

dataset = 'chickens_rabbits_context_aligned'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# Baseline 4L/4H/128d — deliberately NOT scale up; we isolate "context alignment".
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
