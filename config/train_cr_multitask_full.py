# Train nanoGPT on fmt_N multi-task-full data — v5 direct falsification test.
# All 4 CoT subskills supervised as isolated auxiliary tasks, on top of
# fmt_D main task. Same architecture as baseline (0.79M); only train mix
# changes.
#
# fmt_N data composition:
#   50%    fmt_D main task (chickens-and-rabbits, H in [2, 20], reversed+CoT)
#   12.5%  aux_mul2     ("H=X\n2H=Y\n",       H in [2, 50])
#   12.5%  aux_sub_F2H  ("F=X 2H=Y\nD=Z\n",   H in [2, 50])
#   12.5%  aux_div_D    ("D=X\nr=Y\n",        H in [2, 50])
#   12.5%  aux_sub_Hr   ("H=X r=Y\nc=Z\n",    H in [2, 50])
#
# Each aux task supervises EXACTLY ONE of the main task's 4 CoT steps in
# isolation, with H covering the main task's OOD range. If v5's "training
# signal + compositional coverage" claim is right, main-task OOD em should
# jump to 60-80% because all 4 subskills should transfer.
#
# Predictions committed at S5.h kickoff (see notes §5.11):
#
#   train_loss: 0.24-0.32 (mixed distribution; hard to pin exactly)
#   IID em:     100% (main lookup capacity unchanged)
#
#   * Bull case (v5 hardened):
#       OOD em > 40%, per-step 4 steps each > 50%
#       -> subskill transfer works across the board; fmt_N is the recipe
#          for OOD extrapolation on multi-step tasks in <1M params
#
#   * Neutral (v5 mostly right but subskill depth issue):
#       OOD em 20-40%, per-step each 40-70%
#       -> individual subskills partly transfer but cascade error persists;
#          may need more aux samples (fmt_N v2: expand n_train)
#
#   * Bear case (v5 partially falsified):
#       OOD em < 20% OR some per-step < 30%
#       -> subskill transfer bounded; possibly needs "coupled aux tasks"
#          (aux that shares more context with main) or bigger model

out_dir = 'out-cr-multitask-full'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'multitask-full-fmt-N'

dataset = 'chickens_rabbits_multitask_full'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# Baseline architecture — same as all prior experiments. NOT scale-up.
# We're isolating the "training signal composition" variable.
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
