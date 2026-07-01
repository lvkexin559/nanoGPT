# Train nanoGPT on fmt_M multi-task data — the "change training signal"
# v4 falsification attempt (S5.g). Same architecture as baseline (0.79M);
# only the training data mix changes.
#
# fmt_M data composition:
#   50%: fmt_D main task (chickens-and-rabbits, H in [2, 20], reversed+CoT)
#   50%: fmt_mul2 aux task ("multiply by 2", H in [2, 50], reversed)
#
# The aux task exposes the "double" subskill at OOD H values (21-50)
# outside the main task's IID range. This tests whether isolated subskill
# supervision can transfer to the multi-step main task at OOD.
#
# Predictions committed at S5.g kickoff (see notes §5.10):
#
#   train_loss:  0.15-0.22 (aux task is shorter/simpler than main; the
#                mix's overall predictable-token ratio should push loss down)
#   IID em:      100% (main lookup capacity is untouched)
#
#   * Bull case (v4 FALSIFIED):
#       OOD em > 20%,  OOD 2H per-step > 50%
#       -> isolated subskill transferred to main task at OOD, training
#          signal was indeed the bottleneck
#
#   * Neutral (partial transfer):
#       OOD em 5-15%,  OOD 2H per-step 15-40%
#       -> some transfer, v4 needs nuance
#
#   * Bear case (v4 HARDENED to v5):
#       OOD em < 5%,   OOD 2H per-step < 10%
#       -> even isolated subskill supervision doesn't transfer;
#          v4 upgrades to v5: cross-task representation reuse doesn't
#          emerge in <5M params without explicit architectural pressure

out_dir = 'out-cr-multitask'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'multitask-fmt-M'

dataset = 'chickens_rabbits_multitask'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64

# Baseline architecture — same as fmt_A/B/C/D runs. NOT scale-up.
# We're isolating the "training signal" variable; changing scale
# simultaneously would confound the interpretation.
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
