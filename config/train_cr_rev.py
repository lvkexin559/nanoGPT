# Train a tiny GPT on the reversed-digits chickens-and-rabbits dataset (fmt_C).
# Phase 5+ S5.c experiment, paired with baseline fmt_A: same H range [2,20],
# same OOD [21,50], same model, only the digit ordering of the answer changes.
#
# Hypothesis (from "Teaching Arithmetic to Small Transformers", Lee et al. 2023):
# emitting digits low-order first lets the model produce each digit using only
# information available at that position (no need to commit to a high-order
# digit before computing the low-order ones), which dramatically improves
# length generalization on arithmetic. If this hypothesis carries over to
# chickens-and-rabbits, we expect OOD em to climb meaningfully above the
# fmt_A baseline (0%) and fmt_B CoT (1%).
#
# If OOD em stays ~0%, that's also informative: it means CoT and reversal
# alone aren't enough at this model scale, and the "lookup not algorithm"
# verdict (§5.4 / §5.5) survives another attack.

out_dir = 'out-cr-rev'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'rev-fmt-C'

dataset = 'chickens_rabbits_rev'
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
