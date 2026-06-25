# Train a tiny GPT on the extended H=[2,40] chickens-and-rabbits dataset (fmt_B).
# Phase 5+ "expanded H range" experiment: same fmt_B as S5.a CoT but train over
# H in [2,40] instead of [2,20]. OOD now means H in [41,50].
#
# Goal: cross-check the "lookup-not-algorithm" verdict from S5.a.
#   - If em on H=[21,40] (was OOD before, now IID) goes to ~100% -> the model
#     is genuinely a per-(H,c)-cell lookup table; expanding the table just
#     extends the lookup region. OOD H=[41,50] should still be ~0%.
#   - If em on H=[41,50] climbs significantly -> the model may be picking up
#     a partial algorithm; "lookup" verdict needs softening.
#
# Unique (H,c) combos in train: sum_{H=2..40}(H+1) = 858 (vs 228 for [2,20]).
# With 100k samples each combo is seen ~116 times, plenty for a lookup model.

out_dir = 'out-cr-h40'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'h40-fmt-B'

dataset = 'chickens_rabbits_h40'
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
