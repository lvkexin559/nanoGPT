# Train a tiny GPT on the chickens-and-rabbits arithmetic dataset.
# This is the Phase 5+ baseline: fmt_A (direct answer), no CoT, no loss-mask.
# Parameter count ~0.3M; should converge in <5 min on a single GPU.

out_dir = 'out-cr-cot'
eval_interval = 250  # tighter than openwebtext default; dataset is small
eval_iters = 100
log_interval = 50

# Only checkpoint when val improves; baseline overfits trivially on 228 unique
# (H,F) combos and we don't want a stale "final" ckpt to mask that.
always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanogpt-cr'
wandb_run_name = 'cot-fmt-B'

dataset = 'chickens_rabbits'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64  # one sample is at most ~30 chars; 64 gives ~2x headroom

# baby GPT (~0.3M params; shakespeare_char's 10.7M was massive overkill here)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0  # 100k synthetic samples; regularization not the bottleneck
bias = False

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99  # nano-scale tokens/iter -> noisier moment estimates, smooth them

warmup_iters = 100

# torch.compile costs ~1min of frontloaded compile time for a model that
# trains end-to-end in <5min. Net negative. Leave to S5 if curious.
compile = False
