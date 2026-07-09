# S5.ac cross-seed replicate: seed=1000 (§5.33 original was seed=42).
# All other hyperparameters IDENTICAL to train_rl_rope_grok.py.
# Publication-ready: mean±std across 3 training seeds.

base_ckpt      = 'out-cr-wide-5m-O-v4-rope-grok/ckpt.pt'
dataset_dir    = 'data/chickens_rabbits_wide_O_v4'
out_dir        = 'out-cr-rl-rope-grok-s1000'

max_steps      = 600
batch_size     = 32
k_samples      = 4
temperature    = 0.7
top_k          = 10
lr             = 1e-5
kl_beta        = 0.02
baseline_ema   = 0.9
h_min          = 201
h_max          = 500
max_new_tokens = 32

eval_interval  = 100
eval_n         = 100
log_interval   = 20
seed           = 1000
device         = 'cuda'
