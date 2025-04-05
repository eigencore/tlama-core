from tlamacore.data.dataloader import DataLoaderLite
from tlamacore.training.scheduler import get_lr_scheduler
from tlamacore.training.trainer import Trainer

from tlamacore.models.tlama_base import Transformer, TlamaConfig


config = TlamaConfig(
    d_model=512,
    n_layers= 8,
    n_heads= 8,
    n_kv_heads=None,
    vocab_size= 50304,
    multiple_of= 256,  # Ensures hidden layer size in SwiGLU is a multiple of this value
    ffn_dim_multiplier=None,
    norm_eps=1e-5,
    rope_theta=10000.0,
    max_batch_size=32,
    max_seq_len=1024,
    use_parallel=False,
    kv_cache=False
)

model = Transformer(config)

train_data_loader = DataLoaderLite(
    data_dir='edu_fineweb10B',
    batch_size=2,
    seq_len=1024,
    process_rank=0,
    num_process=1,
    split='train',
    shuffle=True,
    verbose=True
)

val_data_loader = DataLoaderLite(
    data_dir='edu_fineweb10B',
    batch_size=2,
    seq_len=1024,
    process_rank=0,
    num_process=1,
    split='val',
    shuffle=False,
    verbose=True
)

scheduler = get_lr_scheduler(
    max_lr=6e-4,
    min_lr=6e-4 * 0.1,
    max_steps=19073,
    warmup_steps=715,
    decay_type='cosine',
    verbose=True,
)

trainer = Trainer(
    model=model,
    scheduler=scheduler,
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    epochs=1,
    steps=19073,
    log_steps=1
)

trainer.train()



