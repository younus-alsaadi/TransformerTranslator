
from tokenizers import Tokenizer # Core fast tokenizer object
from tokenizers.models import WordLevel
from tokenizers.trainers import  WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset #

from src.config import get_weights_file_path, get_config
from src.tf.dataset import TranslationDataset, causal_mask, as_tf_dataset
from src.tf.model import build_transformer,build_encoder_mask, build_decoder_mask,build_decoder_self_mask,build_encoder_self_mask,build_cross_attention_mask
import numpy as np
from src.tf.val import run_validation



def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):

    dataset_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    # split 90/10
    n = len(dataset_raw)
    n_train = int(0.9 * n)
    # reproducible split (HF allows .select)
    indices = np.arange(n)
    rng = np.random.default_rng(1234)
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_raw = [dataset_raw[int(i)] for i in train_idx]
    val_raw = [dataset_raw[int(i)] for i in val_idx]

    # Build PT-like TF dataset, then wrap into tf.data
    train_ds_tf = TranslationDataset(
        dataset=train_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config["lang_src"],
        trg_lang=config["lang_tgt"],
        seq_len=config["seq_len"],
    )
    val_ds_tf = TranslationDataset(
        dataset=val_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config["lang_src"],
        trg_lang=config["lang_tgt"],
        seq_len=config["seq_len"],
    )

    train_ds = as_tf_dataset(train_ds_tf, batch_size=config["batch_size"], shuffle=True)
    # match PT: val batch size 1, shuffled (order doesn’t matter)
    val_ds = as_tf_dataset(val_ds_tf, batch_size=1, shuffle=True)

    # report max lengths
    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    print("="*20)

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_size, vocab_tgt_size):
    model = build_transformer(
        encoder_vocab_size=vocab_src_size,
        decoder_vocab_size=vocab_tgt_size,
        encoder_seq_len=config["seq_len"],
        decoder_seq_len=config["seq_len"],
        d_model=config["d_model"],
    )
    return model


# ===== Loss (padding-aware) =====

def make_loss_fn(pad_id):
    sce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    def loss_fn(logits, labels):
        # logits: (B, T, V), labels: (B, T)
        loss = sce(labels, logits)                    # (B, T)
        mask = tf.cast(tf.not_equal(labels, pad_id), tf.float32)  # (B, T)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-9)
        return loss
    return loss_fn

# ===== Optimizer =====

def make_optimizer(lr):
    return tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-9)

from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

def train_model(config):
    print("Using TensorFlow")
    print("=" * 20)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print("=" * 20)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, tok_src, tok_tgt = get_dataset(config)
    print("Data building done:")
    print("=" * 20)

    model = get_model(config, tok_src.get_vocab_size(), tok_tgt.get_vocab_size())
    print("\nModel building done:")
    print("=" * 20)

    optimizer = make_optimizer(config["lr"])
    loss_fn = make_loss_fn(pad_id=tok_tgt.token_to_id("[PAD]"))

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0, dtype=tf.int64))
    manager = tf.train.CheckpointManager(ckpt, directory=config["model_folder"], max_to_keep=5)

    # resume if requested
    if config["resume"] is not None:
        ckpt_path = get_weights_file_path(config, config["resume"])
        if tf.io.gfile.exists(ckpt_path + ".index"):
            ckpt.restore(ckpt_path)
            print(f"Restored from {ckpt_path}")
        else:
            print(f"Resume checkpoint not found: {ckpt_path}")

    print("Loss and optimizer ready.")
    print("=" * 20)
    print("Training will start:")
    print("=" * 20)

    use_all_batches = (config["max_train_batches"] == -1)

    # figure out steps per epoch (may be unknown for some pipelines)
    card = tf.data.experimental.cardinality(train_ds)
    try:
        steps_in_ds = int(card.numpy())
        if steps_in_ds < 0:  # UNKNOWN (-2) or INFINITE (-1)
            steps_in_ds = None
    except Exception:
        steps_in_ds = None

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            enc_out = model.encode(batch["encoder_input"], batch["encoder_mask"], training=True)
            dec_out = model.decode(enc_out, batch["encoder_mask"], batch["decoder_input"], batch["decoder_mask"], training=True)
            logits  = model.project(dec_out)  # (B, T, V)
            loss    = loss_fn(logits, batch["out_label"])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    global_step = int(ckpt.step.numpy())
    for epoch in range(config["num_epochs"]):
        batch_count = 0

        # set up a tqdm bar with a sensible total
        if use_all_batches:
            total = steps_in_ds  # None is fine; tqdm will be open-ended
        else:
            total = config["max_train_batches"] if steps_in_ds is None else min(config["max_train_batches"], steps_in_ds)

        pbar = tqdm(total=total, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=True)
        running_loss = 0.0

        for batch in train_ds:
            if (not use_all_batches) and (batch_count >= config["max_train_batches"]):
                break

            loss = train_step(batch)
            loss_val = float(loss.numpy())
            running_loss += loss_val
            batch_count += 1
            global_step += 1
            ckpt.step.assign(global_step)

            # update bar + optional line log
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{(running_loss/batch_count):.4f}"})
            pbar.update(1)

            # uncomment if you also want a plain print each batch:
            # print(f"Epoch {epoch+1}/{config['num_epochs']} | batch {batch_count}"
            #       f"{'' if total is None else f'/{total}'} | loss {loss_val:.4f}")

        pbar.close()

        # validation (prints examples + metrics)
        run_validation(
            model=model,
            val_ds=val_ds,
            tokenizer_tgt=tok_tgt,
            max_len_input=config["seq_len"],
            global_step_from_training=global_step,
            num_inference_examples=2,
        )

        # save checkpoint like PT
        ckpt_path = get_weights_file_path(config, f"{epoch:02d}")
        save_path = ckpt.save(ckpt_path)
        print(f"\nSaved checkpoint to {save_path}")
        print("=" * 20)



