import torch
from datasets import load_dataset # HF Datasets: load bilingual data (e.g., "translation" field)
from tokenizers import Tokenizer # Core fast tokenizer object
from tokenizers.models import WordLevel
from tokenizers.trainers import  WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from src.config import get_weights_file_path, get_config,latest_weights_file_path
from src.pytorch.dataset import TranslationDataset, causal_mask
from src.pytorch.model import build_transformer
import torch.nn as nn
from tqdm import tqdm
from src.pytorch.val import run_validation


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config["tokenizer_file"].format(language))

    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language),trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):

    dataset_raw = load_dataset(f"{config['datasource']}",f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])


    train_ds = TranslationDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = TranslationDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    print(f'len of train data loder: {len(train_dataloader)}')
    print(f'len of val data loder: {len(val_dataloader)}')

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_size, vocab_tgt_size):
    """d_model size of Embedding"""
    model = build_transformer(vocab_src_size,vocab_tgt_size, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    print("Using PyTorch")
    print("=" * 20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("="*20)

    # Make sure the weights folder exists
    Path(f"pytorch/{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    Path(f"pytorch/{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    print("Data building done:")
    print("=" * 20)

    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    print("\nmodel building done:")
    print("=" * 20)

    optimizer=torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch=0
    global_step=0
    preload=config['resume']

    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,
                                                                                                        preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn= nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    print("loss and optimizer building done:")
    print("=" * 20)

    # # define our custom x axis metric
    # wandb.define_metric()
    #
    # # define which metrics will be plotted against it
    #
    # wandb.define_metric("validation/*", step_metric="global_step")
    # wandb.define_metric("train/*", step_metric="global_step")

    print("training will start:")
    print("=" * 20)
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        max_batches=config['max_train_batches']

        use_all_batches=(max_batches==-1)

        total_for_bar = len(train_dataloader) if use_all_batches else min(max_batches, len(train_dataloader))

        batch_itreator=tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}",total=total_for_bar)


        for i, batch in enumerate(batch_itreator, start=1):
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            out_put_target = batch['out_label'].to(device)  # (B, seq_len)


            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), out_put_target.view(-1))
            batch_itreator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # # Log the loss
            # wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if not use_all_batches and i >= max_batches:
                break

        model_filename = get_weights_file_path(config,"pytorch", f"{epoch:02d}")

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_itreator.write(msg), global_step,1)



        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)












