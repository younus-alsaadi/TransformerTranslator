import torch
import torchmetrics
import os
from src.pytorch.dataset import causal_mask


def run_validation(model,validation_dataset,tokenizer_src, tokenizer_tgt, max_len_input, device, print_msg, global_step_from_training,num_inference_examples=2):

    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    # Tries to detect terminal width to print nice separators ('-' * width).
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len_input, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # to get the text from the model_out we use tokenizer_tgt

            # convert to plain Python list and skip special tokens for cleaner text
            ids = model_out.detach().cpu().tolist()
            model_out_text = tokenizer_tgt.decode(ids, skip_special_tokens=True)

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_inference_examples:
                print_msg('-' * console_width)
                break

        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        #wandb.log({'validation/cer': cer, 'global_step': global_step_from_training})
        print("="*20)
        print(f"cer: {cer}")

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        # wandb.log({'validation/wer': wer, 'global_step': global_step_from_training})
        print(f"wer: {wer}")

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        # wandb.log({'validation/BLEU': bleu, 'global_step': global_step_from_training})
        print(f"bleu: {bleu}")
        print("="*20)


# calculate the encoder only one and use only one for the val (always pick the most likely next token.)
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len_input, device ):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    # Encode the source (fixed for this sequence)
    encoder_output = model.encode(source, source_mask)

    # Initialize decoder input with SOS
    decoder_input= torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    # keep generating while current length < max_len_input
    while decoder_input.size(1) < max_len_input:
        # Build causal mask for current length
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # project only the last time step
        logits_last = model.project(output[:, -1])
        _, next_word = torch.max(logits_last, dim=1)

        # stop if EOS predicted (don’t append EOS beyond max_len)
        if next_word.item() == eos_idx:
            break

        # append next token
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )

    return decoder_input.squeeze(0)

