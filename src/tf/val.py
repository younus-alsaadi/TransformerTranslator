
from src.tf.model import build_transformer,build_encoder_mask, build_decoder_mask,build_decoder_self_mask,build_encoder_self_mask,build_cross_attention_mask
import numpy as np
import tensorflow as tf


def _edit_distance(a, b):
    # classic DP Levenshtein
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[-1][-1]

def char_error_rate(pred, ref):
    if not ref: return 0.0 if not pred else 1.0
    return _edit_distance(list(pred), list(ref)) / max(1, len(ref))

def word_error_rate(pred, ref):
    pred_w = pred.split()
    ref_w = ref.split()
    if not ref_w: return 0.0 if not pred_w else 1.0
    return _edit_distance(pred_w, ref_w) / max(1, len(ref_w))

def simple_bleu(pred, refs, n_gram=4, smooth=1e-9):
    """
    Very small BLEU approximation (single reference).
    refs: list[str] length=1 here to mirror PT's usage
    """
    ref = refs[0]
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if len(pred_tokens) == 0:
        return 0.0
    weights = [1.0/n_gram]*n_gram
    p_ns = []
    for n in range(1, n_gram+1):
        pred_ngrams = {}
        for i in range(len(pred_tokens)-n+1):
            k = tuple(pred_tokens[i:i+n])
            pred_ngrams[k] = pred_ngrams.get(k, 0) + 1
        ref_ngrams = {}
        for i in range(len(ref_tokens)-n+1):
            k = tuple(ref_tokens[i:i+n])
            ref_ngrams[k] = ref_ngrams.get(k, 0) + 1
        overlap = 0
        total = max(1, len(pred_tokens)-n+1)
        for k, v in pred_ngrams.items():
            overlap += min(v, ref_ngrams.get(k, 0))
        p_n = (overlap + smooth) / (total + smooth)
        p_ns.append(p_n)
    # brevity penalty
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    bp = 1.0 if pred_len > ref_len else np.exp(1 - ref_len / max(1, pred_len))
    score = bp * np.exp(np.sum([w*np.log(p) for w, p in zip(weights, p_ns)]))
    return float(score)

def run_validation(model, val_ds, tokenizer_tgt, max_len_input, global_step_from_training, num_inference_examples=2):
    print("Running validation...")
    count = 0
    expected = []
    predicted = []
    srcs = []

    console_width = 80
    try:
        # best-effort terminal width detection on macOS
        import shutil
        console_width = shutil.get_terminal_size().columns
    except:
        pass

    for batch in val_ds:
        # enforce batch size 1 like PT
        for k, v in batch.items():
            batch[k] = v[:1]

        enc_in = batch["encoder_input"]  # (1, T)
        src_text = batch["src_text"].numpy()[0].decode("utf-8")
        tgt_text = batch["tgt_text"].numpy()[0].decode("utf-8")

        out_ids = greedy_decode(model, enc_in, tokenizer_tgt, max_len_input)
        pred_text = tokenizer_tgt.decode(out_ids, skip_special_tokens=True)

        srcs.append(src_text)
        expected.append(tgt_text)
        predicted.append(pred_text)

        print("-" * console_width)
        print(f"{'SOURCE:':>12} {src_text}")
        print(f"{'TARGET:':>12} {tgt_text}")
        print(f"{'PREDICTED:':>12} {pred_text}")

        count += 1
        if count == num_inference_examples:
            print("-" * console_width)
            break
    # CER/WER/BLEU over the collected examples
    cer = np.mean([char_error_rate(p, r) for p, r in zip(predicted, expected)]) if expected else 0.0
    wer = np.mean([word_error_rate(p, r) for p, r in zip(predicted, expected)]) if expected else 0.0
    bleu = np.mean([simple_bleu(p, [r]) for p, r in zip(predicted, expected)]) if expected else 0.0

    print(f"validation/cer  = {cer:.4f}   (global_step={global_step_from_training})")
    print(f"validation/wer  = {wer:.4f}   (global_step={global_step_from_training})")
    print(f"validation/BLEU = {bleu:.4f}  (global_step={global_step_from_training})")


def greedy_decode(model, source_ids, tokenizer_tgt, max_len_input):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    pad_id  = tokenizer_tgt.token_to_id("[PAD]")

    # 1) Encoder forward with encoder SELF mask
    enc_self_mask = build_encoder_self_mask(source_ids, pad_id)        # (B, T_enc, T_enc)
    enc_out = model.encode(source_ids, enc_self_mask, training=False)

    # 2) Start decoder with SOS
    dec = tf.constant([[sos_idx]], dtype=tf.int32)                     # (1, 1)

    while tf.shape(dec)[1] < max_len_input:
        # 3) Decoder SELF mask for current length
        dec_self_mask  = build_decoder_self_mask(dec, pad_id)          # (B, T_dec, T_dec)

        # 4) CROSS mask: broadcast encoder padding over current decoder length
        cross_mask     = build_cross_attention_mask(dec, source_ids, pad_id)  # (B, T_dec, T_enc)

        # 5) Decode with proper masks
        dec_out = model.decode(enc_out, cross_mask, dec, dec_self_mask, training=False)
        logits  = model.project(dec_out)                                # (B, T_dec, V)
        next_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)[0].numpy()

        if next_id == eos_idx:
            break

        dec = tf.concat([dec, tf.constant([[next_id]], dtype=tf.int32)], axis=1)

    return dec.numpy().reshape(-1)[1:].tolist()

