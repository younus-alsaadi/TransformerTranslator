import tensorflow as tf
import numpy as np


class TranslationDataset():
    def __init__(self, dataset ,tokenizer_src,tokenizer_tgt, src_lang, trg_lang, seq_len):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        self.dataset = dataset

        self.sos_token = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")

        # optional counters for logging (same names)
        self.trunc_enc = 0
        self.trunc_dec = 0

        self._max_enc = seq_len - 2  # +SOS +EOS
        self._max_dec = seq_len - 1  # +SOS (EOS goes to label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.trg_lang]

        # tokenize
        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # truncate
        if len(enc_tokens) > self._max_enc:
            enc_tokens = enc_tokens[:self._max_enc]
            self.trunc_enc += 1

        if len(dec_tokens) > self._max_dec:
            dec_tokens = dec_tokens[:self._max_dec]
            self.trunc_dec += 1

        # paddings
        enc_pad = self.seq_len - len(enc_tokens) - 2
        dec_pad = self.seq_len - len(dec_tokens) - 1

        # tensors
        encoder_input = np.array(
            [self.sos_token] + enc_tokens + [self.eos_token] + [self.pad_token] * enc_pad,
            dtype=np.int32,
        )
        decoder_input = np.array(
            [self.sos_token] + dec_tokens + [self.pad_token] * dec_pad,
            dtype=np.int32,
        )
        out_label = np.array(
            dec_tokens + [self.eos_token] + [self.pad_token] * dec_pad,
            dtype=np.int32,
        )

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert out_label.shape[0] == self.seq_len

        # --- masks to match your shapes/semantics ---
        # encoder_mask: (1,1,T), 1=keep, 0=mask
        enc_keep = (encoder_input != self.pad_token).astype(np.float32)  # (T,)
        encoder_mask = enc_keep[None, None, :]  # (1,1,T)

        #decoder_mask: (1, T, T) = (no - pad broadcast) & (causal)
        #(1,T) & (1,T,T) -> broadcasting. We replicate the result directly.
        dec_keep = (decoder_input != self.pad_token).astype(np.float32)  # (T,)
        # broadcast keep across rows/cols:
        dec_pad_mask = dec_keep[:, None] * dec_keep[None, :]  # (T,T)
        dec_causal = causal_mask(self.seq_len).numpy()  # (1,T,T)
        decoder_mask = dec_pad_mask[None, :, :] * dec_causal

        return {
            "encoder_input": encoder_input,  # (T,)
            "decoder_input": decoder_input,  # (T,)
            "encoder_mask": encoder_mask,  # (1,1,T)
            "decoder_mask": decoder_mask,  # (1,T,T)
            "out_label": out_label,  # (T,)
            "src_text": np.array(src_text, dtype=object),
            "tgt_text": np.array(tgt_text, dtype=object),
        }


def causal_mask(size: int) -> tf.Tensor:
    # (1, T, T) lower-triangular 1s (keep), 0s above diagonal (mask)
    mask = tf.linalg.band_part(tf.ones((size, size), dtype=tf.float32), -1, 0)  # (T,T)
    return mask[tf.newaxis, :, :]  # (1,T,T)


def as_tf_dataset(ds_tf: TranslationDataset, batch_size: int = 32, shuffle=True, buffer_size=5000):
    seq_len = ds_tf.seq_len

    output_signature = {
        "encoder_input": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        "decoder_input": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        "encoder_mask":  tf.TensorSpec(shape=(1, 1, seq_len), dtype=tf.float32),
        "decoder_mask":  tf.TensorSpec(shape=(1, seq_len, seq_len), dtype=tf.float32),
        "out_label":     tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        "src_text":      tf.TensorSpec(shape=(), dtype=tf.string),
        "tgt_text":      tf.TensorSpec(shape=(), dtype=tf.string),
    }

    def gen():
        for i in range(len(ds_tf)):
            yield ds_tf[i]

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
