import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------- Embedding + Positional Encoding --------

class InputEmbedding(layers.Layer):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)

    def call(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.embedding(x)
        return x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))


class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len_seq: int, dropout: float):
        super().__init__()
        self.dropout = layers.Dropout(dropout)

        # Create pe: (1, max_len_seq, d_model)
        position = tf.range(max_len_seq, dtype=tf.float32)[:, tf.newaxis]  # (seq, 1)
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = tf.zeros((max_len_seq, d_model), dtype=tf.float32)
        # use tensor assignment with concat
        sin_part = tf.sin(position * div_term)  # (seq, d_model/2)
        cos_part = tf.cos(position * div_term)  # (seq, d_model/2)
        pe = tf.concat([sin_part, cos_part], axis=-1)  # (seq, d_model)
        pe = pe[tf.newaxis, ...]  # (1, seq, d_model)

        # register as a non-trainable weight/buffer
        self.pe = tf.Variable(pe, trainable=False, name="pe")

    def call(self, x, training=False):
        # x: (batch, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)


# -------- Core Blocks --------

class FeedForwardBlock(layers.Layer):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.ffn = keras.Sequential(
            [
                layers.Dense(d_ff, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )

    def call(self, x, training=False):
        return self.ffn(x, training=training)


class ResidualConnectionBlock(layers.Layer):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, sublayer, training=False):
        # sublayer is a callable that takes normalized x and returns a tensor
        y = sublayer(self.norm(x), training=training)
        y = self.dropout(y, training=training)
        return x + y


class MultiHeadAttentionBlock(layers.Layer):
    """
    Wraps Keras MultiHeadAttention but keeps the interface similar to your PyTorch block.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0, "d_model is not divisible by h"
        self.h = h
        self.d_model = d_model
        self.mha = layers.MultiHeadAttention(
            num_heads=h, key_dim=d_model // h, dropout=dropout
        )

        # expose attention scores (last call)
        self.last_attention_scores = None

    def call(self, q, k, v, mask=None, training=False):
        # Keras MHA expects attention_mask of shape (batch, T_q, T_k),
        # with True/1 for keep and False/0 for mask.
        out, attn_scores = self.mha(
            query=q,
            key=k,
            value=v,
            attention_mask=mask,               # (B, T_q, T_k) boolean/0-1
            return_attention_scores=True,
            training=training,
        )
        self.last_attention_scores = attn_scores
        return out

# -------- Encoder / Decoder --------

class EncoderBlock(layers.Layer):
    def __init__(self, features: int, self_attn: MultiHeadAttentionBlock,
                 ffn: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.ffn = ffn
        self.res1 = ResidualConnectionBlock(features, dropout)
        self.res2 = ResidualConnectionBlock(features, dropout)

    def call(self, x, src_mask=None, training=False):
        x = self.res1(x, lambda x_norm, training=False: self.self_attn(x_norm, x_norm, x_norm, src_mask, training), training=training)
        x = self.res2(x, lambda x_norm, training=False: self.ffn(x_norm, training), training=training)
        return x


class Encoder(layers.Layer):
    def __init__(self, features: int, layers_list):
        super().__init__()
        self.layers_list = layers_list
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask=None, training=False):
        for lyr in self.layers_list:
            x = lyr(x, src_mask=mask, training=training)
        return self.norm(x)


class DecoderBlock(layers.Layer):
    def __init__(self, features: int, self_attn: MultiHeadAttentionBlock,
                 cross_attn: MultiHeadAttentionBlock, ffn: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = ffn
        self.res1 = ResidualConnectionBlock(features, dropout)
        self.res2 = ResidualConnectionBlock(features, dropout)
        self.res3 = ResidualConnectionBlock(features, dropout)

    def call(self, x, encoder_output, enc_mask=None, dec_mask=None, training=False):
        x = self.res1(x, lambda x_norm, training=False: self.self_attn(x_norm, x_norm, x_norm, dec_mask, training), training=training)
        x = self.res2(x, lambda x_norm, training=False: self.cross_attn(x_norm, encoder_output, encoder_output, enc_mask, training), training=training)
        x = self.res3(x, lambda x_norm, training=False: self.ffn(x_norm, training), training=training)
        return x


class Decoder(layers.Layer):
    def __init__(self, features: int, layers_list):
        super().__init__()
        self.layers_list = layers_list
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, encoder_output, src_mask=None, tgt_mask=None, training=False):
        for lyr in self.layers_list:
            x = lyr(x, encoder_output, enc_mask=src_mask, dec_mask=tgt_mask, training=training)
        return self.norm(x)


class ProjectionLayer(layers.Layer):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = layers.Dense(vocab_size)

    def call(self, x):
        return self.proj(x)  # (batch, seq_len, vocab_size)


# -------- Full Transformer --------

class Transformer(keras.Model):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 encoder_embed: InputEmbedding, decoder_embed: InputEmbedding,
                 encoder_pos: PositionalEncoding, decoder_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embed = encoder_embed
        self.decoder_embed = decoder_embed
        self.encoder_pos = encoder_pos
        self.decoder_pos = decoder_pos
        self.projection_layer = projection_layer

    # Encode/Decode/Project split mirrors your PyTorch API
    def encode(self, src, encode_mask=None, training=False):
        x = self.encoder_embed(src)
        x = self.encoder_pos(x, training=training)
        return self.encoder(x, mask=encode_mask, training=training)

    def decode(self, encoder_output, encoder_mask, tgt, decoder_mask, training=False):
        y = self.decoder_embed(tgt)
        y = self.decoder_pos(y, training=training)
        return self.decoder(y, encoder_output, src_mask=encoder_mask, tgt_mask=decoder_mask, training=training)

    def project(self, x):
        return self.projection_layer(x)


# -------- Utilities for masks (compatible with Keras MHA) --------
# Keras MHA expects attention_mask with shape (B, T_q, T_k) where 1/True = keep, 0/False = mask

def create_padding_mask(seq, pad_id=0):
    # seq: (B, T)
    mask = tf.cast(tf.not_equal(seq, pad_id), tf.float32)  # 1 where keep
    # We'll broadcast later to (B, T_q, T_k); for self-attn we use (B, T, T) by outer-product
    return mask  # (B, T)

def create_look_ahead_mask(size):
    # lower triangular matrix of ones (keep), zeros above diagonal (mask future)
    la = tf.linalg.band_part(tf.ones((size, size), dtype=tf.float32), -1, 0)
    return la  # (T, T)

def build_self_attention_mask(pad_mask):
    # pad_mask: (B, T) -> (B, T, T)
    return pad_mask[:, :, None] * pad_mask[:, None, :]

def build_decoder_mask(tgt_seq, pad_id=0):
    # combine look-ahead with target padding
    B = tf.shape(tgt_seq)[0]
    T = tf.shape(tgt_seq)[1]
    pad = create_padding_mask(tgt_seq, pad_id)         # (B, T)
    la = create_look_ahead_mask(T)                     # (T, T)
    # broadcast: (B, T, T)
    dec_mask = build_self_attention_mask(pad) * la[None, :, :]
    return tf.cast(dec_mask > 0, tf.float32)           # 1 keep, 0 mask

def build_encoder_mask(src_seq, pad_id=0):
    pad = create_padding_mask(src_seq, pad_id)         # (B, T)
    enc_mask = build_self_attention_mask(pad)          # (B, T, T)
    return tf.cast(enc_mask > 0, tf.float32)


def build_encoder_self_mask(src_ids, pad_id):
    # src_ids: (B, T_enc) -> (B, T_enc, T_enc)
    keep = tf.cast(tf.not_equal(src_ids, pad_id), tf.float32)
    return keep[:, :, None] * keep[:, None, :]

def build_decoder_self_mask(tgt_ids, pad_id):
    # tgt_ids: (B, T_dec) -> (B, T_dec, T_dec) with look-ahead
    keep = tf.cast(tf.not_equal(tgt_ids, pad_id), tf.float32)
    pad = keep[:, :, None] * keep[:, None, :]
    T = tf.shape(tgt_ids)[1]
    la = tf.linalg.band_part(tf.ones((T, T), tf.float32), -1, 0)
    return pad * la[None, :, :]

def build_cross_attention_mask(dec_ids, enc_ids, pad_id):
    # dec_ids: (B, T_dec), enc_ids: (B, T_enc) -> (B, T_dec, T_enc)
    enc_keep = tf.cast(tf.not_equal(enc_ids, pad_id), tf.float32)      # (B, T_enc)
    T_dec = tf.shape(dec_ids)[1]
    # Repeat encoder keep mask across decoder time
    return tf.repeat(enc_keep[:, None, :], repeats=T_dec, axis=1)      # (B, T_dec, T_enc)


# -------- Factory to mirror your build_transformer --------

def build_transformer(encoder_vocab_size: int, decoder_vocab_size: int,
                         encoder_seq_len: int, decoder_seq_len: int,
                         d_model: int = 512, N: int = 6, h: int = 8,
                         dropout: float = 0.1, d_ff: int = 2048) -> Transformer:

    encoder_embed = InputEmbedding(d_model, encoder_vocab_size)
    decoder_embed = InputEmbedding(d_model, decoder_vocab_size)
    encoder_pos = PositionalEncoding(d_model, encoder_seq_len, dropout)
    decoder_pos = PositionalEncoding(d_model, decoder_seq_len, dropout)

    # Encoder blocks
    enc_blocks = []
    for _ in range(N):
        self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        ffn = FeedForwardBlock(d_model, d_ff, dropout)
        enc_blocks.append(EncoderBlock(d_model, self_attn, ffn, dropout))

    # Decoder blocks
    dec_blocks = []
    for _ in range(N):
        self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        ffn = FeedForwardBlock(d_model, d_ff, dropout)
        dec_blocks.append(DecoderBlock(d_model, self_attn, cross_attn, ffn, dropout))

    encoder = Encoder(d_model, enc_blocks)
    decoder = Decoder(d_model, dec_blocks)
    proj = ProjectionLayer(d_model, decoder_vocab_size)

    model = Transformer(encoder, decoder, encoder_embed, decoder_embed, encoder_pos, decoder_pos, proj)

    # Xavier/Glorot init for Dense kernels (Keras defaults to glorot_uniform), Embedding uses uniform already.
    return model


