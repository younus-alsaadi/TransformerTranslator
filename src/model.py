import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int,vocabulary_size:int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len_seq:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len_seq
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (max_len_seq, d_model)
        pe = torch.zeros(max_len_seq, d_model)

        # Create a vector represent word in the sentence of shape (max_len_seq,1)
        position = torch.arange(0, max_len_seq, dtype=torch.float).unsqueeze(1)

        # Create a vector of shape (d_model)
        div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer to save it with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)

class LayerNormalization(nn.Module):
     def __init__(self, features: int, eps: float = 10 ** -6) -> None:
         super().__init__()
         self.eps = eps
         self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
         self.beta = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

     def forward(self, x: torch.Tensor) -> torch.Tensor:
         # Keep the dimension for broadcasting
         mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
         # Keep the dimension for broadcasting
         std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
         # eps is to prevent dividing by zero or when std is very small
         return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float)-> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k=d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout) -> torch.Tensor:
        # query,key,value: (B, H, T, d_k)
        d_k = query.shape[-1]

        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, dk) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)
            # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
            # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query_ = self.W_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key_ = self.W_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value_ = self.W_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query_ = query_.view(query_.shape[0], query_.shape[1], self.h, self.d_k).transpose(1, 2)
        key_ = key_.view(key_.shape[0], key_.shape[1], self.h, self.d_k).transpose(1, 2)
        value_ = value_.view(value_.shape[0], value_.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query_, key_, value_, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.W_o(x)


class ResidualConnectionBlock(nn.Module):
    def __init__(self,features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm= LayerNormalization(features)

    def forward(self, x, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnectionBlock(features,dropout), ResidualConnectionBlock(features,dropout)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x

#number or encoder
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization(features)

    def forward(self, x, mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, decoder_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 encoder_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) #weight: shape (vocab_size, d_model) + Bais(vocab_size)


    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,decoder: Decoder, encoder_embed: InputEmbedding,
                 decoder_embed: InputEmbedding, encoder_pos: PositionalEncoding,
                 decoder_pos: PositionalEncoding, projection_layer: ProjectionLayer)-> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embed = encoder_embed
        self.decoder_embed = decoder_embed
        self.encoder_pos = encoder_pos
        self.decoder_pos = decoder_pos
        self.projection_layer = projection_layer

    def encode(self, src, encode_mask):
        src = self.encoder_embed(src)
        src = self.encoder_pos(src)
        return self.encoder(src, encode_mask)

    def decode(self, encoder_output,encoder_mask,tgt, decoder_mask):
        tgt = self.decoder_embed(tgt)
        tgt = self.decoder_pos(tgt)
        return self.decoder(tgt, encoder_output, encoder_mask, decoder_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(encoder_vocab_size:int, decoder_vocab_size:int, encoder_seq_len : int, decoder_seq_len:int,
                      d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048)-> Transformer:
    #N: number of layer for encoder_decoder

    # Create the embedding layers
    encoder_embed = InputEmbedding(d_model, encoder_vocab_size)
    decoder_embed = InputEmbedding(d_model, decoder_vocab_size)

    # Create the positional encoding layers
    encoder_pos = PositionalEncoding(d_model, encoder_seq_len, dropout)
    decoder_pos = PositionalEncoding(d_model, decoder_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

        # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, decoder_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, encoder_embed,decoder_embed,encoder_pos,decoder_pos,projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer