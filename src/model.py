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
         self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

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
    def attention(self, query, key, value, mask, dropout: nn.Dropout) -> torch.Tensor:
        d_k = query.shape(-1)

        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores=attention_scores.softmax(dim=-1) # (batch, h, seq_len, dk) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)
            # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
            # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
            query_ = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
            key_ = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
            value_ = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

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
            return self.w_o(x)


class ResidualConnectionBlock(nn.Module):
    def __init__(self,dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm= LayerNormalization()

    def forward(self, x, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnectionBlock(dropout), ResidualConnectionBlock(dropout)])

    def forward(self, x, src_mask):
        x= self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x

#number or encoder
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization()

    def forward(self, x, mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

















