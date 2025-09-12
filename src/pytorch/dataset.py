
from torch.utils.data import Dataset
import torch


class TranslationDataset(Dataset):
    def __init__(self, dataset ,tokenizer_src,tokenizer_tgt, src_lang, trg_lang, seq_len):
        super().__init__()

        # the name of the languages
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.seq_len = seq_len
        self.dataset = dataset

        self.sos_token = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")

        # optional counters for logging
        self.trunc_enc = 0
        self.trunc_dec = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        trg_text = src_target_pair['translation'][self.trg_lang]

        # Transform the text into tokens
        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_tgt.encode(trg_text).ids

        max_enc = self.seq_len - 2
        max_dec = self.seq_len - 1

        # Truncate if needed
        if len(encoder_input_tokens) > max_enc:
            encoder_input_tokens = encoder_input_tokens[:max_enc]
            if not hasattr(self, "trunc_enc"):
                self.trunc_enc = 0
            self.trunc_enc += 1

        if len(decoder_input_tokens) > max_dec:
            decoder_input_tokens = decoder_input_tokens[:max_dec]
            if not hasattr(self, "trunc_dec"):
                self.trunc_dec = 0
            self.trunc_dec += 1

        # make the padding to each sentence
        encoder_num_padding_tokens= self.seq_len - len(encoder_input_tokens)-2 # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        decoder_num_padding_tokens = self.seq_len - len(decoder_input_tokens)-1

        encoder_input = torch.cat([
            torch.tensor([self.sos_token], dtype=torch.int64),
            torch.tensor(encoder_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token], dtype=torch.int64),
            torch.full((encoder_num_padding_tokens,), self.pad_token, dtype=torch.int64),
        ], dim=0)

        decoder_input = torch.cat([
            torch.tensor([self.sos_token], dtype=torch.int64),
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            torch.full((decoder_num_padding_tokens,), self.pad_token, dtype=torch.int64),
        ], dim=0)

        out_label = torch.cat([
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token], dtype=torch.int64),
            torch.full((decoder_num_padding_tokens,), self.pad_token, dtype=torch.int64),
        ], dim=0)

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert out_label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # for self attention, No padding  # (1 batch d, 1 seq d, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(
                decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len), # for self attention, No padding  # (1, 1, seq_len)
            "out_label": out_label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": trg_text,
        }




def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0







