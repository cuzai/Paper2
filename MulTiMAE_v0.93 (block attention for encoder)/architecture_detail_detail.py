import torch
from utils import *
from transformers import ViTModel, AutoImageProcessor, BertModel, AutoTokenizer

class TemporalEmbedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.processing_info["scaling_cols"]:
            self.embedding = torch.nn.Linear(1, d_model)
        
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].get_num_cls()
            self.embedding = torch.nn.Embedding(num_cls, d_model)
    
    def forward(self, key, val, padding_mask_dict, device):
        return self.embedding(val)

class ImgEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.downsize_linear = torch.nn.Linear(768, d_model)
    
    def forward(self, key, val, padding_mask_dict, device):
        embedding = self.img_model(val).last_hidden_state
        embedding = self.downsize_linear(embedding)
        return embedding

class NlpEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.nlp_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.downsize_linear = torch.nn.Linear(768, d_model)   
    
    def forward(self, key, val, padding_mask_dict, device):
        # Make token_type_ids
        token_type_ids = torch.zeros(val.shape).to(torch.int).to(device)

        # Make attention mask
        attention_mask = padding_mask_dict[f"{key}_revert_padding_mask"]
        mask_for_global_token = torch.ones(attention_mask.shape[0], 1).to(device)
        attention_mask = torch.cat([attention_mask, mask_for_global_token], dim=-1)

        # Embed data
        inputs = {"input_ids":val, "token_type_ids":token_type_ids, "attention_mask":attention_mask}
        embedding = self.nlp_model(**inputs).last_hidden_state
        embedding = self.downsize_linear(embedding)
        
        return embedding


class TemporalRemain(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, key, val, global_token, pos_enc, idx_dict, remain_rto, device):
        # Add global token
        global_token = global_token.unsqueeze(0).repeat(val.shape[0], 1, 1)
        val = torch.cat([global_token, val], dim=1)
        
        # Positional encoding
        val = pos_enc(val)

        # Get remain data
        global_token = val[:, :1, :]
        val = val[:, 1:, :]

        remain_idx = idx_dict[f"{key}_remain_idx"].unsqueeze(-1).repeat(1, 1, val.shape[-1])
        val = torch.gather(val, index=remain_idx, dim=1)
        val = torch.cat([global_token, val], dim=1)

        return val

class ImgRemain(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, key, val, global_token, pos_enc, idx_dict, remain_rto, device):
        # Positional encoding
        val = pos_enc(val)

        # Get remain_data
        global_token = val[:, :1, :]
        val = val[:, 1:, :]

        ### Get indexs
        num_remain = int(val.shape[1] * remain_rto["img"])
        noise = torch.rand(val.shape[0], val.shape[1]).to(device)
        shuffle_idx = torch.argsort(noise, dim=1)

        remain_idx = shuffle_idx[:, :num_remain]
        masked_idx = shuffle_idx[:, num_remain:]
        revert_idx = torch.argsort(shuffle_idx, dim=1)

        remain_padding_mask = torch.ones(remain_idx.shape).to(device)
        revert_padding_mask = torch.ones(revert_idx.shape).to(device)

        ### Apply mask
        val = torch.gather(val, index=remain_idx.unsqueeze(-1).repeat(1, 1, val.shape[-1]), dim=1)
        val = torch.cat([global_token, val], dim=1)

        return val, remain_idx, masked_idx, revert_idx, remain_padding_mask, revert_padding_mask

class NlpRemain(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, key, val, global_token, pos_enc, idx_dict, remain_rto, device):
        # Positional encoding
        val = pos_enc(val)

        # Get remain data
        global_token = val[:, :1, :]
        val = val[:, 1:, :]

        remain_idx = idx_dict[f"{key}_remain_idx"].unsqueeze(-1).repeat(1, 1, val.shape[-1])
        val = torch.gather(val, index=remain_idx, dim=1)
        val = torch.cat([global_token, val], dim=1)
        
        return val


class DynamicRevert(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, key, val, mask_token, pos_enc, idx_dict, padding_mask_dict):
        global_token = val[:, :1, :]
        val = val[:, 1:, :]

        # Replace remain_padding_mask to mask_token
        remain_padding_mask = padding_mask_dict[f"{key}_remain_padding_mask"].unsqueeze(-1).repeat(1, 1, val.shape[-1])
        val = torch.where(remain_padding_mask==1, val, mask_token)

        # Append mask token
        revert_idx = idx_dict[f"{key}_revert_idx"].unsqueeze(-1).repeat(1, 1, val.shape[-1])
        mask_token = mask_token.unsqueeze(0).repeat(val.shape[0], revert_idx.shape[1]-val.shape[1], 1)
        val = torch.cat([val, mask_token], dim=1)
        assert revert_idx.shape == val.shape

        # Apply revert
        val = torch.gather(val, index=revert_idx, dim=1)
        val = torch.cat([global_token, val], dim=1)

        # Apply positional encoding
        val = pos_enc(val)

        return val


class TemporalDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation):
        super().__init__()
        self.cross_attn = MultiheadBlockAttention(d_model, nhead, dropout)
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        # Feed forward
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        if activation == "gelu":
            self.activation = torch.nn.GELU()
        
        self.linear_ff1 = torch.nn.Linear(d_model, d_ff)
        self.linear_ff2 = torch.nn.Linear(d_ff, d_model)
        self.dropout_ff1 = torch.nn.Dropout(dropout)
        self.dropout_ff2 = torch.nn.Dropout(dropout)
    
    def forward(self, tgt, memory, self_attn_padding_mask):
        x = tgt

        cross_attn_output, cross_attn_weight = self._ca_block(self.norm1(x.unsqueeze(-2)), memory, memory)
        x = x + cross_attn_output.squeeze()

        self_attn_output, self_attn_weight = self._sa_block(self.norm2(x), self_attn_padding_mask)
        x = x + self_attn_output

        x = x + self._ff_block(self.norm3(x))
        return x, self_attn_weight, cross_attn_weight
    
    def _ca_block(self, query, key, value):
        x, attn_weight = self.cross_attn(query, key, value)
        return self.dropout1(x), attn_weight

    def _sa_block(self, src, padding_mask):
        x, attn_weight = self.self_attn(src, src, src, key_padding_mask=padding_mask, average_attn_weights=False)
        return self.dropout2(x), attn_weight

    def _ff_block(self, x):
        x = self.linear_ff2(self.dropout_ff1(self.activation(self.linear_ff1(x))))
        return self.dropout_ff2(x)

class NonTemporalDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation):
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        
        # Feed forward
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        if activation == "gelu":
            self.activation = torch.nn.GELU()
        
        self.linear_ff1 = torch.nn.Linear(d_model, d_ff)
        self.linear_ff2 = torch.nn.Linear(d_ff, d_model)
        self.dropout_ff1 = torch.nn.Dropout(dropout)
        self.dropout_ff2 = torch.nn.Dropout(dropout)
    
    def forward(self, tgt, memory, cross_attn_padding_mask, self_attn_padding_mask):
        x = tgt

        cross_attn_output, cross_attn_weight = self._ca_block(self.norm1(x), memory, memory, cross_attn_padding_mask)
        x = x + cross_attn_output

        self_attn_output, self_attn_weight = self._sa_block(self.norm2(x), self_attn_padding_mask)
        x = x + self_attn_output

        x = x + self._ff_block(self.norm3(x))

    
        return x, self_attn_weight, cross_attn_weight
    
    def _ca_block(self, query, key, value, padding_mask):
        x, attn_weight = self.cross_attn(query, key, value, key_padding_mask=padding_mask)
        return self.dropout1(x), attn_weight
    
    def _sa_block(self, src, padding_mask):
        x, attn_weight = self.self_attn(src, src, src, key_padding_mask=padding_mask)
        return self.dropout2(x), attn_weight

    def _ff_block(self, x):
        x = self.linear_ff2(self.dropout_ff1(self.activation(self.linear_ff1(x))))
        return self.dropout_ff2(x)

1==1