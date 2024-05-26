import torch
from utils import *
from transformers import ViTModel, AutoImageProcessor, BertModel, AutoTokenizer

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

class TemporalEmbedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model, global_token):
        super().__init__()
        self.global_token = global_token
        if col in data_info.processing_info["scaling_cols"]:
            self.embedding = torch.nn.Linear(1, d_model)
        
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].get_num_cls()
            self.embedding = torch.nn.Embedding(num_cls, d_model)
    
    def forward(self, key, val, padding_mask_dict, device):
        embedding = self.embedding(val)

        global_token = self.global_token.unsqueeze(0).repeat(embedding.shape[0], 1, 1)
        embedding = torch.cat([global_token, embedding], dim=1)
        return embedding

class ImgEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.downsize_linear = torch.nn.Linear(768, d_model)
    
    def forward(self, key, val, padding_mask_dict, device):
        embedding = self.img_model(val).last_hidden_state
        embedding = self.downsize_linear(embedding)
        return embedding

class _ImgEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.patch_size = 16
        self.linear = torch.nn.Linear(self.patch_size**2 * 3, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, key, val, padding_mask_dict, device):
        patches = patchify(val, self.patch_size)
        patches = self.linear(patches)
        # patches = self.norm(patches)

        return patches

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
    
    def forward(self, data_dict, padding_mask_dict, remain_rto, temporal_cols, device):
        result_dict, idx_dict = {}, {}
        # Concat data
        concat_data_li, concat_padding_mask_li = [], []
        for col in temporal_cols:
            temporal_data = data_dict[col]
            temporal_padding_mask = padding_mask_dict["temporal_padding_mask"]
            padding_mask_for_global_token = torch.ones(temporal_data.shape[0], 1, 1).to(device)
            temporal_padding_mask = torch.cat([padding_mask_for_global_token, temporal_padding_mask], dim=1)

            concat_data_li.append(temporal_data)
            concat_padding_mask_li.append(temporal_padding_mask)
        
        concat_data = torch.stack(concat_data_li, dim=-2)
        concat_padding_mask = torch.cat(concat_padding_mask_li, dim=-1)
    
        # Remain mask
        num_modality = concat_data.shape[-2]
        num_remain = int(num_modality * remain_rto)
        
        noise = torch.rand(concat_data.shape[:-1]).to(device)
        shuffle_idx = torch.argsort(noise, dim=-1)

        remain_idx = shuffle_idx[:, :, :num_remain]
        masked_idx = shuffle_idx[:, :, num_remain:]
        revert_idx = torch.argsort(shuffle_idx, dim=-1)

        # Apply mask
        concat_data = torch.gather(concat_data, index=remain_idx.unsqueeze(-1).repeat(1, 1, 1, concat_data.shape[-1]), dim=-2)
        concat_padding_mask = torch.gather(concat_padding_mask, index=remain_idx, dim=-1)

        # Obtain revert padding mask
        revert_padding_mask = padding_mask_dict[f"temporal_padding_mask"]
        padding_mask_for_global_token = torch.ones(revert_padding_mask.shape[0], 1, 1).to(device)
        revert_padding_mask = torch.cat([padding_mask_for_global_token, revert_padding_mask], dim=1)
        padding_mask_dict[f"temporal_revert_padding_mask"] = revert_padding_mask


        result_dict["temporal"] = concat_data
        idx_dict.update({"temporal_remain_idx":remain_idx, "temporal_masked_idx":masked_idx, "temporal_revert_idx":revert_idx})
        padding_mask_dict.update({"temporal_remain_padding_mask":concat_padding_mask})

        return result_dict, idx_dict, padding_mask_dict

class ImgRemain(torch.nn.Module):
    def __init__(self, global_token):
        super().__init__()
        self.global_token = global_token
    
    def forward(self, data_dict, remain_rto, img_cols, device):
        result_dict, idx_dict, padding_mask_dict = {}, {}, {}
        # Get indexs
        for col in img_cols:
            val = data_dict[col]
            
            # # Add global token
            # global_token = self.global_token.unsqueeze(0).repeat(val.shape[0], 1, 1)
            # val = torch.cat([global_token, val], dim=1)
            
            # Split global token
            global_token = val[:, :1, :]
            val = val[:, 1:, :]

            # Apply remain
            num_remain = int(val.shape[1] * remain_rto)
            noise = torch.rand(val.shape[0], val.shape[1]).to(device)
            shuffle_idx = torch.argsort(noise, dim=1)

            remain_idx = shuffle_idx[:, :num_remain]
            masked_idx = shuffle_idx[:, num_remain:]
            revert_idx = torch.argsort(shuffle_idx, dim=1)

            remain_padding_mask = torch.ones(remain_idx.shape[0], remain_idx.shape[1]+1).to(device)
            revert_padding_mask = torch.ones(revert_idx.shape[0], revert_idx.shape[1]+1).to(device)

            # Apply mask
            val = torch.gather(val, index=remain_idx.unsqueeze(-1).repeat(1, 1, val.shape[-1]), dim=1)
            val = torch.cat([global_token, val], dim=1)

            result_dict[col] = val
            idx_dict.update({f"{col}_remain_idx":remain_idx, f"{col}_masked_idx":masked_idx, f"{col}_revert_idx":revert_idx})
            padding_mask_dict.update({f"{col}_remain_padding_mask":remain_padding_mask, f"{col}_revert_padding_mask":revert_padding_mask})

        return result_dict, idx_dict, padding_mask_dict

class NlpRemain(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, remain_rto, nlp_cols, device):
        result_dict = {}
        for col in nlp_cols:
            val = data_dict[col]
            
            # Split global token
            global_token = val[:, :1, :]
            val = val[:, 1:, :]

            # Apply remain mask
            remain_idx = idx_dict[f"{col}_remain_idx"].unsqueeze(-1).repeat(1, 1, val.shape[-1])
            val = torch.gather(val, index=remain_idx, dim=1)
            val = torch.cat([global_token, val], dim=1)
            result_dict[col] = val

            # Update padding_mask
            padding_mask_for_global_token = torch.ones(val.shape[0], 1).to(device)

            remain_padding_mask = padding_mask_dict[f"{col}_remain_padding_mask"]
            remain_padding_mask = torch.cat([padding_mask_for_global_token, remain_padding_mask], dim=-1)
            padding_mask_dict[f"{col}_remain_padding_mask"] = remain_padding_mask

            revert_padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]
            revert_padding_mask = torch.cat([padding_mask_for_global_token, revert_padding_mask], dim=-1)
            padding_mask_dict[f"{col}_revert_padding_mask"] = revert_padding_mask
        return result_dict, padding_mask_dict


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation):
        super().__init__()
        self.self_attn = MultiheadBlockSelfAttention(d_model, nhead, dropout)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        
        # Feed forward
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        if activation == "gelu":
            self.activation = torch.nn.GELU()
        
        self.linear_ff1 = torch.nn.Linear(d_model, d_ff)
        self.linear_ff2 = torch.nn.Linear(d_ff, d_model)
        self.dropout_ff1 = torch.nn.Dropout(dropout)
        self.dropout_ff2 = torch.nn.Dropout(dropout)

    def forward(self, src, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask):
        x = src
        attn_output = self._sa_block(self.norm1(x), temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask)
        x = x + attn_output

        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, src, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask):
        x = self.self_attn(src, src, src, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear_ff2(self.dropout_ff1(self.activation(self.linear_ff1(x))))
        return self.dropout_ff2(x)


class TemporalRevert(torch.nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = mask_token
    
    def forward(self, temporal, idx_dict, temporal_cols):
        # Append mask token
        revert_idx = idx_dict["temporal_revert_idx"]
        mask_token = self.mask_token.unsqueeze(0).unsqueeze(1).repeat(temporal.shape[0], temporal.shape[1], revert_idx.shape[-1] - temporal.shape[-2], 1)
        temporal = torch.cat([temporal, mask_token], dim=-2)

        # Apply revert
        temporal = torch.gather(temporal, index=revert_idx.unsqueeze(-1).repeat(1, 1, 1, temporal.shape[-1]), dim=-2)

        # Split to dictionary
        temporal_revert_dict = {}
        for n, col in enumerate(temporal_cols):
            temporal_revert_dict[col] = temporal[:, :, n, :]

        return temporal_revert_dict

class ImgRevert(torch.nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = mask_token
    
    def forward(self, img_dict, idx_dict, img_cols):
        img_revert_dict = {}
        for col in img_cols:
            img_data = img_dict[col]
            
            # Split global token
            global_token = img_data[:, :1, :]
            img_data = img_data[:, 1:, :]

            # Append mask token
            revert_idx = idx_dict[f"{col}_revert_idx"]
            mask_token = self.mask_token.unsqueeze(0).repeat(img_data.shape[0], revert_idx.shape[1]-img_data.shape[-2], 1)
            img_data = torch.cat([img_data, mask_token], dim=-2)

            # Apply revert
            img_data = torch.gather(img_data, index=revert_idx.unsqueeze(-1).repeat(1, 1, img_data.shape[-1]), dim=-2)
            img_data = torch.cat([global_token, img_data], dim=1)

            img_revert_dict[col] = img_data

        return img_revert_dict

class NlpRevert(torch.nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = mask_token
    
    def forward(self, nlp_dict, idx_dict, nlp_cols):
        nlp_revert_dict = {}
        for col in nlp_cols:
            nlp_data = nlp_dict[col]

            # Split global token
            global_token = nlp_data[:, :1, :]
            nlp_data = nlp_data[:, 1:, :]
            
            # Append mask token
            revert_idx = idx_dict[f"{col}_revert_idx"]
            mask_token = self.mask_token.unsqueeze(0).repeat(nlp_data.shape[0], revert_idx.shape[1]-nlp_data.shape[-2], 1)
            nlp_data = torch.cat([nlp_data, mask_token], dim=-2)

            # Apply revert
            nlp_data = torch.gather(nlp_data, index=revert_idx.unsqueeze(-1).repeat(1, 1, nlp_data.shape[-1]), dim=-2)
            nlp_data = torch.cat([global_token, nlp_data], dim=1)

            nlp_revert_dict[col] = nlp_data
       
        return nlp_revert_dict


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
    
    def forward(self, tgt, memory, cross_attn_padding_mask, self_attn_padding_mask):
        x = tgt

        self_attn_output, self_attn_weight = self._sa_block(self.norm2(x), self_attn_padding_mask)
        x = x + self_attn_output

        cross_attn_output, cross_attn_weight = self._ca_block(self.norm1(x.unsqueeze(-2)), memory, memory, cross_attn_padding_mask)
        x = x + cross_attn_output.squeeze()

        x = x + self._ff_block(self.norm3(x))
        return x, self_attn_weight, cross_attn_weight

    def _ca_block(self, query, key, value, padding_mask):
        x, attn_weight = self.cross_attn(query, key, value, padding_mask)
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

        self_attn_output, self_attn_weight = self._sa_block(self.norm2(x), self_attn_padding_mask)
        x = x + self_attn_output

        cross_attn_output, cross_attn_weight = self._ca_block(self.norm1(x), memory, memory, cross_attn_padding_mask)
        x = x + cross_attn_output

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