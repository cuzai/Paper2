import torch
from transformers import ViTModel, BertModel

from architecture.shared_module import *

class TemporalEmbedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.processing_info["scaling_cols"]:
            self.embedding = torch.nn.Linear(1, d_model)
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].get_num_cls()
            self.embedding = torch.nn.Embedding(num_cls, d_model)
    
    def forward(self, data):
        return self.embedding(data)
    
class ImgEmbedding(torch.nn.Module):
    def __init__(self, global_token, patch_size, d_model):
        super().__init__()
        
        # if self.is_from_pretrained:
        if True:
            self.img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.downsize_linear = torch.nn.Linear(768, d_model)
        else:
            self.patch_size = patch_size
            self.linear = torch.nn.Linear(self.patch_size**2 * 3, d_model)
            self.global_token = torch.nn.Parameter(torch.rand(1, d_model))
            self.norm = torch.nn.LayerNorm(d_model)

            self.conv = torch.nn.Conv2d(3, d_model, self.patch_size, self.patch_size)
    
    def forward(self, data):
        # if self.is_from_pretrained:
        if True:
            embedding = self.img_model(data).last_hidden_state
            embedding = self.downsize_linear(embedding)
        else:
            # patches = self.patchify(data, self.patch_size)
            # embedding = self.linear(patches)
            embedding = self.conv(data)
            embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], -1).permute(0,2,1)

            # Append global token
            batch_size, seq_len, d_model = embedding.shape
            global_token = self.global_token.unsqueeze(0).repeat(batch_size, 1, 1)
            embedding = torch.cat([global_token, embedding], dim=1)
            embedding = self.norm(embedding)
        
        return embedding

    def patchify(self, imgs, patch_size):
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

class NlpEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # if self.is_from_pretrained:
        if True:
            self.nlp_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
            self.downsize_linear = torch.nn.Linear(768, d_model)
        else:
            self.embedding = torch.nn.Embedding(30522, d_model)
  
    def forward(self, key, val, padding_mask_dict, device):
        if True:
            # Make necessary dict (token_type_ids, attention mask)
            token_type_ids = torch.zeros(val.shape).to(torch.int).to(device)
            attention_mask = padding_mask_dict[f"{key}_revert_padding_mask"]

            # Embed data
            inputs = {"input_ids":val, "token_type_ids":token_type_ids, "attention_mask":attention_mask}
            embedding = self.nlp_model(**inputs).last_hidden_state
            embedding = self.downsize_linear(embedding)
        else:
            embedding = self.embedding(val)
        
        return embedding

class Embedding(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict, d_model, temporal_cols, img_cols, nlp_cols, patch_size):
        super().__init__()
        self.data_info, self.temporal_cols, self.img_cols, self.nlp_cols = data_info, temporal_cols, img_cols, nlp_cols
        self.global_token = torch.nn.Parameter(torch.rand(1, d_model))

        self.embedding_dict = torch.nn.ModuleDict()
        for col in self.temporal_cols:
            self.embedding_dict[col] = TemporalEmbedding(col, data_info, label_encoder_dict, d_model)
        for col in self.img_cols:
            self.embedding_dict[col] = ImgEmbedding(self.global_token, patch_size, d_model)
        for col in self.nlp_cols:
            self.embedding_dict[col] = NlpEmbedding(d_model)
        
    def forward(self, data_dict, padding_mask_dict, device):
        result_dict = {}

        # Generate temporal global token sequence
        target_data_shape = data_dict[self.data_info.modality_info["target"][0]].shape
        batch_size, seq_len, d_model = target_data_shape
        global_token_seq = self.global_token.unsqueeze(0).repeat(batch_size, seq_len, 1)
        result_dict["global"] = global_token_seq

        # Embed data
        for col in self.temporal_cols + self.img_cols:
            result_dict[col] = self.embedding_dict[col](data_dict[col])
        
        for col in self.nlp_cols:
            result_dict[col] = self.embedding_dict[col](col, data_dict[col], padding_mask_dict, device)
        
        return result_dict


class TemporalRemain(torch.nn.Module):
    def __init__(self, temporal_cols):
        super().__init__()
        self.temporal_cols = temporal_cols
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, remain_rto, device):
        # Make temporal data to a block matrix
        temporal_data_li = [val for key, val in data_dict.items() if key in self.temporal_cols]
        temporal_block_data = torch.stack(temporal_data_li, dim=-2)

        # Obtain block revert padding mask
        total_num_modality = temporal_block_data.shape[-2]
        temporal_block_revert_padding_mask = padding_mask_dict["temporal_padding_mask"].unsqueeze(-1).repeat(1, 1, total_num_modality)
        # temporal_block_revert_padding_mask = torch.ones(padding_mask_dict["temporal_padding_mask"].shape).unsqueeze(-1).repeat(1, 1, total_num_modality).to(device)
        temporal_block_revert_padding_mask[:, :, 1] = padding_mask_dict["target_fcst_mask"]
        
        # Split global token sequence
        ### Data
        global_token_data = temporal_block_data[:, :, :1, :]
        valid_block_data = temporal_block_data[:, :, 1:, :]
        ### Padding mask
        global_token_padding_mask = temporal_block_revert_padding_mask[:, :, :1]
        valid_block_padding_mask = temporal_block_revert_padding_mask[:, :, 1:]

        # Obtain remain idx
        valid_num_modality = valid_block_data.shape[-2]
        num_remain = int(valid_num_modality * remain_rto)
        noise = torch.rand(valid_block_data.shape[:-1]).to(device)
        shuffle_idx = torch.argsort(noise, dim=-1)

        remain_idx = shuffle_idx[:, :, :num_remain]
        masked_idx = shuffle_idx[:, :, num_remain:]
        revert_idx = torch.argsort(shuffle_idx, dim=-1)

        # Apply mask
        ### Data
        remain_idx_ = remain_idx.unsqueeze(-1).repeat(1, 1, 1, valid_block_data.shape[-1])
        valid_block_remain_data = torch.gather(valid_block_data, index=remain_idx_, dim=-2)
        temporal_block_remain_data = torch.cat([global_token_data, valid_block_remain_data], dim=-2)
        ### Padding mask
        valid_block_padding_mask = torch.gather(valid_block_padding_mask, index=remain_idx, dim=-1)
        temporal_block_remain_padding_mask = torch.cat([global_token_padding_mask, valid_block_padding_mask], dim=-1)
        
        # Update dicts
        idx_dict.update({"temporal_block_remain_idx":remain_idx, "temporal_block_masked_idx":masked_idx, "temporal_block_revert_idx":revert_idx})
        padding_mask_dict.update({"temporal_block_remain_padding_mask":temporal_block_remain_padding_mask, "temporal_block_revert_padding_mask":temporal_block_revert_padding_mask})

        return temporal_block_remain_data, idx_dict, padding_mask_dict

class ImgRemain(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, col, data, idx_dict, padding_mask_dict, remain_rto, device):
        # Obtain revert padding mask
        revert_padding_mask = torch.ones(data.shape[:-1]).to(device)

        # Split global token
        ### Data
        global_token = data[:, :1, :]
        valid_data = data[:, 1:, :]
        ### Padding mask
        global_padding_mask = revert_padding_mask[:, :1]
        valid_padding_mask = revert_padding_mask[:, 1:]

        # Obtain remain idx
        seq_len = valid_data.shape[1]
        num_remain = int(seq_len * remain_rto)
        noise = torch.rand(valid_data.shape[:-1]).to(device)
        shuffle_idx = torch.argsort(noise, dim=1)

        remain_idx = shuffle_idx[:, :num_remain]
        masked_idx = shuffle_idx[:, num_remain:]
        revert_idx = torch.argsort(shuffle_idx, dim=1)

        # Apply mask
        ### Data
        remain_idx_ = remain_idx.unsqueeze(-1).repeat(1, 1, valid_data.shape[-1])
        valid_remain_data = torch.gather(valid_data, index=remain_idx_, dim=1)
        total_remain_data = torch.cat([global_token, valid_remain_data], dim=1)
        ### Padding mask
        valid_padding_mask = torch.gather(valid_padding_mask, index=remain_idx, dim=1)
        total_remain_padding_mask = torch.cat([global_padding_mask, valid_padding_mask], dim=1)
        
        # Update dicts
        idx_dict.update({f"{col}_remain_idx":remain_idx, f"{col}_masked_idx":masked_idx, f"{col}_revert_idx":revert_idx})
        padding_mask_dict.update({f"{col}_remain_padding_mask":total_remain_padding_mask, f"{col}_revert_padding_mask":revert_padding_mask})

        return total_remain_data, idx_dict, padding_mask_dict

class NlpRemain(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, col, data, idx_dict):
        # Split global token
        global_token = data[:, :1, :]
        valid_data = data[:, 1:, :]

        # Apply mask
        remain_idx = idx_dict[f"{col}_remain_idx"]
        remain_idx_ = remain_idx.unsqueeze(-1).repeat(1, 1, valid_data.shape[-1])
        valid_remain_data = torch.gather(valid_data, index=remain_idx_, dim=1)
        total_remain_data = torch.cat([global_token, valid_remain_data], dim=1)

        return total_remain_data

class Remain(torch.nn.Module):
    def __init__(self, temporal_cols, img_cols, nlp_cols):
        super().__init__()
        self.img_cols, self.nlp_cols = img_cols, nlp_cols

        # Temporal
        self.temporal_remain = TemporalRemain(temporal_cols)

        # Img
        self.img_remain_dict = torch.nn.ModuleDict()
        for col in self.img_cols:
            self.img_remain_dict[col] = ImgRemain()
        
        # Nlp
        self.nlp_remain_dict = torch.nn.ModuleDict()
        for col in self.nlp_cols:
            self.nlp_remain_dict[col] = NlpRemain()
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, remain_rto, device):
        # Temporal
        temporal_block_remain, idx_dict, padding_mask_dict = self.temporal_remain(data_dict, idx_dict, padding_mask_dict, remain_rto["temporal"], device)
        
        # Img
        img_remain_dict = {}
        for col in self.img_cols:
            img_remain, idx_dict, padding_mask_dict = self.img_remain_dict[col](col, data_dict[col], idx_dict, padding_mask_dict, remain_rto["img"], device)
            img_remain_dict[col] = img_remain
            assert img_remain.shape[:-1] == padding_mask_dict[f"{col}_remain_padding_mask"].shape, f'{img_remain.shape[:-1]}, {padding_mask_dict[f"{col}_remain_padding_mask"].shape}'
        
        # Nlp
        nlp_remain_dict = {}
        for col in self.nlp_cols:
            nlp_remain = self.nlp_remain_dict[col](col, data_dict[col], idx_dict)
            nlp_remain_dict[col] = nlp_remain
            assert nlp_remain.shape[:-1] == padding_mask_dict[f"{col}_remain_padding_mask"].shape, f'{nlp_remain.shape[:-1]}, {padding_mask_dict[f"{col}_remain_padding_mask"].shape}'

        
        return temporal_block_remain, img_remain_dict, nlp_remain_dict, idx_dict, padding_mask_dict
            

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation):
        super().__init__()
        self.self_attn = MultiheadBlockAttention(d_model, nhead)

        self.sa_norm = torch.nn.LayerNorm(d_model)
        self.sa_dropout = torch.nn.Dropout(dropout)

        # Feed forward
        if activation == "relu": self.activation = torch.nn.ReLU()
        if activation == "gelu": self.activation = torch.nn.GELU()
        self.ff_norm = torch.nn.LayerNorm(d_model)
        self.ff_linear1 = torch.nn.Linear(d_model, d_ff)
        self.ff_linear2 = torch.nn.Linear(d_ff, d_model)
        self.ff_dropout1 = torch.nn.Dropout(dropout)
        self.ff_dropout2 = torch.nn.Dropout(dropout)

    def forward(self, src_dict, src_key_padding_mask_dict):
        x = src_dict

        attn_output_dict, attn_weight_dict = self._sa_block(self.flatten_apply(x, module=self.sa_norm), src_key_padding_mask_dict)
        x = self.sum_dicts(x, attn_output_dict)
        
        ff_output_dict = self._ff_block(self.flatten_apply(x, module=self.ff_norm))
        x = self.sum_dicts(x, ff_output_dict)
        
        return x, attn_weight_dict
    
    def flatten_apply(self, data_dict, module):
        # Flaten
        flatten_li = []
        temporal_length = 0
        for key, val in data_dict.items():
            if key == "temporal":
                temporal_block_shape = val.shape
                temporal_flattened = val.view(temporal_block_shape[0], -1, temporal_block_shape[-1])
                temporal_length = temporal_flattened.shape[-2]
                flatten_li.append(temporal_flattened)
                
            elif key == "static":
                flatten_li.append(val)
            
        flattened_data = torch.cat(flatten_li, dim=1)

        # Apply module
        moduled_data = module(flattened_data)
        
        # Un-flatten
        result_dict = {}
        for key, val in data_dict.items():
            if key == "temporal":
                result_dict[key] = moduled_data[:, :temporal_length, :].view(temporal_block_shape)
            elif key == "static":
                result_dict[key] = moduled_data[:, temporal_length:, :]
        
        return result_dict

    def sum_dicts(self, dict1, dict2):
        result_dict = {}
        for (key1, val1), (key2, val2) in zip(dict1.items(), dict2.items()):
            assert key1 == key2
            result_dict[key1] = val1 + val2
        
        return result_dict


    def _sa_block(self, src_dict, src_key_padding_mask_dict):
        attn_output_dict, attn_weight_dict = self.self_attn(src_dict, src_dict, src_dict, src_key_padding_mask_dict)
        return self.flatten_apply(attn_output_dict, module=self.sa_dropout), attn_weight_dict
    
    def _ff_block(self, data_dict):
        result_dict = {}
        for key, val in data_dict.items():
            x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(val))))
            result_dict[key] = self.ff_dropout2(x)
        
        return result_dict

class Encoder(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, temporal_block, img_dict, nlp_dict, padding_mask_dict, mode="remain"):
        src_dict, src_key_padding_mask_dict = self.get_src(temporal_block, img_dict, nlp_dict, padding_mask_dict, mode)

        # Encode
        x = src_dict
        for mod in self.encoder_layers:
            x, attn_weight_dict = mod(x, src_key_padding_mask_dict)
        
        # Un-block data
        temporal_encoding_block, img_encoding_dict, nlp_encoding_dict = self.undo_src(x, img_dict, nlp_dict)
        
        return temporal_encoding_block, img_encoding_dict, nlp_encoding_dict, attn_weight_dict
    
    def get_src(self, temporal_block, img_dict, nlp_dict, padding_mask_dict, mode):
        src_dict, src_key_padding_mask_dict = {}, {}
        # Temporal
        ### Data
        temporal_src = temporal_block
        src_dict.update({"temporal":temporal_src})
        ### Padding mask
        temporal_padding_mask = padding_mask_dict[f"temporal_block_{mode}_padding_mask"]
        src_key_padding_mask_dict.update({"temporal":temporal_padding_mask})

        # Static
        if len(img_dict) or len(nlp_dict):
            ### Data
            img_data_li = list(img_dict.values())
            nlp_data_li = list(nlp_dict.values())
            static_src = torch.cat(img_data_li + nlp_data_li, dim=1)
            src_dict.update({"static":static_src})
            ### Padding mask
            img_padding_mask_li = [padding_mask_dict[f"{col}_{mode}_padding_mask"] for col in img_dict.keys()]
            nlp_padding_mask_li = [padding_mask_dict[f"{col}_{mode}_padding_mask"] for col in nlp_dict.keys()]
            static_padding_mask = torch.cat(img_padding_mask_li + nlp_padding_mask_li, dim=1)
            src_key_padding_mask_dict.update({"static":static_padding_mask})
        
        return src_dict, src_key_padding_mask_dict

    def undo_src(self, encoding_dict, img_dict, nlp_dict):
        for key, val in encoding_dict.items():
            # Temporal
            if key == "temporal":
                temporal_encoding_block = val
            
            # Static
            img_encoding_dict, nlp_encoding_dict = {}, {}
            if key == "static":
                static_data = val
                ### Image
                idx = 0
                for k, v in img_dict.items():
                    length = v.shape[1]
                    img_encoding_dict[k] = static_data[:, idx:idx+length, :]
                    idx += length
                
                ### Nlp
                for k, v in nlp_dict.items():
                    length = v.shape[1]
                    nlp_encoding_dict[k] = static_data[:, idx:idx+length, :]
                    idx += length
                
                assert idx == static_data.shape[1]
        
        return temporal_encoding_block, img_encoding_dict, nlp_encoding_dict


class TemporalRevert(torch.nn.Module):
    def __init__(self, mask_token, temporal_cols):
        super().__init__()
        self.mask_token = mask_token
        self.temporal_cols = temporal_cols
    
    def forward(self, temporal_block, idx_dict, padding_mask_dict):
        # Split global token seq
        global_seq = temporal_block[:, :, :1, :]
        valid_seq = temporal_block[:, :, 1:, :]

        # Append mask token
        batch_size, seq_len, num_modality, d_model = valid_seq.shape
        revert_idx = idx_dict["temporal_block_revert_idx"]
        mask_token = self.mask_token.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, revert_idx.shape[-1]-num_modality, 1)
        valid_seq = torch.cat([valid_seq, mask_token], dim=-2)
        # valid_seq = torch.where(padding_mask_dict["temporal_block_revert_padding_mask"][:, :, 1:].unsqueeze(-1).repeat(1, 1, 1, valid_seq.shape[-1])==1, valid_seq, self.mask_token)

        # Apply revert
        revert_idx_ = revert_idx.unsqueeze(-1).repeat(1, 1, 1, d_model)
        reverted_valid_seq = torch.gather(valid_seq, index=revert_idx_, dim=-2)
        revert_seq = torch.cat([global_seq, reverted_valid_seq], dim=-2)
        
        # Split to dictionary
        temporal_revert_dict = {}
        for n, col in enumerate(self.temporal_cols):
            temporal_revert_dict[col] = revert_seq[:, :, n, :]
        assert len(temporal_revert_dict) == revert_seq.shape[-2], f"{len(temporal_revert_dict)}, {revert_seq.shape}"

        return temporal_revert_dict

class NonTemporalRevert(torch.nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = mask_token
    
    def forward(self, col, data, idx_dict):
        # Split global token 
        global_token = data[:, :1, :]
        valid_data = data[:, 1:, :]

        # Append mask token
        batch_size, seq_len, d_model = valid_data.shape
        revert_idx = idx_dict[f"{col}_revert_idx"]
        mask_token = self.mask_token.unsqueeze(0).repeat(batch_size, revert_idx.shape[-1]-seq_len, 1)
        valid_data = torch.cat([valid_data, mask_token], dim=1)
        assert valid_data.shape[:-1] == revert_idx.shape

        # Apply revert
        revert_idx_ = revert_idx.unsqueeze(-1).repeat(1, 1, d_model)
        reverted_valid_data = torch.gather(valid_data, index=revert_idx_, dim=1)
        revert_data = torch.cat([global_token, reverted_valid_data], dim=1)
        
        return revert_data

class Revert(torch.nn.Module):
    def __init__(self, d_model, temporal_cols, img_cols, nlp_cols):
        super().__init__()
        
        mask_token = torch.nn.Parameter(torch.rand(1, d_model))

        # Temporal
        self.temporal_revert = TemporalRevert(mask_token, temporal_cols)
        
        # Img, Nlp
        self.non_temporal_revert_dict = torch.nn.ModuleDict()
        for col in img_cols + nlp_cols:
            self.non_temporal_revert_dict[col] = NonTemporalRevert(mask_token)
    
    def forward(self, temporal_block_encoding, img_encoding_dict, nlp_encoding_dict, idx_dict, padding_mask_dict):
        result_dict = {}

        # Temporal
        temporal_revert_dict = self.temporal_revert(temporal_block_encoding, idx_dict, padding_mask_dict)
        
        # Img
        img_revert_dict = {}
        for key, val in img_encoding_dict.items():
            img_revert_dict[key] = self.non_temporal_revert_dict[key](key, val, idx_dict)
        
        # Nlp
        nlp_revert_dict = {}
        for key, val in nlp_encoding_dict.items():
            nlp_revert_dict[key] = self.non_temporal_revert_dict[key](key, val, idx_dict)
        
        # Update dict
        result_dict.update(**temporal_revert_dict, **img_revert_dict, **nlp_revert_dict)

        return result_dict

1==1