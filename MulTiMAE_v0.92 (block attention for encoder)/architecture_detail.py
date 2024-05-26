import copy
import torch
from architecture_detail_detail import *
from utils import *

class Embedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.modality_info["target"] + data_info.modality_info["temporal"]:
            self.embedding = TemporalEmbedding(col, data_info, label_encoder_dict, d_model)
        elif col in data_info.modality_info["img"]:
            self.embedding = ImgEmbedding(d_model)
        elif col in data_info.modality_info["nlp"]:
            self.embedding = NlpEmbedding(d_model)
    
    def forward(self, key, val, padding_mask, device):
        return self.embedding(key, val, padding_mask, device)

class Remain(torch.nn.Module):
    def __init__(self, col, data_info, d_model, dropout):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.global_token = torch.nn.Parameter(torch.rand(1, d_model))
        
        if col in data_info.modality_info["target"] + data_info.modality_info["temporal"]:
            self.remain = TemporalRemain()
        elif col in data_info.modality_info["img"]:
            self.remain = ImgRemain()
        elif col in data_info.modality_info["nlp"]:
            self.remain = NlpRemain()
    
    def forward(self, key, val, idx_dict, remain_rto, device):
        return self.remain(key, val, self.global_token, self.pos_enc, idx_dict, remain_rto, device)

class ModalityEmbedding(torch.nn.Module):
    def __init__(self, col, num_modality, modality, d_model):
        super().__init__()
        self.modality = modality[col]
        self.modality_embedding = torch.nn.Embedding(num_modality, d_model)
    
    def forward(self, key, val, device):
        modality = torch.zeros(val.shape[1]).to(torch.int).to(device) + self.modality
        modality = self.modality_embedding(modality)
        
        return val + modality

class Encoder(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation=activation, norm_first=True), num_layers)
    
    def forward(self, data_dict, padding_mask_dict, total_cols, device):
        # Concat data and padding_mask
        data_concat_li, padding_mask_concat_li = [], []
        for col in total_cols:
            data = data_dict[col]
            
            padding_mask = padding_mask_dict[f"{col}_remain_padding_mask"]
            padding_mask_for_global_token = torch.ones(data.shape[0], 1).to(device)
            new_padding_mask = torch.cat([padding_mask_for_global_token, padding_mask], dim=1)

            data_concat_li.append(data)
            padding_mask_concat_li.append(new_padding_mask)
        
        data_concat = torch.cat(data_concat_li, dim=1)
        padding_mask_concat = torch.cat(padding_mask_concat_li, dim=1)
        padding_mask_concat = torch.where(padding_mask_concat==1, 0, -torch.inf)
        assert data_concat.shape[:-1] == padding_mask_concat.shape

        encoding = self.encoder(data_concat, src_key_padding_mask=padding_mask_concat)

        return encoding

class Split(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data_to_split, data_for_length, total_cols):
        split_dict = {}
        start_idx = 0

        for col in total_cols:
            length = data_for_length[col].shape[1]
            split_dict[col] = data_to_split[:, start_idx:start_idx+length, :]
            start_idx += length
        assert start_idx == data_to_split.shape[1]

        return split_dict

class Revert(torch.nn.Module):
    def __init__(self, col, d_model, dropout):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.mask_token = torch.nn.Parameter(torch.rand(1, d_model))
        self.revert = DynamicRevert()
        
    def forward(self, key, val, idx_dict, padding_mask_dict):
        return self.revert(key, val, self.mask_token, self.pos_enc, idx_dict, padding_mask_dict)


class TemporalDecoder(torch.nn.Module):
    def __init__(self, col, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(TemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation)) for _ in range(num_layers)])
    
    def forward(self, key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device):
        tgt = val

        # Get memory
        memory_li = []
        ### Temporal memory
        for col in temporal_cols:
            if col == key: continue
            memory_li.append(data_dict[col])
        memory = torch.stack(memory_li, dim=-2)
        
        ### Img memory & Nlp memory
        for col in img_cols + nlp_cols:
            img_data = data_dict[col].unsqueeze(1).repeat(1, memory.shape[1], 1, 1)
            memory = torch.cat([memory, img_data], dim=-2)
        
        # Obtain self_attn_padding_mask
        self_attn_padding_mask = padding_mask_dict[f"{key}_revert_padding_mask"]
        padding_mask_for_global_token = torch.ones(val.shape[0], 1).to(device)
        self_attn_padding_mask = torch.cat([padding_mask_for_global_token, self_attn_padding_mask], dim=-1)
        self_attn_padding_mask = torch.where(self_attn_padding_mask==1, 0, -torch.inf)

        # Apply decoder
        for mod in self.layers:
            tgt, self_attn_weight, cross_attn_weight = mod(tgt, memory, self_attn_padding_mask)
            
        return tgt, self_attn_weight, cross_attn_weight

class NonTemporalDecoder(torch.nn.Module):
    def __init__(self, col, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(NonTemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation)) for _ in range(num_layers)])
    
    def forward(self, key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device):
        tgt = val

        # Get memory and cross_attn_padding_mask
        memory_li, cross_attn_padding_mask_li = [], []
        ### Temporal memory & Nlp memory
        for col in temporal_cols + nlp_cols:
            temporal_data = data_dict[col]
            padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]
            padding_mask_for_global_token = torch.ones(val.shape[0], 1).to(device)
            new_padding_mask = torch.cat([padding_mask_for_global_token, padding_mask], dim=-1)
            
            memory_li.append(temporal_data)
            cross_attn_padding_mask_li.append(new_padding_mask)
        
        memory = torch.cat(memory_li, dim=1)
        cross_attn_padding_mask = torch.cat(cross_attn_padding_mask_li, dim=1)
        cross_attn_padding_mask = torch.where(cross_attn_padding_mask == 1, 0, -torch.inf)

        # Obtain self_attn_padding_mask
        self_attn_padding_mask = padding_mask_dict[f"{key}_revert_padding_mask"]
        padding_mask_for_global_token = torch.ones(val.shape[0], 1).to(device)
        self_attn_padding_mask = torch.cat([padding_mask_for_global_token, self_attn_padding_mask], dim=-1)
        self_attn_padding_mask = torch.where(self_attn_padding_mask==1, 0, -torch.inf)

        # Apply decoder
        for mod in self.layers:
            tgt, self_attn_weight, cross_attn_weight = mod(tgt, memory, cross_attn_padding_mask, self_attn_padding_mask)
            
        return tgt, self_attn_weight, cross_attn_weight


class TemporalOutput(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.processing_info["scaling_cols"]:
            self.output = torch.nn.Sequential(
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, 1))
        
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].get_num_cls()
            self.output = torch.nn.Sequential(
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, num_cls))

    
    def forward(self, key, val):
        return self.output(val)

class ImgOutput(torch.nn.Module):
    def __init__(self, col, d_model, patch_size=16):
        super().__init__()
        self.output = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Linear(d_model, 3*patch_size*patch_size))
    
    def forward(self, key, val):
        return self.output(val)
        
class NlpOutput(torch.nn.Module):
    def __init__(self, col, d_model, num_vocab=30522):
        super().__init__()
        self.output = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Linear(d_model, num_vocab))
    
    def forward(self, key, val):
        return self.output(val)
1==1