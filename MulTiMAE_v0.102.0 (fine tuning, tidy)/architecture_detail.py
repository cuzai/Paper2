import torch
from architecture_detail_detail import *
import copy

class Embedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model, global_token):
        super().__init__()
        if col in data_info.modality_info["target"] + data_info.modality_info["temporal"]:
            self.embedding = TemporalEmbedding(col, data_info, label_encoder_dict, d_model, global_token)
        elif col in data_info.modality_info["img"]:
            self.embedding = ImgEmbedding(d_model)
        elif col in data_info.modality_info["nlp"]:
            self.embedding = NlpEmbedding(d_model)
    
    def forward(self, key, val, padding_mask, device):
        return self.embedding(key, val, padding_mask, device)

class PosModEncoding(torch.nn.Module):
    def __init__(self, col, pos_enc, num_modality, modality, d_model):
        super().__init__()
        self.pos_enc = pos_enc
        self.modality = modality[col]
        self.modality_embedding = torch.nn.Embedding(num_modality, d_model)

    def forward(self, key, val, device):
        # Positional encoding
        val = self.pos_enc(val)

        # Modality embedding
        modality = torch.zeros(val.shape[1]).to(torch.int).to(device) + self.modality
        modality = self.modality_embedding(modality)

        return val + modality

class Remain(torch.nn.Module):
    def __init__(self, global_token):
        super().__init__()
        self.temporal_remain = TemporalRemain()
        self.img_remain = ImgRemain(global_token)
        self.nlp_remain = NlpRemain()
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, remain_rto, temporal_cols, img_cols, nlp_cols, device):
        temporal_dict, temporal_idx_dict, temporal_padding_mask_dict = self.temporal_remain(data_dict, padding_mask_dict, remain_rto["temporal"], temporal_cols, device)
        idx_dict.update(temporal_idx_dict)
        padding_mask_dict.update(temporal_padding_mask_dict)

        img_dict, img_idx_dict, img_padding_mask_dict = self.img_remain(data_dict, remain_rto["img"], img_cols, device)
        idx_dict.update(img_idx_dict)
        padding_mask_dict.update(img_padding_mask_dict)

        nlp_dict, nlp_padding_mask_dict = self.nlp_remain(data_dict, idx_dict, padding_mask_dict, remain_rto["nlp"], nlp_cols, device)
        padding_mask_dict.update(nlp_padding_mask_dict)

        return temporal_dict, img_dict, nlp_dict, idx_dict, padding_mask_dict

class Encoder(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers, to_decoder_dim):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(EncoderLayer(d_model, nhead, d_ff, dropout, activation)) for _ in range(num_layers)])
        self.to_decoder_dim = to_decoder_dim
    
    def forward(self, temporal_dict, img_dict, nlp_dict, padding_mask_dict, img_cols, nlp_cols, device):
        src, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask, img_idx_li, nlp_idx_li = self.get_src(temporal_dict, img_dict, nlp_dict, padding_mask_dict, img_cols, nlp_cols, device)
        for mod in self.encoder_layers:
            src = mod(src, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask)
        
        encoded = self.to_decoder_dim(src)
        temporal_dict, img_dict, nlp_dict = self.undo_src(encoded, temporal_shape, temporal_idx, img_idx_li, nlp_idx_li, img_cols, nlp_cols)

        return temporal_dict, img_dict, nlp_dict
    
    def get_src(self, temporal_dict, img_dict, nlp_dict, padding_mask_dict, img_cols, nlp_cols, device):
        # Temporal
        temporal = temporal_dict["temporal"]
        temporal_shape = temporal.shape
        batch_size, seq_len, _, d_model = temporal_shape
        temporal = temporal.view(batch_size, -1, d_model)
        temporal_padding_mask = padding_mask_dict["temporal_remain_padding_mask"]
        temporal_idx = torch.arange(0, temporal.shape[1]).to(device)

        # Static
        static_start_idx = temporal_idx[-1]+1
        idx = static_start_idx.clone()

        ### Img
        img_li, img_padding_mask_li, img_idx_li = [], [], []
        for col in img_cols:
            img_data = img_dict[col]
            img_padding_mask = padding_mask_dict[f"{col}_remain_padding_mask"]
            assert img_data.shape[1] == img_padding_mask.shape[-1], f"{img_data.shape}, {img_padding_mask.shape}"

            img_li.append(img_data)
            img_padding_mask_li.append(img_padding_mask)
            img_idx_li.append(torch.arange(idx, idx+img_data.shape[1]).to(device))
            idx += img_data.shape[1]
        
        img = torch.cat(img_li, dim=1)
        img_padding_mask = torch.cat(img_padding_mask_li, dim=1)

        ### Nlp
        nlp_li, nlp_padding_mask_li, nlp_idx_li = [], [], []
        for col in nlp_cols:
            nlp_data = nlp_dict[col]
            nlp_padding_mask = padding_mask_dict[f"{col}_remain_padding_mask"]
            assert nlp_data.shape[1] == nlp_padding_mask.shape[-1], f"{nlp_data.shape}, {nlp_padding_mask.shape}"

            nlp_li.append(nlp_data)
            nlp_padding_mask_li.append(nlp_padding_mask)
            nlp_idx_li.append(torch.arange(idx, idx+nlp_data.shape[1]).to(device))
            idx += nlp_data.shape[1]

        nlp = torch.cat(nlp_li, dim=1) if len(nlp_li)>0 else torch.tensor([]).to(device)
        nlp_padding_mask = torch.cat(nlp_padding_mask_li, dim=1) if len(nlp_li)>0 else torch.tensor([]).to(torch.int64).to(device)

        ### Src
        static = torch.cat([img, nlp], dim=1)
        static_padding_mask = torch.cat([img_padding_mask, nlp_padding_mask], dim=1)

        img_idx = torch.cat(img_idx_li)
        nlp_idx = torch.cat(nlp_idx_li) if len(nlp_idx_li)>0 else torch.tensor([]).to(torch.int64).to(device)
        static_idx = torch.cat([img_idx, nlp_idx])

        # Src
        src = torch.cat([temporal, static], dim=1)
        
        return src, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask, img_idx_li, nlp_idx_li
    
    def undo_src(self, src, temporal_shape, temporal_idx, img_idx_li, nlp_idx_li, img_cols, nlp_cols):
        # Temporal
        temporal_dict = {}
        temporal_idx = temporal_idx.unsqueeze(0).unsqueeze(-1).repeat(src.shape[0], 1, src.shape[-1])
        temporal = torch.gather(src, index=temporal_idx, dim=1).view(temporal_shape[0], temporal_shape[1], temporal_shape[2], -1)
        temporal_dict["temporal"] = temporal

        # Img
        img_dict = {}
        for col, idx in zip(img_cols, img_idx_li):
            idx = idx.unsqueeze(0).unsqueeze(-1).repeat(src.shape[0], 1, src.shape[-1])
            img_dict[col] = torch.gather(src, index=idx, dim=1)

        # Nlp
        nlp_dict = {}
        for col, idx in zip(nlp_cols, nlp_idx_li):
            idx = idx.unsqueeze(0).unsqueeze(-1).repeat(src.shape[0], 1, src.shape[-1])
            nlp_dict[col] = torch.gather(src, index=idx, dim=1)
        
        return temporal_dict, img_dict, nlp_dict

class Revert(torch.nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.temporal_revert = TemporalRevert(mask_token)
        self.img_revert = ImgRevert(mask_token)
        self.nlp_revert = NlpRevert(mask_token)
    
    def forward(self, temporal_dict, img_dict, nlp_dict, idx_dict, temporal_cols, img_cols, nlp_cols):
        temporal_dict = self.temporal_revert(temporal_dict["temporal"], idx_dict, temporal_cols)
        img_dict = self.img_revert(img_dict, idx_dict, img_cols)
        nlp_dict = self.nlp_revert(nlp_dict, idx_dict, nlp_cols)

        revert_dict = {}
        revert_dict.update(temporal_dict)
        revert_dict.update(img_dict)
        revert_dict.update(nlp_dict)
        
        return revert_dict

class Decoder(torch.nn.Module):
    def __init__(self, col, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.temporal_decoder_layers = torch.nn.ModuleList([copy.deepcopy(TemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation)) for _ in range(num_layers)])
        self.non_temporal_decoder_layers = torch.nn.ModuleList([copy.deepcopy(NonTemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation)) for _ in range(num_layers)])
    
    def forward(self, key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device):
        # Temporal
        if key in temporal_cols:
            tgt = val
            memory, cross_attn_padding_mask, self_attn_padding_mask = self.get_temporal_tgt_memory(key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device)
            decoder_layers = self.temporal_decoder_layers
        elif key in img_cols + nlp_cols:
            tgt = val
            memory, cross_attn_padding_mask, self_attn_padding_mask = self.get_non_temporal_tgt_memory(key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device)
            decoder_layers = self.non_temporal_decoder_layers

        for mod in decoder_layers:
            tgt, cross_attn_weight, self_attn_weight = mod(tgt, memory, cross_attn_padding_mask, self_attn_padding_mask)
        
        return tgt, cross_attn_weight, self_attn_weight

    def get_temporal_tgt_memory(self, key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device):
        memory_li = []

        # Temporal memory
        for col in temporal_cols:
            if col == key:
                continue
            memory_li.append(data_dict[col])
        memory = torch.stack(memory_li, dim=-2)
        cross_attn_padding_mask = torch.ones(memory.shape[:-1]).to(device)
        
        # Static memory
        for col in img_cols + nlp_cols:
            static_data = data_dict[col].unsqueeze(1).repeat(1, memory.shape[1], 1, 1)
            static_padding_msak = padding_mask_dict[f"{col}_revert_padding_mask"].unsqueeze(1).repeat(1, memory.shape[1], 1)

            memory = torch.cat([memory, static_data], dim=-2)
            cross_attn_padding_mask = torch.cat([cross_attn_padding_mask, static_padding_msak], dim=-1)
        
        # Self attn padding mask
        self_attn_padding_mask = padding_mask_dict["temporal_revert_padding_mask"].squeeze()
        self_attn_padding_mask = torch.where(self_attn_padding_mask==1, 0, -torch.inf)
        cross_attn_padding_mask = torch.where(cross_attn_padding_mask==1, 0, -torch.inf)

        return memory, cross_attn_padding_mask, self_attn_padding_mask

    def get_non_temporal_tgt_memory(self, key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device):
        memory_li, cross_attn_padding_mask_li = [], []
        
        # Temporal memory
        for col in temporal_cols:
            temporal_data = data_dict[col]
            temporal_padding_mask = padding_mask_dict[f"temporal_revert_padding_mask"].squeeze()
            
            memory_li.append(temporal_data)
            cross_attn_padding_mask_li.append(temporal_padding_mask)
        
        memory = torch.cat(memory_li, dim=1)
        cross_attn_padding_mask = torch.cat(cross_attn_padding_mask_li, dim=-1)

        # Img memory or Nlp memory
        for col in img_cols + nlp_cols:
            if key == col: continue
            static_data = data_dict[col]
            static_padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]

            memory = torch.cat([memory, static_data], dim=1)
            cross_attn_padding_mask = torch.cat([cross_attn_padding_mask, static_padding_mask], dim=-1)
        
        # Self attn padding mask
        self_attn_padding_mask = padding_mask_dict[f"{key}_revert_padding_mask"].squeeze()
        self_attn_padding_mask = torch.where(self_attn_padding_mask==1, 0, -torch.inf)
        cross_attn_padding_mask = torch.where(cross_attn_padding_mask==1, 0, -torch.inf)

        return memory, cross_attn_padding_mask, self_attn_padding_mask

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

1==1