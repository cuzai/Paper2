import copy
import torch
from architecture.shared_module import *
from collections import defaultdict

def get_indices(data_shape, num_remain):
    # Get indices
    noise = torch.rand(data_shape)
    shuffle_idx = torch.argsort(noise, dim=-1)
    
    remain_idx = shuffle_idx[..., :num_remain]
    masked_idx = shuffle_idx[..., num_remain:]
    revert_idx = torch.argsort(shuffle_idx, dim=-1)

    return remain_idx, masked_idx, revert_idx


class Embedding(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, d_model):
        super().__init__()
        self.target_col = config.target_col[0]
        self.temporal_cols, self.img_cols, self.nlp_cols = config.temporal_cols, config.img_cols, config.nlp_cols
        self.global_token = torch.nn.Parameter(torch.rand(1, d_model))

        self.embedding_dict = torch.nn.ModuleDict()
        for col in config.temporal_cols + config.img_cols + config.nlp_cols:
            # Temporal
            if col in config.temporal_cols and col in config.scaling_cols:
                self.embedding_dict[col] = torch.nn.Linear(1, d_model)
            elif col in config.temporal_cols and col in config.embedding_cols:
                num_cls = label_encoder_dict[col].get_num_cls()
                self.embedding_dict[col] = torch.nn.Embedding(num_cls, d_model)
            # Img
            elif col in config.img_cols:
                self.embedding_dict[col] = PatchEmbed(224, config.patch_size, 3, d_model)
            # Nlp
            elif col in config.nlp_cols:
                self.embedding_dict[col] = torch.nn.Embedding(30522, d_model)

    def forward(self, data_dict):
        result_dict = {}

        # Temporal global token
        batch_size, seq_len = data_dict[self.target_col].shape[:-1]
        result_dict["global"] = self.global_token.unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Embed others
        for col in self.temporal_cols + self.img_cols + self.nlp_cols:
            result_dict[col] = self.embedding_dict[col](data_dict[col])
        
        self.assert_result(result_dict)
        return result_dict
    
    def assert_result(self, result_dict):
        for key, val in result_dict.items():
            assert len(val.shape) == 3


class BlockRemain(torch.nn.Module):
    def __init__(self, config, d_model, dropout):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = ["global"] + config.temporal_cols, config.img_cols, config.nlp_cols
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.mod_emb = torch.nn.Embedding(len(self.temporal_cols), d_model)

        self.remain_rto = config.remain_rto
    
    def forward(self, data_dict, idx_dict, mask_dict, device):
        result_dict = {}

        # PosMod Embedding
        for modality_idx, col in enumerate(self.temporal_cols):
            # Positional encoding
            val = self.pos_enc(data_dict[col])
            # Modality embedding
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = self.mod_emb(modality)
            result_dict[col] = val + modality
        
        # Temporal block
        temporal_block = torch.stack([val for key, val in result_dict.items()], dim=-2)
        global_block = temporal_block[:, :, :1, :]
        valid_block = temporal_block[:, :, 1:, :]
        assert temporal_block.shape[-2] == len(self.temporal_cols)

        global_mask = torch.ones(global_block.shape[:-1]).to(device)
        valid_mask = torch.ones(valid_block.shape[:-1]).to(device)
        fcst_mask = mask_dict["target_fcst_mask"]
        assert valid_mask.shape[:-1] == fcst_mask.shape
        valid_mask[:, :, 0] = fcst_mask

        # Get indices
        num_remain = int(valid_block.shape[-2] * self.remain_rto["temporal"])
        remain_idx, masked_idx, revert_idx = get_indices(valid_block.shape[:-1], num_remain)
        remain_idx, masked_idx, revert_idx = [i.to(device) for i in [remain_idx, masked_idx, revert_idx]]
        assert remain_idx.shape[-1]+masked_idx.shape[-1]+1 == revert_idx.shape[-1]+1 == valid_block.shape[-2]+1 == len(self.temporal_cols)

        # Apply remain
        remain_block = torch.gather(valid_block, index=remain_idx.unsqueeze(-1).expand(-1, -1, -1, valid_block.shape[-1]), dim=-2)
        remain_block = torch.cat([global_block, remain_block], dim=-2)
        assert remain_block.shape[-2] == num_remain+1

        remain_mask = torch.gather(valid_mask, index=remain_idx, dim=-1)
        remain_mask = torch.cat([global_mask, remain_mask], dim=-1)
        revert_mask = torch.cat([global_mask, valid_mask], dim=-1)

        idx_dict.update({"temporal_masked_idx":masked_idx, "temporal_revert_idx":revert_idx})
        mask_dict.update({"temporal_remain_mask":remain_mask, "temporal_revert_mask":revert_mask})

        return remain_block, idx_dict, mask_dict, self.mod_emb

class StaticRemain(torch.nn.Module):
    def __init__(self, config, d_model, dropout):
        super().__init__()
        self.img_cols, self.nlp_cols = config.img_cols, config.nlp_cols
        self.pos_enc_2d = torch.nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, grid_size=224//config.patch_size, cls_token=False)), requires_grad=False)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        self.remain_rto = config.remain_rto
    
    def forward(self, data_dict, idx_dict, mask_dict, device):
        result_dict = {}

        result_dict, idx_dict, mask_dict = self.process_img(result_dict, data_dict, idx_dict, mask_dict, device)
        result_dict = self.process_nlp(result_dict, data_dict, idx_dict, mask_dict, device)

        return result_dict, idx_dict, mask_dict
    
    def process_img(self, result_dict, data_dict, idx_dict, mask_dict, device):
        for key in self.img_cols:
            val = data_dict[key]
            revert_mask = torch.ones(val.shape[:-1]).to(device)

            # Positional encoding
            val = val + self.pos_enc_2d

            # Get indices
            num_remain = int(val.shape[1] * self.remain_rto["img"])
            remain_idx, masked_idx, revert_idx = get_indices(val.shape[:-1], num_remain)
            remain_idx, masked_idx, revert_idx = [i.to(device) for i in [remain_idx, masked_idx, revert_idx]]
            assert remain_idx.shape[-1]+masked_idx.shape[-1] == revert_idx.shape[-1] == val.shape[-2]

            # Apply remain
            remain = torch.gather(val, index=remain_idx.unsqueeze(-1).expand(-1, -1, val.shape[-1]), dim=1)
            remain_mask = torch.gather(revert_mask, index=remain_idx, dim=-1)
            masked_mask = torch.gather(revert_mask, index=masked_idx, dim=-1)
            
            result_dict.update({f"{key}_remain":remain})
            idx_dict.update({f"{key}_masked_idx":masked_idx, f"{key}_revert_idx":revert_idx})
            mask_dict.update({f"{key}_remain_mask":remain_mask, f"{key}_revert_mask":revert_mask})
            
            return result_dict, idx_dict, mask_dict

    def process_nlp(self, result_dict, data_dict, idx_dict, mask_dict, device):
        for key in self.nlp_cols:
            val = data_dict[key]
            
            # Positional encoding
            val = self.pos_enc(val)

            # Get indices
            remain_idx, masked_idx, revert_idx = [idx_dict[f"{key}_{idx_type}_idx"] for idx_type in ["remain", "masked", "revert"]]
            remain_idx, masked_idx, revert_idx = [i.to(device) for i in [remain_idx, masked_idx, revert_idx]]
            assert remain_idx.shape[-1]+masked_idx.shape[-1] == revert_idx.shape[-1] == val.shape[-2]

            # Apply remain
            remain = torch.gather(val, index=remain_idx.unsqueeze(-1).expand(-1, -1, val.shape[-1]), dim=1)

            result_dict.update({f"{key}_remain": remain})

            return result_dict


class BlockEncodingLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, dropout, activation):
        super().__init__()
        self.attn = MultiheadBlockAttention(d_model, nhead)
        self.attn_norm = torch.nn.LayerNorm(d_model)
        self.attn_dropout = torch.nn.Dropout(dropout)
    
        # Feed forward
        if activation == "relu": self.activation = torch.nn.ReLU()
        elif activation == "gelu": self.activation = torch.nn.GELU()

        self.ff_norm = torch.nn.LayerNorm(d_model)
        self.ff_linear1 = torch.nn.Linear(d_model, d_ff); self.ff_linear2 = torch.nn.Linear(d_ff, d_model)
        self.ff_dropout1 = torch.nn.Dropout(dropout); self.ff_dropout2 = torch.nn.Dropout(dropout)
    
    def forward(self, src, key_padding_mask):
        x = src
        
        attn_output, attn_weight = self.attn_block(self.attn_norm(x), key_padding_mask)
        x = x + attn_output

        ff_output = self.ff_block(self.ff_norm(x))
        x = x + ff_output

        return x, attn_weight
    
    def attn_block(self, src, key_padding_mask):
        attn_output, attn_weight = self.attn(src, src, src, key_padding_mask)
        return self.attn_dropout(attn_output), attn_weight
    
    def ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(x)
    
class BlockEncoding(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        self.layers = torch.nn.ModuleList([BlockEncodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
        self.norm = torch.nn.LayerNorm(d_model)
          
    def forward(self, temporal_block, mask_dict, device, mode):
        x = temporal_block
        mask = mask_dict[f"temporal_{mode}_mask"]
        if mode == "pre_train":
            assert len(torch.where(mask==0)[0]) == 0
        elif mode == "fine_tuning":
            assert len(torch.where(mask==0)[0]) != 0

        mask = torch.where(mask==1, 0, -torch.inf)
        
        for mod in self.layers:
            x, attn_weight = mod(x, mask)

        return self.norm(x)


class TotalEncodingLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.sa_norm = torch.nn.LayerNorm(d_model)
        self.sa_dropout = torch.nn.Dropout(dropout)

        self.cross_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ca_norm = torch.nn.LayerNorm(d_model)
        self.ca_dropout = torch.nn.Dropout(dropout)

        # Feed forward
        if activation == "relu": self.activation = torch.nn.ReLU()
        if activation == "gelu": self.activation = torch.nn.GELU()
        self.ff_norm = torch.nn.LayerNorm(d_model)
        self.ff_linear1 = torch.nn.Linear(d_model, d_ff)
        self.ff_linear2 = torch.nn.Linear(d_ff, d_model)
        self.ff_dropout1 = torch.nn.Dropout(dropout)
        self.ff_dropout2 = torch.nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask, memory_mask):
        x = tgt
        sa_output, sa_weight = self.sa_block(self.sa_norm(x), tgt_mask)
        x = x + sa_output

        ca_output, ca_weight = self.ca_block(self.ca_norm(x), memory, memory_mask)
        x = x + ca_output

        ff_output = self.ff_block(self.ff_norm(x))
        x = x + ff_output
        
        return x, sa_weight, ca_weight
    
    def sa_block(self, src, mask):
        attn_output, attn_weight = self.self_attn(src, src, src, key_padding_mask=mask)
        return self.sa_dropout(attn_output), attn_weight

    def ca_block(self, tgt, memory, mask):
        attn_output, attn_weight = self.cross_attn(tgt, memory, memory, key_padding_mask=mask)
        return self.ca_dropout(attn_output), attn_weight

    def ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(x)

class TotalEncoding_(torch.nn.Module):
    def __init__(self, config, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        self.img_cols, self.nlp_cols = config.img_cols, config.nlp_cols
        num_modality = len(["global"] + config.img_cols + config.nlp_cols)
        self.modality_embedding = torch.nn.Embedding(num_modality, d_model)

        self.total_encoding_dict = torch.nn.ModuleDict()
        for col in ["temporal"] + config.img_cols + config.nlp_cols:
            self.total_encoding_dict[col] = torch.nn.ModuleList([TotalEncodingLayer(d_model, d_ff, nhead, num_layers, dropout, activation) for _ in range(num_layers)])
        
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, temporal, static_dict, mask_dict, device, mode, modality_embedding=None):
        data_dict = {f"temporal_{mode}":temporal}
        data_dict.update(static_dict)
        
        # Modality embedding
        self.modality_embedding = modality_embedding if mode=="revert" else self.modality_embedding
        for modality_idx, (key, val) in enumerate(data_dict.items()):
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = self.modality_embedding(modality)

        # Total encoding
        encoding_dict, sa_weight_dict, ca_weight_dict = {}, {}, {}
        for col in  ["temporal"] + self.img_cols + self.nlp_cols:
            tgt = data_dict[f"{col}_{mode}"].clone()
            memory = torch.cat([data_dict[f"{i}_{mode}"] for i in ["temporal"] + self.img_cols + self.nlp_cols if i != col], dim=1)
            # memory = torch.cat([data_dict[f"{i}_{mode}"] for i in ["temporal"] + self.img_cols + self.nlp_cols], dim=1)

            tgt_mask = mask_dict[f"{col}_{mode}_mask"] if col!="temporal" else mask_dict[f"temporal_mask"]
            memory_mask = torch.cat([mask_dict[f"{i}_{mode}_mask"] if i!="temporal" else mask_dict[f"temporal_mask"] for i in ["temporal"] + self.img_cols + self.nlp_cols if i != col], dim=1)
            # memory_mask = torch.cat([mask_dict[f"{i}_{mode}_mask"] if i!="temporal" else mask_dict[f"temporal_mask"] for i in ["temporal"] + self.img_cols + self.nlp_cols], dim=1)

            tgt_mask = torch.where(tgt_mask==1, 0, -torch.inf)
            memory_mask = torch.where(memory_mask==1, 0, -torch.inf)
            assert tgt.shape[:-1] == tgt_mask.shape and memory.shape[:-1] == memory_mask.shape

            for mod in self.total_encoding_dict[col]:
                tgt, sa_weight, ca_weight = mod(tgt, memory, tgt_mask, memory_mask)
            
            encoding_dict[col] = self.norm(tgt)
            sa_weight_dict[col], ca_weight_dict[col] = sa_weight, ca_weight

        return encoding_dict, sa_weight_dict, ca_weight_dict, self.modality_embedding

class TotalEncoding(torch.nn.Module):
    def __init__(self, config, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        self.img_cols, self.nlp_cols = config.img_cols, config.nlp_cols
        num_modality = len(["global"] + config.img_cols + config.nlp_cols)
        self.modality_embedding = torch.nn.Embedding(num_modality, d_model)

        self.total_encoding_dict = torch.nn.ModuleDict()
        for col in ["temporal"] + config.img_cols + config.nlp_cols:
            self.total_encoding_dict[col] = torch.nn.ModuleList([TotalEncodingLayer(d_model, d_ff, nhead, num_layers, dropout, activation) for _ in range(num_layers)])
        
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, temporal, static_dict, mask_dict, device, mode, modality_embedding=None):
        data_dict = {f"temporal_{mode}":temporal}
        data_dict.update(static_dict)
        
        # Modality embedding
        self.modality_embedding = modality_embedding if mode=="revert" else self.modality_embedding
        for modality_idx, (key, val) in enumerate(data_dict.items()):
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = self.modality_embedding(modality)

        # Total encoding
        encoding_dict, sa_weight_dict, ca_weight_dict = {}, {}, {}
        for col in  ["temporal"] + self.img_cols + self.nlp_cols:
            tgt = data_dict[f"{col}_{mode}"].clone()
            memory = torch.cat([data_dict[f"{i}_{mode}"] for i in ["temporal"] + self.img_cols + self.nlp_cols if i != col], dim=1)
            # memory = torch.cat([data_dict[f"{i}_{mode}"] for i in ["temporal"] + self.img_cols + self.nlp_cols], dim=1)

            tgt_mask = mask_dict[f"{col}_{mode}_mask"] if col!="temporal" else mask_dict[f"temporal_mask"]
            memory_mask = torch.cat([mask_dict[f"{i}_{mode}_mask"] if i!="temporal" else mask_dict[f"temporal_mask"] for i in ["temporal"] + self.img_cols + self.nlp_cols if i != col], dim=1)
            # memory_mask = torch.cat([mask_dict[f"{i}_{mode}_mask"] if i!="temporal" else mask_dict[f"temporal_mask"] for i in ["temporal"] + self.img_cols + self.nlp_cols], dim=1)

            tgt_mask = torch.where(tgt_mask==1, 0, -torch.inf)
            memory_mask = torch.where(memory_mask==1, 0, -torch.inf)
            assert tgt.shape[:-1] == tgt_mask.shape and memory.shape[:-1] == memory_mask.shape

            for mod in self.total_encoding_dict[col]:
                tgt, sa_weight, ca_weight = mod(tgt, memory, tgt_mask, memory_mask)
            
            encoding_dict[col] = self.norm(tgt)
            sa_weight_dict[col], ca_weight_dict[col] = sa_weight, ca_weight

        return encoding_dict, sa_weight_dict, ca_weight_dict, self.modality_embedding


class BlockRevert(torch.nn.Module):
    def __init__(self, mask_token, d_model, dropout):
        super().__init__()
        self.mask_token = mask_token
        self.pos_enc = PositionalEncoding(d_model, dropout)
    
    def forward(self, temporal_block, idx_dict, mask_dict, temporal_mod_emb, device):
        global_block = temporal_block[:, :, :1, :]
        valid_block = temporal_block[:, :, 1:, :]

        # Append mask token
        batch_size, seq_len, num_modality, _ = valid_block.shape
        masked_idx, revert_idx = idx_dict["temporal_masked_idx"], idx_dict["temporal_revert_idx"]
        mask_token = self.mask_token.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, masked_idx.shape[-1], -1)
        assert mask_token.shape[:-1] == masked_idx.shape and revert_idx.shape[-1]==(num_modality+masked_idx.shape[-1])
        valid_block = torch.cat([valid_block, mask_token], dim=-2)
        assert valid_block.shape[:-1] == revert_idx.shape

        # Apply mask
        valid_block = torch.gather(valid_block, index=revert_idx.unsqueeze(-1).expand(-1, -1, -1, valid_block.shape[-1]), dim=-2)
        temporal_block = torch.cat([global_block, valid_block], dim=-2)
        
        # Pos mod embedding
        result_block = torch.zeros(temporal_block.shape).to(device)
        for modality_idx in range(temporal_block.shape[-2]):
            val = temporal_block[:, :, modality_idx, :]
            val = self.pos_enc(val)
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = temporal_mod_emb(modality)
            result_block[:, :, modality_idx, :] = val + modality
        # assert torch.where(result_block==0, 1, 0).sum() == 0, f"{result_block}"
        
        return result_block

class StaticRevert(torch.nn.Module):
    def __init__(self, config, mask_token, d_model, dropout):
        super().__init__()
        self.mask_token = mask_token
        self.img_cols, self.nlp_cols = config.img_cols, config.nlp_cols

        self.pos_enc_2d = torch.nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, grid_size=224//config.patch_size, cls_token=False)), requires_grad=False)
        self.pos_enc = PositionalEncoding(d_model, dropout)
    
    def forward(self, data_dict, idx_dict, mask_dict, device):
        result_dict = {}
        for col in self.img_cols + self.nlp_cols:
            val = data_dict[col]

            # Replace paddings to mask_token
            remain_mask = mask_dict[f"{col}_remain_mask"]
            val = torch.where(remain_mask.unsqueeze(-1).expand(-1, -1, val.shape[-1])==1, val, self.mask_token)

            # Append mask token
            batch_size, seq_len, _ = val.shape
            masked_idx, revert_idx = idx_dict[f"{col}_masked_idx"], idx_dict[f"{col}_revert_idx"]
            mask_token = self.mask_token.unsqueeze(0).expand(batch_size, masked_idx.shape[1], -1)
            assert mask_token.shape[:-1] == masked_idx.shape and revert_idx.shape[-1]==seq_len+masked_idx.shape[1]
            val = torch.cat([val, mask_token], dim=1)
            assert val.shape[:-1] == revert_idx.shape

            # Apply mask
            val = torch.gather(val, index=revert_idx.unsqueeze(-1).expand(-1, -1, val.shape[-1]), dim=1)

            # Positional encoding
            if col in self.img_cols:
                val = val + self.pos_enc_2d
            elif col in self.nlp_cols:
                val = self.pos_enc(val)

            result_dict[f"{col}_revert"] = val

        return result_dict


class Output(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, d_model):
        super().__init__()
        self.temporal_cols = config.temporal_cols
        self.output = torch.nn.ModuleDict()

        # Temporal
        for col in config.temporal_cols:
            if col in config.scaling_cols:
                self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, 1))
            elif col in config.embedding_cols:
                num_cls = label_encoder_dict[col].get_num_cls()
                self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, num_cls))
        # Img
        for col in config.img_cols:
            self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, 3*config.patch_size*config.patch_size))
        # Nlp
        for col in config.nlp_cols:
            self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, 30522))   
    
    def forward(self, data_dict):
        result_dict = {}
        for key, mod in self.output.items():
            data_key = "temporal" if key in self.temporal_cols else key
            result_dict[key] = mod(data_dict[data_key])

        return result_dict


class MBAEEncoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        # 1. Embedding
        self.embedding = Embedding(config, label_encoder_dict, d_model)
        # 2. Block remain
        self.block_remain = BlockRemain(config, d_model, dropout)
        # 3. Static remain
        self.static_remain = StaticRemain(config, d_model, dropout)
        # 4. Block encoding
        self.block_encoding = BlockEncoding(d_model, d_ff, nhead, num_layers, dropout, activation)
        # 5. Total encoding
        self.total_encoding = TotalEncoding(config, d_model, d_ff, nhead, num_layers, dropout, activation)
    
    def forward(self, data_dict, idx_dict, mask_dict, device):
        # 1. Embedding
        embedding_dict = self.embedding(data_dict)
        # 2. Block remain
        remain_block, idx_dict, mask_dict, temporal_mod_emb = self.block_remain(embedding_dict, idx_dict, mask_dict, device)
        # 3. Static remain
        remain_dict, idx_dict, mask_dict = self.static_remain(embedding_dict, idx_dict, mask_dict, device)
        # 4. Block encoding
        temporal_block = self.block_encoding(remain_block, mask_dict, device, "remain")
        # 5. Total encoding
        encoding_dict, sa_weight_dict, ca_weight_dict, total_mod_emb = self.total_encoding(temporal_block[:, :, 0, :], remain_dict, mask_dict, device, "remain")
        # 6. Update temporal block
        temporal_block[:, :, 0, :] = encoding_dict.pop("temporal")

        return temporal_block, encoding_dict, sa_weight_dict, ca_weight_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb

class MBAEDecoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, encoder_dmodel, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        # 0. Linear
        self.linear = torch.nn.Linear(encoder_dmodel, d_model)
        # 1. Block revert
        self.mask_token = torch.nn.Parameter(torch.rand(1, d_model))
        self.block_revert = BlockRevert(self.mask_token, d_model, dropout)
        # 2. Static revert
        self.static_revert = StaticRevert(config, self.mask_token, d_model, dropout)
        # 3. Block encoding
        self.block_encoding = BlockEncoding(d_model, d_ff, nhead, num_layers, dropout, activation)
        # 5. Total encoding
        self.total_encoding = TotalEncoding(config, d_model, d_ff, nhead, num_layers, dropout, activation)
        # 6. Output
        self.output = Output(config, label_encoder_dict, d_model)
  
    def forward(self, temporal_block, encoding_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb, device):
        # 0. Linear
        temporal_mod_emb = torch.nn.Sequential(temporal_mod_emb, self.linear)
        total_mod_emb = torch.nn.Sequential(total_mod_emb, self.linear)
        temporal_block = self.linear(temporal_block)
        data_dict = {key:self.linear(val) for key, val in encoding_dict.items()}
        # 1. Block revert
        revert_block = self.block_revert(temporal_block, idx_dict, mask_dict, temporal_mod_emb, device)
        # 2. Static revert
        revert_dict = self.static_revert(data_dict, idx_dict, mask_dict, device)
        # 3. Block encoding
        temporal_block = self.block_encoding(revert_block, mask_dict, device, "revert")
        # 4. Total encoding
        encoding_dict, sa_weight_dict, ca_weight_dict, _ = self.total_encoding(temporal_block[:, :, 0, :], revert_dict, mask_dict, device, "revert", total_mod_emb)
        # 5. Output
        output_dict = self.output(encoding_dict)
        
        return output_dict, sa_weight_dict, ca_weight_dict


class MaskedBlockAutoencoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict):
        super().__init__()
        self.config = config
        d_model, d_ff, nhead, num_layers, dropout, activation = self.get_params(config, "encoder")
        self.encoder = MBAEEncoder(config, label_encoder_dict, d_model, d_ff, nhead, num_layers, dropout, activation)

        d_model, d_ff, nhead, num_layers, dropout, activation = self.get_params(config, "decoder")
        self.decoder = MBAEDecoder(config, label_encoder_dict, config.d_model["encoder"], d_model, d_ff, nhead, num_layers, dropout, activation)

        self.mse_loss = torch.nn.MSELoss(reduction="none")
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, data_input, device):
        data_dict, idx_dict, mask_dict = self.to_gpu(data_input, device)

        temporal_block, encoding_dict, sa_weight_dict, ca_weight_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb = self.encoder(data_dict, idx_dict, mask_dict, device)
        output_dict, sa_weight_dict, ca_weight_dict = self.decoder(temporal_block, encoding_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb, device)

        loss_dict = self.get_loss(output_dict, data_dict, idx_dict, mask_dict)

        return output_dict, sa_weight_dict, ca_weight_dict, loss_dict

    def to_gpu(self, data_input, device):
        data_dict, idx_dict, mask_dict = {}, {}, {}
        data_cols = self.config.temporal_cols + self.config.img_cols + self.config.nlp_cols
        for key, val in data_input.items():
            if key in data_cols:
                data_dict[key] = data_input[key].to(device)
            elif key.endswith("idx"):
                idx_dict[key] = data_input[key].to(device)
            elif key.endswith("mask"):
                mask_dict[key] = data_input[key].to(device)
            
        return data_dict, idx_dict, mask_dict

    def get_params(self, config, mode):
        return config.d_model[mode], config.d_ff[mode], config.nhead[mode], config.num_layers[mode], config.dropout, config.activation
    
    def get_loss(self, output_dict, data_dict, idx_dict, mask_dict):
        loss_dict = {}
        # Temporal
        for n, col in enumerate(self.config.temporal_cols):
            pred, y = output_dict[col].squeeze(), data_dict[col].squeeze()
    
            ### Calculate loss
            if col in self.config.scaling_cols:
                loss = self.mse_loss(pred, y)
            elif col in self.config.embedding_cols:
                pred = pred.view(-1, pred.shape[-1])
                loss = self.ce_loss(pred, y.view(-1).to(torch.long)).view(y.shape)
            
            ### Apply mask
            masking_mask = (idx_dict["temporal_masked_idx"]==n).sum(dim=-1)
            padding_mask = mask_dict["temporal_mask"]
            mask = torch.where((masking_mask==1)&(padding_mask==1), 1, 0)
            loss_dict[col] = (loss*mask).sum() / mask.sum()
        
        # Img
        for col in self.config.img_cols:
            pred, y = output_dict[col].squeeze(), data_dict[col].squeeze()
            y = patchify(y, self.config.patch_size)

            # Calculate loss
            loss = self.mse_loss(pred, y)
            # Apply mask
            masked_idx = idx_dict[f"{col}_masked_idx"]
            loss = torch.gather(loss, index=masked_idx.unsqueeze(-1).expand(-1, -1, loss.shape[-1]), dim=1)
            loss_dict[col] = loss.mean()

        # Nlp
        for col in self.config.nlp_cols:
            pred, y = output_dict[col].squeeze(), data_dict[col].squeeze()
            
            # Calculate loss
            pred = pred.view(-1, pred.shape[-1])
            loss = self.ce_loss(pred, y.view(-1).to(torch.long)).view(y.shape)
            # Apply mask
            masked_idx = idx_dict[f"{col}_masked_idx"]
            padding_mask = mask_dict[f"{col}_masked_mask"]
            loss = torch.gather(loss, index=masked_idx, dim=1)
            loss_dict[col] = (loss*padding_mask).sum() / padding_mask.sum()
        
        return loss_dict


            