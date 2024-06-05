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
        self.global_token = torch.nn.Parameter(torch.rand(1, 1, d_model))

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
        result_dict["global"] = self.global_token.expand(batch_size, seq_len, -1)

        # Embed others
        for col in self.temporal_cols + self.img_cols + self.nlp_cols:
            result_dict[col] = self.embedding_dict[col](data_dict[col])

        return result_dict


class BlockRemain(torch.nn.Module):
    def __init__(self, config, d_model, dropout):
        super().__init__()
        self.remain_rto = config.remain_rto
        self.temporal_cols, self.img_cols, self.nlp_cols = ["global"] + config.temporal_cols, config.img_cols, config.nlp_cols
        
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.mod_emb = torch.nn.Embedding(len(self.temporal_cols), d_model)
    
    def forward(self, data_dict, idx_dict, device):
        result_dict = {}
        
        # PosMod embedding
        for modality_idx, col in enumerate(self.temporal_cols):
            val = self.pos_enc(data_dict[col]) # Pos enc
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = self.mod_emb(modality) # Mod emb
            result_dict[col] = val + modality
        
        # Temporal block
        temporal_block = torch.stack([val for key, val in result_dict.items() if key in self.temporal_cols], dim=-2)
        global_block = temporal_block[:, :, :1, :]
        valid_block = temporal_block[:, :, 1:, :]
        
        # Get indices
        remain_idx, masked_idx, revert_idx = get_indices(valid_block.shape[:-1], int(valid_block.shape[-2] * self.remain_rto["temporal"]))
        remain_idx, masked_idx, revert_idx = [i.to(device) for i in [remain_idx, masked_idx, revert_idx]]

        # Apply remain
        remain_block = torch.gather(valid_block, index=remain_idx.unsqueeze(-1).expand(-1, -1, -1, valid_block.shape[-1]), dim=-2)
        temporal_remain_block = torch.cat([global_block, remain_block], dim=-2)
        idx_dict.update({"temporal_masked_idx":masked_idx, "temporal_revert_idx":revert_idx})

        return temporal_remain_block, idx_dict, self.mod_emb

class StaticRemain(torch.nn.Module):
    def __init__(self, config, d_model, dropout):
        super().__init__()
        self.remain_rto = config.remain_rto
        self.img_cols, self.nlp_cols = config.img_cols, config.nlp_cols
        self.pos_enc_2d = torch.nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, grid_size=224//config.patch_size, cls_token=False)), requires_grad=False)
        self.pos_enc = PositionalEncoding(d_model, dropout)

    def forward(self, data_dict, idx_dict, mask_dict, device):
        result_dict = {}
        
        result_dict, idx_dict, mask_dict = self.process_img(data_dict, result_dict, idx_dict, mask_dict, device)
        result_dict, idx_dict = self.process_nlp(data_dict, result_dict, idx_dict, device)
        return result_dict, idx_dict, mask_dict
    
    def process_img(self, data_dict, result_dict, idx_dict, mask_dict, device):
        for col in self.img_cols:
            val = data_dict[col]
            revert_padding_mask = torch.ones(val.shape[:-1]).to(device)

            # Pos enc
            val = val + self.pos_enc_2d
        
            # Get indices
            remain_idx, masked_idx, revert_idx = get_indices(val.shape[:-1], int(val.shape[1] * self.remain_rto["img"]))
            remain_idx, masked_idx, revert_idx = [i.to(device) for i in [remain_idx, masked_idx, revert_idx]]

            # Apply remain
            remain = torch.gather(val, index=remain_idx.unsqueeze(-1).expand(-1, -1, val.shape[-1]), dim=1)
            remain_padding_mask = torch.gather(revert_padding_mask, index=remain_idx, dim=1)
            masked_padding_mask = torch.gather(revert_padding_mask, index=masked_idx, dim=1)

            result_dict.update({f"{col}_remain":remain})
            idx_dict.update({f"{col}_masked_idx":masked_idx, f"{col}_revert_idx":revert_idx})
            mask_dict.update({f"{col}_remain_padding_mask":remain_padding_mask, f"{col}_masked_padding_mask":masked_padding_mask, f"{col}_revert_padding_mask":revert_padding_mask})

            return result_dict, idx_dict, mask_dict

    def process_nlp(self, data_dict, result_dict, idx_dict, device):
        for col in self.nlp_cols:
            val = data_dict[col]

            # Pos enc
            val = self.pos_enc(val)

            # Get indices
            remain_idx, masked_idx, revert_idx = [idx_dict[f"{col}_{idx_type}_idx"] for idx_type in ["remain", "masked", "revert"]]

            # Apply remain
            remain = torch.gather(val, index=remain_idx.unsqueeze(-1).expand(-1, -1, val.shape[-1]), dim=1)

            result_dict.update({f"{col}_remain":remain})
            idx_dict.update({f"{col}_masked_idx":masked_idx, f"{col}_revert_idx":revert_idx})

            return result_dict, idx_dict


class BlockEncodingLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, dropout, activation):
        super().__init__()
        self.self_attn = MultiheadBlockAttention(d_model, nhead)
        self.sa_norm = torch.nn.LayerNorm(d_model)
        self.sa_dropout = torch.nn.Dropout(dropout)

        # Feed forward
        if activation == "relu": self.activation = torch.nn.ReLU()
        elif activation == "gelu": self.activation = torch.nn.GELU()

        self.ff_norm = torch.nn.LayerNorm(d_model)
        self.ff_linear1 = torch.nn.Linear(d_model, d_ff); self.ff_linear2 = torch.nn.Linear(d_ff, d_model)
        self.ff_dropout1 = torch.nn.Dropout(dropout); self.ff_dropout2 = torch.nn.Dropout(dropout)
    
    def forward(self, src, src_key_padding_mask):
        x = src
        attn_output, attn_weight = self._sa_block(self.sa_norm(x), src_key_padding_mask)
        x = x + attn_output

        ff_output = self._ff_block(self.ff_norm(x))
        x = x + ff_output
        
        return x, attn_output
    
    def _sa_block(self, src, src_key_padding_mask):
        attn_output, attn_weight = self.self_attn(src, src, src, src_key_padding_mask)
        return self.sa_dropout(attn_output), attn_weight
    
    def _ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(x)

class BlockEncoding(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        self.layers = torch.nn.ModuleList([BlockEncodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
        self.norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, temporal_block, mask_dict, device):
        x = temporal_block
        mask = torch.ones(x.shape[:-1]).to(device)
        # mask[:, :, 1] = mask_dict["target_fcst_mask"]
        mask = mask.unsqueeze(-2).expand(-1, -1, x.shape[-2], -1)
        mask = torch.where(mask==1, 0, -torch.inf)
        for mod in self.layers:
            x, attn_weight = mod(x, mask)
        
        return self.norm(x)


class TotalEncodingLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, dropout, activation):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
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
    
    def forward(self, src, src_key_padding_mask):
        x = src
        attn_output, attn_weight = self._sa_block(self.sa_norm(x), src_key_padding_mask)
        x = x + attn_output

        ff_output = self._ff_block(self.ff_norm(x))
        x = x + ff_output

        return x, attn_weight

    def _sa_block(self, src, src_key_padding_mask):
        attn_output, attn_weight = self.self_attn(src, src, src, src_key_padding_mask)
        return self.sa_dropout(attn_output), attn_weight
    
    def _ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(x)

class TotalEncoding(torch.nn.Module):
    def __init__(self, config, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        num_modality = len(["global"] + config.img_cols + config.nlp_cols)
        self.modality_embedding = torch.nn.Embedding(num_modality, d_model)

        self.layers = torch.nn.ModuleList([TotalEncodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
        self.norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, temporal, static, mask_dict, device, modality_embedding=None):
        data_dict, data_li, mask_li = {}, [], []
        data_dict.update({"temporal":temporal}); data_dict.update(static)

        # Modality embedding
        if modality_embedding:
            self.modality_embedding = modality_embedding
        for modality_idx, (key, val) in enumerate(data_dict.items()):
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = self.modality_embedding(modality)
            data_li.append(val + modality)

            mask = mask_dict["temporal_padding_mask"] if key == "temporal" else mask_dict[f"{key}_padding_mask"]
            mask_li.append(mask)
        
        # Encoding
        x = torch.cat(data_li, dim=1)
        mask = torch.cat(mask_li, dim=1)
        mask = torch.where(mask==1, 0, -torch.inf)

        for mod in self.layers:
            x, attn_weight = mod(x, mask)
        
        result = self.norm(x)
        
        # Split data
        length_dict, start_idx = {}, 0
        for key, val in data_dict.items():
            length = val.shape[1]
            length_dict[key] = [start_idx, start_idx + length]
            start_idx += length
        assert result.shape[1] == start_idx

        result_dict, attn_weight_dict = {}, defaultdict(dict)
        for key, val in length_dict.items():
            result_dict[key] = result[:, val[0]:val[1], :]
            for k, v in length_dict.items():
                attn_weight_dict[key].update({k:attn_weight[:, val[0]:val[1], v[0]:v[1]]})
        
        return result_dict, attn_weight_dict, self.modality_embedding


class BlockRevert(torch.nn.Module):
    def __init__(self, mask_token, d_model, dropout):
        super().__init__()
        self.mask_token = mask_token
        self.pos_enc = PositionalEncoding(d_model, dropout)
    
    def forward(self, temporal_block, idx_dict, mod_emb, device):
        global_block = temporal_block[:, :, :1, :]
        valid_block = temporal_block[:, :, 1:, :]

        # Append mask token
        batch_size, seq_len, num_modality, d_model = valid_block.shape
        revert_idx = idx_dict["temporal_revert_idx"]
        masked_idx = idx_dict["temporal_masked_idx"]
        mask_token = self.mask_token.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, revert_idx.shape[-1], -1)
        valid_block = torch.cat([valid_block, mask_token], dim=-2)

        # Apply mask idx
        revert_block = torch.gather(valid_block, index=revert_idx.unsqueeze(-1).expand(-1, -1, -1, d_model), dim=-2)

        # Pos mod emb
        temporal_revert_block = torch.cat([global_block, revert_block], dim=-2)
        result_block = temporal_revert_block.clone()
        for modality_idx in range(temporal_revert_block.shape[-2]):
            val = self.pos_enc(temporal_revert_block[:, :, modality_idx, :]) # Pos enc
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx # Mod emb
            modality = mod_emb(modality)
            result_block[:, :,  modality_idx, :] = val + modality

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
            remain_mask = mask_dict[f"{col}_remain_padding_mask"]
            val = torch.where(remain_mask.unsqueeze(-1).expand(-1, -1, val.shape[-1]) == 1, val, self.mask_token)

            # Append mask token
            batch_size, seq_len, d_model = val.shape
            revert_idx = idx_dict[f"{col}_revert_idx"]
            mask_token = self.mask_token.unsqueeze(0).expand(batch_size, revert_idx.shape[-1]-seq_len, -1)
            val = torch.cat([val, mask_token], dim=1)
            assert val.shape[:-1] == revert_idx.shape

            # Apply revert idx
            revert_data = torch.gather(val, index=revert_idx.unsqueeze(-1).expand(-1, -1, d_model), dim=1)

            # Pos enc
            if col in self.img_cols:
                revert_data = revert_data + self.pos_enc_2d
            elif col in self.nlp_cols:
                revert_data = self.pos_enc(revert_data)
            
            result_dict[f"{col}_revert"] = revert_data

        return result_dict


class Output(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, d_model):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = config.temporal_cols, config.img_cols, config.nlp_cols
        self.output = torch.nn.ModuleDict()
        d_model, nhead, d_ff, dropout, activation = config.d_model["decoder"],config. nhead["decoder"], config.d_ff["decoder"], config.dropout, config.activation

        # # Temporal
        # for col in self.temporal_cols:
        #     if col in config.scaling_cols:
        #         self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
        #                                                 torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), 2),
        #                                                 torch.nn.Linear(d_model, 1))
        #     elif col in config.embedding_cols:
        #         num_cls = label_encoder_dict[col].get_num_cls()
        #         self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
        #                                                 torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), 2),
        #                                                 torch.nn.Linear(d_model, num_cls))
        
        # # Img 
        # for col in self.img_cols:
        #     self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
        #                                             torch.nn.Linear(d_model, 3*config.patch_size*config.patch_size))
    
        # # Nlp
        # for col in self.nlp_cols:
        #     self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
        #                                             torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), 2),
        #                                             torch.nn.Linear(d_model, 30522))
        
        # Temporal
        for col in self.temporal_cols:
            if col in config.scaling_cols:
                self.output[col] = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model),
                                                        torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), 2),
                                                        torch.nn.Linear(d_model, 1)])
            elif col in config.embedding_cols:
                num_cls = label_encoder_dict[col].get_num_cls()
                self.output[col] = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model),
                                                        torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), 2),
                                                        torch.nn.Linear(d_model, num_cls)])
        
        # Img 
        for col in self.img_cols:
            self.output[col] = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model),
                                                    torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), 2),
                                                    torch.nn.Linear(d_model, 3*config.patch_size*config.patch_size)])
        # Nlp
        for col in self.nlp_cols:
            self.output[col] = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model),
                                                    torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), 2),
                                                    torch.nn.Linear(d_model, 30522)])
      
    def forward(self, data_dict, mask_dict):
        result_dict = {}
        # for col in self.temporal_cols+self.img_cols+self.nlp_cols:
        #     val = data_dict["temporal"] if col in self.temporal_cols else data_dict[f"{col}_revert"]
        #     mask = mask_dict["temporal_padding_mask"] if col in self.temporal_cols else mask_dict[f"{col}_revert_padding_mask"]
        #     result_dict[col] = self.output[col](val, src_key_padding_mask=mask)

        for col in self.temporal_cols+self.img_cols+self.nlp_cols:
            val = data_dict["temporal"] if col in self.temporal_cols else data_dict[f"{col}_revert"]
            mask = mask_dict["temporal_padding_mask"] if col in self.temporal_cols else mask_dict[f"{col}_revert_padding_mask"]
            mask = torch.where(mask==1, 0, -torch.inf)
            for n, mod in enumerate(self.output[col]):
                if n==1:
                    val = mod(val, src_key_padding_mask=mask)
                else:
                    val = mod(val)
            result_dict[col] = val

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
        temporal_remain_block, idx_dict, temporal_mod_emb = self.block_remain(embedding_dict, idx_dict, device)
        # 3. Static remain
        static_remain_dict, idx_dict, mask_dict = self.static_remain(embedding_dict, idx_dict, mask_dict, device)
        # 4. Block encoding
        temporal_remain_block = self.block_encoding(temporal_remain_block, mask_dict, device)
        # 5. Total encoding
        encoding_dict, attn_weight_dict, total_mod_emb = self.total_encoding(temporal_remain_block[:, :, 0, :], static_remain_dict, mask_dict, device)
        # 6. Update temporal block
        encoding_dict = self.update_temporal_block(temporal_remain_block, encoding_dict)

        return encoding_dict, attn_weight_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb
    
    def update_temporal_block(self, temporal_remain_block, encoding_dict):
        result_dict = {}
        updated_temporal_block = temporal_remain_block
        updated_temporal_block[:, :, 0, :] = encoding_dict["temporal"]
        result_dict["temporal"] = updated_temporal_block
        result_dict.update({key.replace("_remain", ""):val for key, val in encoding_dict.items() if key != "temporal"})
        return result_dict
        
class MBAEDecoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, encoder_d_model, d_model, d_ff, nhead, num_layers, dropout, activation):
        super().__init__()
        # 0. Linear and Mask token
        self.linear = torch.nn.Linear(encoder_d_model, d_model)
        self.mask_token = torch.nn.Parameter(torch.rand(1, d_model))
        # 1. Block revert
        self.block_revert = BlockRevert(self.mask_token, d_model, dropout)
        # 2. Static revert
        self.static_revert = StaticRevert(config, self.mask_token, d_model, dropout)
        # 3. Block encoding
        self.block_encoding = BlockEncoding(d_model, d_ff, nhead, num_layers, dropout, activation)
        # 4. Total encoding
        self.total_encoding = TotalEncoding(config, d_model, d_ff, nhead, num_layers, dropout, activation)
        # 5. Output
        self.output = Output(config, label_encoder_dict, d_model)
    
    def forward(self, encoding_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb, device):
        # 0. Linear
        temporal_mod_emb = torch.nn.Sequential(temporal_mod_emb, self.linear)
        total_mod_emb = torch.nn.Sequential(total_mod_emb, self.linear)
        encoding_dict = {key:self.linear(val) for key, val in encoding_dict.items()}
        
        # 1. Block revert
        # temporal_revert_block = self.block_revert(encoding_dict["temporal"], idx_dict, temporal_mod_emb, device)
        temporal_revert_block = self.block_revert(encoding_dict["temporal"], idx_dict, temporal_mod_emb, device)
        # 2. Static revert
        static_revert_dict = self.static_revert(encoding_dict, idx_dict, mask_dict, device)
        # 3. Block encoding
        temporal_revert_block = self.block_encoding(temporal_revert_block, mask_dict, device)
        # 4. Total encoding
        encoding_dict, attn_weight_dict, _ = self.total_encoding(temporal_revert_block[:, :, 0, :], static_revert_dict, mask_dict, device, total_mod_emb)
        # 5. Output
        result_dict = self.output(encoding_dict, mask_dict)
        
        return result_dict, attn_weight_dict


class MaskedBlockAutoencoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict):
        super().__init__()
        self.config = config

        mode = "encoder"
        d_model, d_ff, nhead, num_layers, dropout, activation = config.d_model[mode], config.d_ff[mode], config.nhead[mode], config.num_layers[mode], config.dropout, config.activation
        self.encoder = MBAEEncoder(config, label_encoder_dict, d_model, d_ff, nhead, num_layers, dropout, activation)
        
        mode = "decoder"
        d_model, d_ff, nhead, num_layers, dropout, activation = config.d_model[mode], config.d_ff[mode], config.nhead[mode], config.num_layers[mode], config.dropout, config.activation
        self.decoder = MBAEDecoder(config, label_encoder_dict, config.d_model["encoder"], d_model, d_ff, nhead, num_layers, dropout, activation)
    
    def forward(self, data_input, device):
        data_dict, idx_dict, mask_dict = self.to_gpu(data_input, device)
        
        encoding_dict, encoding_attn_weight_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb = self.encoder(data_dict, idx_dict, mask_dict, device)
        decoding_dict, decoding_attn_weight_dict = self.decoder(encoding_dict, idx_dict, mask_dict, temporal_mod_emb, total_mod_emb, device)
        return decoding_dict, decoding_attn_weight_dict, idx_dict, mask_dict
 
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