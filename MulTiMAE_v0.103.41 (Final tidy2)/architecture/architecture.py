import torch
from architecture.shared_module import *

def get_indices(data_shape, num_remain):
    # Get indices
    noise = torch.rand(data_shape)
    shuffle_idx = torch.argsort(noise, dim=-1)
    
    remain_idx = shuffle_idx[..., :num_remain]
    masked_idx = shuffle_idx[..., num_remain:]
    revert_idx = torch.argsort(shuffle_idx, dim=-1)

    return remain_idx, masked_idx, revert_idx


class Embedding(torch.nn.Module):
    """
    Embed data and append global tokens
    """
    def __init__(self, config, label_encoder_dict, global_token, d_model):
        super().__init__()
        self.config, self.global_token = config, global_token
        
        self.embedding_dict = torch.nn.ModuleDict()
        for col in config.temporal_cols + config.img_cols + config.nlp_cols:
            # Temporal
            if col in config.temporal_cols:
                if col in config.scaling_cols:
                    self.embedding_dict[col] = torch.nn.Linear(1, d_model)
                elif col in config.embedding_cols:
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

        # Global token
        batch_size, seq_len = data_dict[self.config.target_col[0]].shape[:-1]
        result_dict["global"] = self.global_token.expand(batch_size, seq_len, -1)

        # Temporal
        for col in self.config.temporal_cols:
            result_dict[col] = self.embedding_dict[col](data_dict[col])
        # Static
        for col in self.config.img_cols + self.config.nlp_cols:
            val = self.embedding_dict[col](data_dict[col])
            global_token = self.global_token.expand(val.shape[0], -1, -1)
            result_dict[col] = torch.cat([global_token, val], dim=1)
        
        return result_dict


class IndivPosEmbedding(torch.nn.Module):
    """
    Individual positional encoding without total modality embedding
    """
    def __init__(self, config, d_model, dropout):
        super().__init__()
        self.temporal_cols = ["global"] + config.temporal_cols
        self.img_cols = config.img_cols
        self.nlp_cols = config.nlp_cols
        self.max_seq_len = config.MAX_SEQ_LEN

        temporal_pos_enc_2d = torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, grid_size=config.MAX_SEQ_LEN, cls_token=False))
        self.temporal_pos_enc_2d = torch.nn.Parameter(temporal_pos_enc_2d.reshape(config.MAX_SEQ_LEN, config.MAX_SEQ_LEN, -1), requires_grad=False)
        self.img_pos_enc_2d = torch.nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, grid_size=224//config.patch_size, cls_token=True)), requires_grad=False)
        self.pos_enc = PositionalEncoding(d_model, dropout)
    
    def forward(self, data_dict):
        result_dict = {}
        # Temporal
        temporal_block = torch.stack([val for key, val in data_dict.items() if key in self.temporal_cols], dim=-2)
        temporal_pos_enc_2d = self.temporal_pos_enc_2d[:temporal_block.shape[1], :len(self.temporal_cols), :]
        result_dict["temporal_block"] = temporal_block + temporal_pos_enc_2d
        # Img
        for col in self.img_cols:
            result_dict[col] = data_dict[col] + self.img_pos_enc_2d
        # Nlp
        for col in self.nlp_cols:
            result_dict[col] = self.pos_enc(data_dict[col])
        
        return result_dict


class RemainMasking(torch.nn.Module):
    """
    Apply remain masking without total modality embedding
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temporal_cols, self.img_cols, self.nlp_cols = ["global"] + config.temporal_cols, config.img_cols, config.nlp_cols
        self.remain_rto = config.remain_rto
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, device):
        remain_dict = {}
        remain_dict, idx_dict = self.process_temporal(remain_dict, data_dict, idx_dict, device)
        remain_dict, idx_dict, padding_mask_dict = self.process_img(remain_dict, data_dict, idx_dict, padding_mask_dict, device)
        remain_dict = self.process_nlp(remain_dict, data_dict, idx_dict, padding_mask_dict, device)

        return remain_dict, idx_dict, padding_mask_dict
    
    def process_temporal(self, remain_dict, data_dict, idx_dict, device):
        temporal_block = data_dict["temporal_block"]
        global_block = temporal_block[:, :, :1, :]
        valid_block = temporal_block[:, :, 1:, :]
        remain_idx, masked_idx, revert_idx = get_indices(valid_block.shape[:-1], int(valid_block.shape[-2] * self.remain_rto["temporal"]))
        remain_idx, masked_idx, revert_idx = [i.to(device) for i in [remain_idx, masked_idx, revert_idx]]

        remain_block = torch.gather(valid_block, index=remain_idx.unsqueeze(-1).expand(-1, -1, -1, valid_block.shape[-1]), dim=-2)
        temporal_remain_block = torch.cat([global_block, remain_block], dim=-2)

        remain_dict.update({"temporal_remain_block":temporal_remain_block})
        idx_dict.update({"temporal_masked_idx":masked_idx, "temporal_revert_idx":revert_idx})

        return remain_dict, idx_dict

    def process_img(self, remain_dict, data_dict, idx_dict, padding_mask_dict, device):
        for col in self.img_cols:
            img_data = data_dict[col]
            img_revert_padding_mask = torch.ones(img_data.shape[:-1]).to(device)
            
            # Obtain masking idx
            global_token = img_data[:, :1, :]; global_padding_mask = img_revert_padding_mask[:, :1]
            valid_token = img_data[:, 1:, :]; valid_padding_mask = img_revert_padding_mask[:, 1:]
            remain_idx, masked_idx, revert_idx = get_indices(valid_token.shape[:-1], int(valid_token.shape[1] * self.config.remain_rto["img"]))
            remain_idx, masked_idx, revert_idx = [i.to(device) for i in [remain_idx, masked_idx, revert_idx]]

            # Apply masking
            remain_token = torch.gather(valid_token, index=remain_idx.unsqueeze(-1).expand(-1, -1, valid_token.shape[-1]), dim=1)
            remain_padding_mask = torch.gather(valid_padding_mask, index=remain_idx, dim=1)
            masked_padding_mask = torch.gather(valid_padding_mask, index=masked_idx, dim=1)

            img_remain = torch.cat([global_token, remain_token], dim=1)
            img_remain_padding_mask = torch.cat([global_padding_mask, remain_padding_mask], dim=1)
            img_masked_padding_mask = torch.cat([global_padding_mask, masked_padding_mask], dim=1)

            assert img_remain.shape[1] == img_remain_padding_mask.shape[1], f"remain_token: {remain_token.shape},  remain_padding_mask: {static_remain_padding_mask.shape}"
            assert remain_idx.shape[1]+1 == img_remain_padding_mask.shape[1], f"remain_idx: {remain_idx.shape}, remain_padding_mask: {static_remain_padding_mask.shape}"
            assert masked_idx.shape[1]+1 == img_masked_padding_mask.shape[1], f"masked_idx: {masked_idx.shape}, masked_padding_mask: {static_masked_padding_mask.shape}"
            assert revert_idx.shape[1]+1 == img_revert_padding_mask.shape[1], f"revert_idx: {revert_idx.shape}, revert_padding_mask: {static_revert_padding_mask.shape}"

            remain_dict.update({f"{col}_remain":img_remain})
            idx_dict.update({f"{col}_masked_idx":masked_idx, f"{col}_revert_idx":revert_idx})
            padding_mask_dict.update({f"{col}_remain_padding_mask":img_remain_padding_mask, f"{col}_masked_padding_mask":img_masked_padding_mask, f"{col}_revert_padding_mask":img_revert_padding_mask})
        
        return remain_dict, idx_dict, padding_mask_dict
    
    def process_nlp(self, remain_dict, data_dict, idx_dict, padding_mask_dict, device):
        for col in self.nlp_cols:
            nlp_data = data_dict[col]
            nlp_revert_padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]

            # Obtain masking idx
            global_token = nlp_data[:, :1, :]; global_padding_mask = nlp_revert_padding_mask[:, :1]
            valid_token = nlp_data[:, 1:, :]; valid_padding_mask = nlp_revert_padding_mask[:, 1:]
            remain_idx, masked_idx, revert_idx = [idx_dict[f"{col}_{idx_type}_idx"] for idx_type in ["remain", "masked", "revert"]]
            
            # Apply masking
            remain_token = torch.gather(valid_token, index=remain_idx.unsqueeze(-1).expand(-1, -1, valid_token.shape[-1]), dim=1)
            remain_padding_mask = torch.gather(valid_padding_mask, index=remain_idx, dim=1)
            masked_padding_mask = torch.gather(valid_padding_mask, index=masked_idx, dim=1)

            nlp_remain = torch.cat([global_token, remain_token], dim=1)
            nlp_remain_padding_mask = torch.cat([global_padding_mask, remain_padding_mask], dim=1)
            nlp_masked_padding_mask = torch.cat([global_padding_mask, masked_padding_mask], dim=1)

            assert nlp_remain.shape[1] == nlp_remain_padding_mask.shape[1], f"remain_token: {remain_token.shape},  remain_padding_mask: {static_remain_padding_mask.shape}"
            assert remain_idx.shape[1]+1 == nlp_remain_padding_mask.shape[1], f"remain_idx: {remain_idx.shape}, remain_padding_mask: {static_remain_padding_mask.shape}"
            assert masked_idx.shape[1]+1 == nlp_masked_padding_mask.shape[1], f"masked_idx: {masked_idx.shape}, masked_padding_mask: {static_masked_padding_mask.shape}"
            assert revert_idx.shape[1]+1 == nlp_revert_padding_mask.shape[1], f"revert_idx: {revert_idx.shape}, revert_padding_mask: {static_revert_padding_mask.shape}"

            remain_dict.update({f"{col}_remain": nlp_remain})
            padding_mask_dict.update({f"{col}_remain_padding_mask":nlp_remain_padding_mask, f"{col}_masked_padding_mask":nlp_masked_padding_mask, f"{col}_revert_padding_mask":nlp_revert_padding_mask})

        return remain_dict


class IndividualEncodingLayer(torch.nn.Module):
    def __init__(self, col, d_model, d_ff, nhead, dropout, activation):
        super().__init__()
        self.self_attn = MultiheadBlockAttention(d_model, nhead) if col == "temporal" else torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
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

class IndividualEncoding(torch.nn.Module):
    def __init__(self, config, mode, d_model, d_ff, nhead, dropout, activation, num_layers):
        super().__init__()
        self.config, self.mode = config, mode
        self.img_cols, self.nlp_cols = config.img_cols, config.nlp_cols
        
        self.layers_dict = torch.nn.ModuleDict()
        for col in ["temporal"] + self.img_cols + self.nlp_cols:
            self.layers_dict[col] = torch.nn.ModuleList([IndividualEncodingLayer(col, d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, data_dict, padding_mask_dict):
        attn_output_dict, attn_weight_dict = {}, {}

        # Temporal
        x = data_dict[f"temporal_{self.mode}_block"]
        for mod in self.layers_dict["temporal"]:
            x, attn_weight = mod(x, src_key_padding_mask=None)
        attn_output_dict["temporal"] = x
        attn_weight_dict["temporal"] = attn_weight

        # Static
        for key in self.img_cols + self.nlp_cols:
            x = data_dict[f"{key}_{self.mode}"]
            padding_mask = padding_mask_dict[f"{key}_{self.mode}_padding_mask"]
            padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
            for mod in self.layers_dict[key]:
                x, attn_weight = mod(x, src_key_padding_mask=padding_mask)
            
            attn_output_dict[key] = x
            attn_weight_dict[key] = attn_weight
        
        return attn_output_dict, attn_weight_dict


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
    def __init__(self, config, modality_embedding, d_model, d_ff, nhead, dropout, activation, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([TotalEncodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
        self.modality_embedding = modality_embedding
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, data_dict, padding_mask_dict, device):
        # Modality embedding
        data_li, padding_mask_li = [], []
        for modality_idx, (key, val) in enumerate(data_dict.items()):
            val = val[:, :, 0, :] if key == "temporal" else val
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = self.modality_embedding(modality)
            data_li.append(val + modality)

            padding_mask = padding_mask_dict["temporal_padding_mask"] if key == "temporal" else padding_mask_dict[f"{key}_remain_padding_mask"]
            padding_mask_li.append(padding_mask)
        
        # Encoding
        x = torch.cat(data_li, dim=1)
        padding_mask = torch.cat(padding_mask_li, dim=1)
        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
        
        for mod in self.layers:
            x, attn_weight = mod(x, padding_mask)
        x = self.norm(x)
        
        # Split data
        result_dict, attn_weight_dict = {}, {}
        start_idx = 0
        for key, val in data_dict.items():
            length = val.shape[1]
            result_dict[key] = x[:, start_idx:start_idx+length, :]
            attn_weight_dict[key] = attn_weight[:, start_idx:start_idx+length, :]
            start_idx += length
        assert x.shape[1] == start_idx

        return result_dict, attn_weight_dict


class Revert(torch.nn.Module):
    def __init__(self, config, mask_token):
        super().__init__()
        self.mask_token = mask_token
        self.img_cols, self.nlp_cols = config.img_cols, config.nlp_cols
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, device):
        result_dict = {}
        temporal_revert_block = self.get_temporal_revert(data_dict["temporal"], idx_dict)
        static_revert_dict = self.get_static_revert(data_dict, idx_dict, padding_mask_dict)

        result_dict.update({"temporal_revert_block": temporal_revert_block})
        result_dict.update(static_revert_dict)

        return result_dict
    
    def get_temporal_revert(self, data, idx_dict):
        global_block = data[:, :, :1, :]
        valid_block = data[:, :, 1:, :]

        # Append mask token
        batch_size, seq_len, num_modality, d_model = valid_block.shape
        revert_idx = idx_dict["temporal_revert_idx"]
        mask_token = self.mask_token.unsqueeze(1).expand(batch_size, seq_len, revert_idx.shape[-1]-num_modality, -1)
        filled_token = torch.cat([valid_block, mask_token], dim=-2)

        # Apply mask idx
        revert_block = torch.gather(filled_token, index=revert_idx.unsqueeze(-1).expand(-1, -1, -1, d_model), dim=-2)
        temporal_revert_block = torch.cat([global_block, revert_block], dim=-2)

        return temporal_revert_block

    def get_static_revert(self, data_dict, idx_dict, padding_mask_dict):
        result_dict = {}
        for col in self.img_cols + self.nlp_cols:
            data = data_dict[col]

            # Replace paddings to mask_token
            remain_padding_mask = padding_mask_dict[f"{col}_remain_padding_mask"]
            data = torch.where(remain_padding_mask.unsqueeze(-1).expand(-1, -1, data.shape[-1])==1, data, self.mask_token)

            # Split global token
            global_token = data[:, :1, :]
            valid_token = data[:, 1:, :]

            # Append mask token
            batch_size, seq_len, d_model = valid_token.shape
            revert_idx = idx_dict[f"{col}_revert_idx"]
            mask_token = self.mask_token.expand(batch_size, revert_idx.shape[-1]-seq_len, -1)
            valid_token = torch.cat([valid_token, mask_token], dim=1)
            assert valid_token.shape[:-1] == revert_idx.shape, f"valid_token: {valid_token.shape}, revert_idx: {revert_idx.shape}"
            
            # Apply revert idx
            revert_data = torch.gather(valid_token, index=revert_idx.unsqueeze(-1).expand(-1, -1, d_model), dim=1)
            static_revert = torch.cat([global_token, revert_data], dim=1)

            result_dict[col] = static_revert

        return result_dict


class TotalDecodingLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, dropout, activation):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.sa_norm = torch.nn.LayerNorm(d_model)
        self.sa_dropout = torch.nn.Dropout(dropout)

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
    
    def forward(self, tgt, memory, tgt_padding_mask, memory_padding_mask):
        x = tgt

        sa_weight = None
        sa_output, sa_weight = self._sa_block(self.sa_norm(x), tgt_padding_mask)
        x = x + sa_output

        ca_output, ca_weight = self._ca_block(self.ca_norm(x), memory, memory_padding_mask)
        x = x + ca_output

        ff_output = self._ff_block(self.ff_norm(x))
        x = x + ff_output

        return x, sa_weight, ca_weight

    def _sa_block(self, src, src_key_padding_mask):
        attn_output, attn_weight = self.self_attn(src, src, src, src_key_padding_mask)
        return self.sa_dropout(attn_output), attn_weight

    def _ca_block(self, tgt, memory, memory_key_padding_mask):
        attn_output, attn_weight = self.cross_attn(tgt, memory, memory, memory_key_padding_mask)
        return self.ca_dropout(attn_output), attn_weight
    
    def _ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(x)

class TotalDecoding(torch.nn.Module):
    def __init__(self, config, modality_embedding, d_model, d_ff, nhead, dropout, activation, num_layers, linear):
        super().__init__()
        self.linear, self.modality_embedding = linear, modality_embedding
        self.temporal_cols, self.img_cols, self.nlp_cols = config.temporal_cols, config.img_cols, config.nlp_cols
        
        self.decoder_layer_dict = torch.nn.ModuleDict()
        for col in self.temporal_cols + self.img_cols + self.nlp_cols:
            self.decoder_layer_dict[col] = torch.nn.ModuleList([TotalDecodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
        
        self.norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, data_dict, padding_mask_dict, device):
        modemb_dict, result_dict, sa_weight_dict, ca_weight_dict = {}, {}, {}, {}
        
        # Modality embedding
        for modality_idx, (key, val) in enumerate(data_dict.items()):
            val = val[:, :, 0, :] if key=="temporal" else val
            modality = torch.zeros(val.shape[:-1]).to(torch.int).to(device) + modality_idx
            modality = self.linear(self.modality_embedding(modality))
            modemb_dict[key] = val + modality

        # Temporal decoding
        for col in self.temporal_cols:
            tgt = modemb_dict["temporal"]
            tgt_padding_mask = padding_mask_dict["temporal_padding_mask"]
            tgt_padding_mask = torch.where(tgt_padding_mask==1, 0, -torch.inf)

            memory = torch.cat([modemb_dict[i] for i in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.cat([padding_mask_dict["temporal_padding_mask"] if i=="temporal" else padding_mask_dict[f"{i}_revert_padding_mask"] for i in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.where(memory_padding_mask==1, 0, -torch.inf)

            for mod in self.decoder_layer_dict[col]:
                tgt, sa_weight, ca_weight = mod(tgt, memory, tgt_padding_mask, memory_padding_mask)

            result_dict[col], sa_weight_dict[col], ca_weight_dict[col] = tgt, sa_weight_dict, ca_weight
        
        # Static decoding
        for col in self.img_cols+self.nlp_cols:
            tgt = modemb_dict[col]
            tgt_padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]
            tgt_padding_mask = torch.where(tgt_padding_mask==1, 0, -torch.inf)

            memory = torch.cat([modemb_dict[i] for i in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.cat([padding_mask_dict["temporal_padding_mask"] if i=="temporal" else padding_mask_dict[f"{i}_revert_padding_mask"] for i in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.where(memory_padding_mask==1, 0, -torch.inf)

            for mod in self.decoder_layer_dict[col]:
                tgt, sa_weight, ca_weight = mod(tgt, memory, tgt_padding_mask, memory_padding_mask)
            
            result_dict[col], sa_weight_dict[col], ca_weight_dict[col] = tgt, sa_weight_dict, ca_weight

        return result_dict, ca_weight_dict


class Output(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, d_model):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = config.temporal_cols, config.img_cols, config.nlp_cols
        self.output = torch.nn.ModuleDict()
        
        # Temporal
        for col in self.temporal_cols:
            if col in config.scaling_cols:
                self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                        torch.nn.Linear(d_model, 1))
            elif col in config.embedding_cols:
                num_cls = label_encoder_dict[col].get_num_cls()
                self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                        torch.nn.Linear(d_model, num_cls))

        # Img 
        for col in self.img_cols:
            self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                    torch.nn.Linear(d_model, 3*config.patch_size*config.patch_size))
    
        # Nlp
        for col in self.nlp_cols:
            self.output[col] = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                    torch.nn.Linear(d_model, 30522))
    
    def forward(self, data_dict):
        result_dict = {}
        for col in self.temporal_cols+self.img_cols+self.nlp_cols:
            result_dict[col] = self.output[col](data_dict[col])

        return result_dict


class MBAEEncoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict,
                    global_token, mask_token, modality_embedding):
        super().__init__()
        self.config = config()

        # 1. Embedding
        self.embedding = Embedding(config, label_encoder_dict, global_token, config.d_model["encoder"])
        # 2. Individual/Total pos_mod embedding
        self.posmod_embedding = IndivPosEmbedding(config, config.d_model["encoder"], config.dropout)
        # 3. Remain masking
        self.remain_masking = RemainMasking(config)
        # 4. Individual encoding
        self.indinv_encoding = IndividualEncoding(config, "remain", config.d_model["encoder"], config.d_ff["encoder"], config.nhead["encoder"], config.dropout, config.activation, config.num_layers["encoder"])
        # 5. Total encoding
        self.total_encoding = TotalEncoding(config, modality_embedding, config.d_model["encoder"], config.d_ff["encoder"], config.nhead["encoder"], config.dropout, config.activation, config.num_layers["encoder"])
        # 7. Revert
        self.revert = Revert(config, mask_token)
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, device):
        # 1. Embedding
        embedding_dict = self.embedding(data_dict)
        # 2. Pos_mod embedding
        embedding_dict = self.posmod_embedding(embedding_dict)
        # 3. Remain masking
        remain_dict, idx_dict, padding_mask_dict = self.remain_masking(embedding_dict, idx_dict, padding_mask_dict, device)
        # 4. Individual encoding
        indiv_encoding_dict, indiv_weight_dict = self.indinv_encoding(remain_dict, padding_mask_dict)
        # 5. Total encoding
        total_encoding_dict, total_weight_dict = self.total_encoding(indiv_encoding_dict, padding_mask_dict, device)
        # 6. Update temporal block
        total_encoding_dict = self.update_temporal_block(indiv_encoding_dict, total_encoding_dict)
        # 7. Revert
        revert_dict = self.revert(total_encoding_dict, idx_dict, padding_mask_dict, device)
        # 8. Split individual
        revert_dict = self.split_individual(revert_dict)
        
        return revert_dict, total_weight_dict, idx_dict, padding_mask_dict

    def update_temporal_block(self, indiv_encoding_dict, total_encoding_dict):
        temporal_block = indiv_encoding_dict["temporal"]
        temporal_block[:, :, 0, :] = total_encoding_dict["temporal"]
        total_encoding_dict["temporal"] = temporal_block
        
        return total_encoding_dict
        
    def split_individual(self, data_dict):
        result_dict = {}
        temporal_block = data_dict["temporal_revert_block"]

        for n, col in enumerate(["global"] + self.config.temporal_cols):
            result_dict[col] = temporal_block[:, :, n, :]
        
        result_dict.update({key:val for key, val in data_dict.items() if key != "temporal_revert_block"})
        return result_dict

class MBAEDecoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, modality_embedding):
        super().__init__()
        # 0. To decoder dimension
        self.linear = torch.nn.Linear(config.d_model["encoder"], config.d_model["decoder"])
        # 1. Individual/Total pos_mod embedding
        self.posmod_embedding = IndivPosEmbedding(config, config.d_model["decoder"], config.dropout)
        # 3. Individual encoding
        self.indinv_encoding = IndividualEncoding(config, "revert", config.d_model["decoder"], config.d_ff["decoder"], config.nhead["decoder"], config.dropout, config.activation, config.num_layers["decoder"])
        # 4. Total decoding
        self.total_decoding = TotalDecoding(config, modality_embedding, config.d_model["decoder"], config.d_ff["decoder"], config.nhead["decoder"], config.dropout, config.activation, config.num_layers["decoder"], self.linear)
        # 5. Output
        self.output = Output(config, label_encoder_dict, config.d_model["decoder"])
    
    def forward(self, encoding_dict, padding_mask_dict, device):
        # 0. To decoder dimension
        decoding_dict = {key:self.linear(val) for key, val in encoding_dict.items()}
        # 1. Pos_mod embedding
        decoding_dict = self.posmod_embedding(decoding_dict)
        # 2. Rename Dict
        decoding_dict = {"temporal_revert_block" if key=="temporal_block" else f"{key}_revert":val for key, val in decoding_dict.items()}
        # 3. Individual encoding
        indiv_encoding_dict, indiv_weight_dict = self.indinv_encoding(decoding_dict, padding_mask_dict)
        # 4. Total decoding
        total_decoding_dict, total_weight_dict = self.total_decoding(indiv_encoding_dict, padding_mask_dict, device)
        # 5. Output
        output_dict = self.output(total_decoding_dict)

        return output_dict, total_weight_dict


class MaskedBlockAutoencoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict):
        super().__init__()
        # Define shared variables
        self.config = config
        global_token = torch.nn.Parameter(torch.rand(1, 1, config.d_model["encoder"]))
        mask_token = torch.nn.Parameter(torch.rand(1, 1, config.d_model["encoder"]))

        num_modality = len(["global"] + config.img_cols + config.nlp_cols)
        modality_embedding = torch.nn.Embedding(num_modality, config.d_model["encoder"])

        # Define encoder and decoder
        self.encoder = MBAEEncoder(config, label_encoder_dict,
                                    global_token, mask_token, modality_embedding)
        self.decoder = MBAEDecoder(config, label_encoder_dict, modality_embedding)
    
    def forward(self, data_input, device, mode="pre_train"):
        data_dict, idx_dict, padding_mask_dict = self.to_gpu(data_input, device)

        encoding_dict, encoding_weight_dict, idx_dict, padding_mask_dict = self.encoder(data_dict, idx_dict, padding_mask_dict, device)
        
        if mode == "pre_train":
            decoding_dict, decoding_weight_dict = self.decoder(encoding_dict, padding_mask_dict, device)
            return decoding_dict, encoding_weight_dict, decoding_weight_dict, idx_dict, padding_mask_dict
        elif mode == "fine_tuning":
            return encoding_dict, data_dict, encoding_weight_dict, padding_mask_dict

    def to_gpu(self, data_input, device):
        data_dict, idx_dict, padding_mask_dict = {}, {}, {}
        data_cols = self.config.temporal_cols + self.config.img_cols + self.config.nlp_cols
        for key, val in data_input.items():
            if key in data_cols:
                data_dict[key] = data_input[key].to(device)
            elif key.endswith("idx"):
                idx_dict[key] = data_input[key].to(device)
            elif key.endswith("mask"):
                padding_mask_dict[key] = data_input[key].to(device)
            
        return data_dict, idx_dict, padding_mask_dict