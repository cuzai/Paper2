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
    def __init__(self, config, label_encoder_dict, global_token, d_model, patch_size):
        super().__init__()
        self.config = config
        self.global_token = global_token
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
                self.global_token = global_token
                self.embedding_dict[col] = PatchEmbed(224, patch_size, 3, d_model)
            
            # Nlp
            elif col in config.nlp_cols:
                self.embedding_dict[col] = torch.nn.Embedding(30522, d_model)
        
    def forward(self, data):
        result_dict = {}
        # Global token
        target_col = self.config.target_col[0]
        batch_size, seq_len = data[target_col].shape[:-1]
        result_dict["global"] = self.global_token.expand(batch_size, seq_len, -1)

        for key, val in data.items():
            result_dict[key] = self.embedding_dict[key](val)

        return result_dict

class RemainMasking(torch.nn.Module):
    def __init__(self, temporal_cols, img_cols, nlp_cols, temporal_modality_embedding, total_modality_embedding, global_token, remain_rto, patch_size, d_model, dropout):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols
        self.global_token = global_token 
        self.remain_rto = remain_rto

        self.total_modality_embedding = total_modality_embedding
        # Temporal
        self.temporal_modality_embedding = temporal_modality_embedding
        self.temporal_pos_enc = PositionalEncoding(d_model, dropout)
        # Img
        self.pos_enc_2d = torch.nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, grid_size=224//patch_size, cls_token=True)), requires_grad=False)
        # Nlp
        self.nlp_pos_enc = PositionalEncoding(d_model, dropout)
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, device):
        remain_dict = {}
        remain_dict, idx_dict = self.get_temporal_remain_block(data_dict, remain_dict, idx_dict, self.remain_rto["temporal"], device)
        remain_dict, idx_dict, padding_mask_dict = self.get_static_remain(data_dict, remain_dict, idx_dict, padding_mask_dict, self.remain_rto["img"], device)

        return remain_dict, idx_dict, padding_mask_dict
    
    def get_temporal_remain_block(self, data_dict, remain_dict, idx_dict, remain_rto, device):
        # Make temporal block
        total_modality_idx = 0
        temporal_dict = {}
        for temporal_modality_idx, key in enumerate(self.temporal_cols):
            val = data_dict[key]
            val = self.temporal_pos_enc(val)
            temporal_modality = torch.zeros(1, val.shape[1]).to(torch.int).to(device) + temporal_modality_idx
            temporal_modality = self.temporal_modality_embedding(temporal_modality).expand(val.shape[0], -1, -1)

            total_modality = torch.zeros(1, val.shape[1]).to(torch.int).to(device) + total_modality_idx
            total_modality = self.total_modality_embedding(total_modality).expand(val.shape[0], -1, -1)
            val = val + temporal_modality + total_modality
            temporal_dict[key] = val
        temporal_block = torch.stack([val for key, val in temporal_dict.items()], dim=-2)

        # Obtain masking idx
        global_block = temporal_block[:, :, :1, :]
        valid_block = temporal_block[:, :, 1:, :]
        remain_idx, masked_idx, revert_idx = get_indices(valid_block.shape[:-1], int(valid_block.shape[-2]*remain_rto))
        remain_idx, masked_idx, revert_idx = remain_idx.to(device), masked_idx.to(device), revert_idx.to(device)

        # Apply masking
        remain_block = torch.gather(valid_block, index=remain_idx.unsqueeze(-1).expand(-1, -1, -1, valid_block.shape[-1]), dim=-2)
        temporal_remain_block = torch.cat([global_block, remain_block], dim=-2)

        remain_dict.update({"temporal_remain_block":temporal_remain_block})
        idx_dict.update({"temporal_masked_idx":masked_idx, "temporal_revert_idx":revert_idx})
        return remain_dict, idx_dict

    def get_static_remain(self, data_dict, remain_dict, idx_dict, padding_mask_dict, remain_rto, device):
        for col in self.img_cols + self.nlp_cols:
            # Append global token
            global_token = self.global_token.expand(data_dict[col].shape[0], -1, -1)
            static_data = torch.cat([global_token, data_dict[col]], dim=-2)
            # Positional encoding
            if col in self.img_cols: static_data += self.pos_enc_2d
            elif col in self.nlp_cols: static_data = self.nlp_pos_enc(static_data)

            # Padding mask
            if col in self.img_cols: revert_padding_mask = torch.ones(static_data.shape[:-1]).to(device)
            elif col in self.nlp_cols: 
                revert_padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]
            # Obtain masking idx
            global_token = static_data[:, :1, :]; global_padding_mask = revert_padding_mask[:, :1]
            valid_token = static_data[:, 1:, :]; valid_padding_mask = revert_padding_mask[:, 1:]
            remain_idx, masked_idx, revert_idx = get_indices(valid_token.shape[:-1], int(valid_token.shape[1]*remain_rto))
            remain_idx, masked_idx, revert_idx = remain_idx.to(device), masked_idx.to(device), revert_idx.to(device)
            
            # Apply masking
            remain_token = torch.gather(valid_token, index=remain_idx.unsqueeze(-1).expand(-1, -1, valid_token.shape[-1]), dim=1)
            remain_padding_mask = torch.gather(valid_padding_mask, index=remain_idx, dim=1)
            masked_padding_mask = torch.gather(valid_padding_mask, index=masked_idx, dim=1)

            static_remain = torch.cat([global_token, remain_token], dim=1)
            remain_padding_mask = torch.cat([global_padding_mask, remain_padding_mask], dim=1)
            masked_padding_mask = torch.cat([global_padding_mask, masked_padding_mask], dim=1)
            
            assert remain_token.shape[1]+1 == remain_padding_mask.shape[1], f"remain_token: {remain_token.shape} != remain_padding_mask: {remain_padding_mask.shape}"
            assert remain_idx.shape[1]+1 == remain_padding_mask.shape[1]
            assert masked_idx.shape[1]+1 == masked_padding_mask.shape[1]
            assert revert_idx.shape[1]+1 == revert_padding_mask.shape[1], f"revert_idx: {revert_idx.shape}, revert_padding_mask: {revert_padding_mask.shape}"

            remain_dict.update({f"{col}_remain":static_remain})
            idx_dict.update({f"{col}_masked_idx":masked_idx, f"{col}_revert_idx":revert_idx})
            padding_mask_dict.update({f"{col}_remain_padding_mask":remain_padding_mask, f"{col}_masked_padding_mask":masked_padding_mask, f"{col}_revert_padding_mask":revert_padding_mask})
            
        return remain_dict, idx_dict, padding_mask_dict


class IndividualEncodingLayer(torch.nn.Module):
    def __init__(self, col, d_model, d_ff, nhead, dropout, activation):
        super().__init__()
        if col == "temporal": self.self_attn = MultiheadBlockAttention(d_model, nhead)
        else: self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
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

        return x, attn_output

    def _sa_block(self, src, src_key_padding_mask):
        attn_output, attn_weight = self.self_attn(src, src, src, src_key_padding_mask)
        return self.sa_dropout(attn_output), attn_weight
    
    def _ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(x)

class IndividualEncoding(torch.nn.Module):
    def __init__(self, temporal_cols, img_cols, nlp_cols, d_model, d_ff, nhead, dropout, activation, num_layers):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols
        self.layers_dict = torch.nn.ModuleDict()
        
        self.layers_dict["temporal"] = torch.nn.ModuleList([IndividualEncodingLayer("temporal", d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
        for col in img_cols + nlp_cols:
            self.layers_dict[col] = torch.nn.ModuleList([IndividualEncodingLayer(col, d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, data_dict, padding_mask_dict, mode="remain"):
        attn_output_dict, attn_weight_dict = {}, {}

        # Temporal
        x = data_dict[f"temporal_{mode}_block"]
        for mod in self.layers_dict["temporal"]:
            x, attn_weight = mod(x, None)
        attn_output_dict["temporal"] = x
        attn_weight_dict["temporal"] = attn_weight
        
        # Static
        for key in self.img_cols + self.nlp_cols:
            x = data_dict[f"{key}_{mode}"]
            padding_mask = padding_mask_dict[f"{key}_{mode}_padding_mask"]
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
    def __init__(self, tempral_cols, img_cols, nlp_cols, modality_embedding, d_model, d_ff, nhead, dropout, activation, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([TotalEncodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
        self.pos_enc = PositionalEncoding(d_model, dropout)

        self.modality_embedding = modality_embedding

        self.norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, indiv_encoding_dict, padding_mask_dict, device, mode="remain"):
        data_li, padding_mask_li = [], []
        for modality_idx, (key, val) in enumerate(indiv_encoding_dict.items()):
            if key == "temporal":
                # val = self.pos_enc(val[:, :, 0, :])
                val = val[:, :, 0, :]
                padding_mask_li.append(padding_mask_dict[f"{key}_padding_mask"])
            else:
                padding_mask_li.append(padding_mask_dict[f"{key}_{mode}_padding_mask"])
            
            # Modality embedding
            modality = torch.zeros(1, val.shape[1]).to(torch.int).to(device) + modality_idx
            modality = self.modality_embedding(modality).expand(val.shape[0], -1, -1)
                
            data_li.append(val + modality)
        
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
        for key, val in indiv_encoding_dict.items():
            length = val.shape[1]
            result_dict[key] = x[:, start_idx:start_idx+length, :]
            attn_weight_dict[key] = attn_weight[:, start_idx:start_idx+length, :]
            start_idx += length
        assert x.shape[1] == start_idx

        return result_dict, attn_weight_dict


class Revert(torch.nn.Module):
    def __init__(self, img_cols, nlp_cols, mask_token, temporal_modality_embedding, patch_size, d_model, dropout):
        super().__init__()
        self.img_cols, self.nlp_cols = img_cols, nlp_cols
        self.mask_token = mask_token

        # Temporal
        self.temporal_modality_embedding = temporal_modality_embedding
        # Img
        self.pos_enc_2d = torch.nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, grid_size=224//patch_size, cls_token=True)), requires_grad=False)
        # Nlp
        self.pos_enc = PositionalEncoding(d_model, dropout)
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, device):
        result_dict = {}
        temporal_revert_block = self.get_temporal_revert(data_dict["temporal"], idx_dict, device)
        static_revert_dict = self.get_static_revert(data_dict, idx_dict, padding_mask_dict)

        result_dict.update({"temporal_revert_block": temporal_revert_block})
        result_dict.update(static_revert_dict)

        return result_dict
    
    def get_temporal_revert(self, temporal_block, idx_dict, device):
        global_seq = temporal_block[:, :, :1, :]
        valid_seq = temporal_block[:, :, 1:, :]

        # Append mask token
        batch_size, seq_len, num_modality, d_model = valid_seq.shape
        revert_idx = idx_dict["temporal_revert_idx"]
        mask_token =self.mask_token.unsqueeze(1).expand(valid_seq.shape[0], valid_seq.shape[1], revert_idx.shape[-1]-num_modality, -1)
        full_seq = torch.cat([valid_seq, mask_token], dim=-2)

        # Apply mask token
        revert_seq = torch.gather(full_seq, index=revert_idx.unsqueeze(-1).expand(-1, -1, -1, d_model), dim=-2)
        temporal_revert_block = torch.cat([global_seq, revert_seq], dim=-2)

        # Temporal modality embedding
        for modality_idx in range(num_modality+1):
            val = temporal_revert_block[:, :, modality_idx, :]
            modality = torch.zeros(1, val.shape[1]).to(torch.int).to(device) + modality_idx
            modality = self.temporal_modality_embedding(modality).expand(val.shape[0], -1, -1)

            temporal_revert_block[:, :, modality_idx, :] = val + modality

        return temporal_revert_block
        
    def get_static_revert(self, data_dict, idx_dict, padding_mask_dict):
        result_dict = {}
        for col in self.img_cols + self.nlp_cols:
            data = data_dict[col]

            # Replace paddings to mask token
            remain_padding_mask = padding_mask_dict[f"{col}_remain_padding_mask"].unsqueeze(-1).expand(-1, -1, data.shape[-1])
            data = torch.where(remain_padding_mask==1, data, self.mask_token)
        
            # Split global token
            global_token = data[:, :1, :]
            valid_token = data[:, 1:, :]

            # Append mask token
            batch_size, seq_len, d_model = valid_token.shape
            revert_idx = idx_dict[f"{col}_revert_idx"]
            mask_token = self.mask_token.expand(batch_size, revert_idx.shape[-1]-seq_len, -1)
            valid_token = torch.cat([valid_token, mask_token], dim=1)
            assert valid_token.shape[:-1] == revert_idx.shape

            # Apply revert
            reverted_data = torch.gather(valid_token, index=revert_idx.unsqueeze(-1).expand(-1, -1, valid_token.shape[-1]), dim=1)
            static_revert = torch.cat([global_token, reverted_data], dim=1)

            # Positional encoding
            if col in self.img_cols: static_revert += self.pos_enc_2d
            elif col in self.nlp_cols: static_revert = self.pos_enc(static_revert)
            
            result_dict[f"{col}_revert"] = static_revert
        
        return result_dict


class TotalDecoderLayer(torch.nn.Module):
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
    def __init__(self, temporal_cols, img_cols, nlp_cols, modality_embedding, d_model, d_ff, nhead, dropout, activation, num_layers):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols
        self.decoder_layer_dict = torch.nn.ModuleDict()
        for col in temporal_cols + img_cols + nlp_cols:
            self.decoder_layer_dict[col] = torch.nn.ModuleList([TotalDecoderLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])

        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.modality_embedding = modality_embedding
        self.norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, indiv_encoding_dict, padding_mask_dict, device, mode="remain"):
        result_dict, sa_weight_dict, ca_weight_dict = {}, {}, {}
        # Pos mod embedding
        for modality_idx, (key, val) in enumerate(indiv_encoding_dict.items()):
            if key == "temporal":
                val = self.pos_enc(val[:, :, 0, :])
            
            modality = torch.zeros(1, val.shape[1]).to(torch.int).to(device) + modality_idx
            modality = self.modality_embedding(modality).expand(val.shape[0], -1, -1)
            indiv_encoding_dict[key] = val + modality
        
        # Temporal decoding
        for col in self.temporal_cols:
            tgt = indiv_encoding_dict["temporal"]
            tgt_padding_mask = padding_mask_dict["temporal_padding_mask"]
            tgt_padding_mask = torch.where(tgt_padding_mask==1, 0, -torch.inf)
            
            memory = torch.cat([indiv_encoding_dict[col] for col in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.cat([padding_mask_dict["temporal_padding_mask"] if col == "temporal" else padding_mask_dict[f"{col}_revert_padding_mask"] for col in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.where(memory_padding_mask==1, 0, -torch.inf)

            for mod in self.decoder_layer_dict[col]:
                tgt, sa_weight, ca_weight = mod(tgt, memory, tgt_padding_mask, memory_padding_mask)

            result_dict[col] = tgt; sa_weight_dict[col] = sa_weight; ca_weight_dict[col] = ca_weight

        # Static decoding
        for col in self.img_cols + self.nlp_cols:
            tgt = indiv_encoding_dict[col]
            tgt_padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]
            tgt_padding_mask = torch.where(tgt_padding_mask==1, 0, -torch.inf)

            memory = torch.cat([indiv_encoding_dict[col] for col in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.cat([padding_mask_dict["temporal_padding_mask"] if col == "temporal" else padding_mask_dict[f"{col}_revert_padding_mask"] for col in ["temporal"]+self.img_cols+self.nlp_cols], dim=1)
            memory_padding_mask = torch.where(memory_padding_mask==1, 0, -torch.inf)

            for mod in self.decoder_layer_dict[col]:
                tgt, sa_weight, ca_weight = mod(tgt, memory, tgt_padding_mask, memory_padding_mask)

            result_dict[col] = tgt; sa_weight_dict[col] = sa_weight; ca_weight_dict[col] = ca_weight
        
        return result_dict, ca_weight_dict

class Output(torch.nn.Module):
    def __init__(self, config, temporal_cols, img_cols, nlp_cols, scaling_cols, embedding_cols, label_encoder_dict, patch_size, d_model):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols
        self.output = torch.nn.ModuleDict()

        # Temporal
        for col in temporal_cols:
            if col in scaling_cols:
                self.output[col] = torch.nn.Sequential(
                                                        torch.nn.LayerNorm(d_model), 
                                                        # torch.nn.Linear(d_model, d_model),
                                                        # torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(config.d_model["decoder"], config.nhead["decoder"], config.d_ff["decoder"], config.dropout, config.activation, batch_first=True, norm_first=True), config.num_layers["output"]),
                                                        torch.nn.Linear(d_model, d_model),
                                                        torch.nn.Linear(d_model, 1))
            elif col in embedding_cols:
                num_cls = label_encoder_dict[col].get_num_cls()
                self.output[col] = torch.nn.Sequential(
                                                        torch.nn.LayerNorm(d_model), 
                                                        # torch.nn.Linear(d_model, d_model),
                                                        # torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(config.d_model["decoder"], config.nhead["decoder"], config.d_ff["decoder"], config.dropout, config.activation, batch_first=True, norm_first=True), config.num_layers["output"]),
                                                        torch.nn.Linear(d_model, d_model),
                                                        torch.nn.Linear(d_model, num_cls))
        
        # Img
        for col in img_cols:
            self.output[col] = torch.nn.Sequential(
                                                    torch.nn.LayerNorm(d_model), 
                                                    # torch.nn.Linear(d_model, d_model),
                                                    # torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(config.d_model["decoder"], config.nhead["decoder"], config.d_ff["decoder"], config.dropout, config.activation, batch_first=True, norm_first=True), config.num_layers["output"]),
                                                    torch.nn.Linear(d_model, d_model),
                                                    torch.nn.Linear(d_model, 3*patch_size*patch_size))
        # Nlp
        for col in nlp_cols:
            self.output[col] = torch.nn.Sequential(
                                                    torch.nn.LayerNorm(d_model), 
                                                    # torch.nn.Linear(d_model, d_model),
                                                    # torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(config.d_model["decoder"], config.nhead["decoder"], config.d_ff["decoder"], config.dropout, config.activation, batch_first=True, norm_first=True), config.num_layers["output"]),
                                                    torch.nn.Linear(d_model, d_model),
                                                    torch.nn.Linear(d_model, 30522))
    
    def forward(self, data_dict):
        result_dict = {}
        for col in self.temporal_cols:
            data = data_dict[col]
            result_dict[col] = self.output[col](data)

        for col in self.img_cols + self.nlp_cols:
            data = data_dict[col]
            result_dict[col] = self.output[col](data)
        
        return result_dict


class MBAEEncoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, global_token, mask_token, temporal_modality_embedding, total_modality_embedding):
        super().__init__()
        self.config = config

        # 1. Embedding
        self.embedding = Embedding(config, label_encoder_dict, global_token, config.d_model["encoder"], config.patch_size)
        # 2. Remain masking
        self.remain_masking = RemainMasking(["global"]+config.temporal_cols, config.img_cols, config.nlp_cols, temporal_modality_embedding, total_modality_embedding, global_token, config.remain_rto, config.patch_size, config.d_model["encoder"], config.dropout)
        # 3. Individual encoding
        self.indiv_encoding = IndividualEncoding(config.temporal_cols, config.img_cols, config.nlp_cols, config.d_model["encoder"], config.d_ff["encoder"], config.nhead["encoder"], config.dropout, config.activation, config.num_layers["encoder"])
        # 4. Total encoding
        self.total_encoding = TotalEncoding(config.temporal_cols, config.img_cols, config.nlp_cols, total_modality_embedding, config.d_model["encoder"], config.d_ff["encoder"], config.nhead["encoder"], config.dropout, config.activation, config.num_layers["encoder"])
        # 6. Revert
        self.revert = Revert(config.img_cols, config.nlp_cols, mask_token, temporal_modality_embedding, config.patch_size, config.d_model["encoder"], config.dropout)
    
    def forward(self, data_dict, idx_dict, padding_mask_dict, device):
        # 1. Embedding
        embedding_dict = self.embedding(data_dict)
        # 2. Remain masking
        remain_dict, idx_dict, padding_mask_dict = self.remain_masking(embedding_dict, idx_dict, padding_mask_dict, device)
        # 3. Individual encoding
        indiv_encoding_dict, indiv_attn_weight_dict = self.indiv_encoding(remain_dict, padding_mask_dict)
        # 4. Total encoding
        attn_output_dict, attn_weight_dict = self.total_encoding(indiv_encoding_dict, padding_mask_dict, device)
        # 5. Update temporal block
        temporal_block = indiv_encoding_dict["temporal"]
        temporal_block[:, :, 0, :] = attn_output_dict["temporal"]
        attn_output_dict["temporal"] = temporal_block
        # 6. Revert
        revert_dict = self.revert(attn_output_dict, idx_dict, padding_mask_dict, device)

        return revert_dict, attn_weight_dict, idx_dict, padding_mask_dict

class MBAEDecoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict, global_token, mask_token, temporal_modality_embedding, modality_embedding):
        super().__init__()
        # 1. To decoder dimension
        self.linear = torch.nn.Linear(config.d_model["encoder"], config.d_model["decoder"])
        # 2. Individual encoding
        self.indiv_encoding = IndividualEncoding(config.temporal_cols, config.img_cols, config.nlp_cols, config.d_model["decoder"], config.d_ff["decoder"], config.nhead["decoder"], config.dropout, config.activation, config.num_layers["decoder"])
        # 3. Total decoding
        self.total_decoding = TotalDecoding(config.temporal_cols, config.img_cols, config.nlp_cols, modality_embedding, config.d_model["decoder"], config.d_ff["decoder"], config.nhead["decoder"], config.dropout, config.activation, config.num_layers["decoder"])
        # 4. Output
        self.output = Output(config, config.temporal_cols, config.img_cols, config.nlp_cols, config.scaling_cols, config.embedding_cols, label_encoder_dict, config.patch_size, config.d_model["decoder"])
    
    def forward(self, encoding_dict, idx_dict, padding_mask_dict, device):
        # 1. To decoder dimenstion
        linear_dict = {key:self.linear(val) for key, val in encoding_dict.items()}
        # 2. Individual encoding
        indiv_encoding_dict, indiv_attn_weight_dict = self.indiv_encoding(linear_dict, padding_mask_dict, mode="revert")
        # 3. Total encoding
        attn_output_dict, attn_weight_dict = self.total_decoding(indiv_encoding_dict, padding_mask_dict, device, mode="revert")
        # 4. Output
        output_dict = self.output(attn_output_dict)
        return output_dict, attn_weight_dict

class MaskedBlockAutoencoder(torch.nn.Module):
    def __init__(self, config, label_encoder_dict):
        super().__init__()
        self.config = config
        
        global_token = torch.nn.Parameter(torch.rand(1, 1, config.d_model["encoder"]))
        mask_token = torch.nn.Parameter(torch.rand(1, 1, config.d_model["encoder"]))
        
        num_temporal_modality = len(config.temporal_cols + config.img_cols + config.nlp_cols)
        temporal_modality_embedding = torch.nn.Embedding(num_temporal_modality, config.d_model["encoder"])

        num_total_modality = len(["tity_embedding emporal"] + config.img_cols + config.nlp_cols)
        total_modal= torch.nn.Embedding(num_total_modality, config.d_model["encoder"])

        self.encoder = MBAEEncoder(config, label_encoder_dict, global_token, mask_token, temporal_modality_embedding, total_modality_embedding)
        self.decoder = MBAEDecoder(config, label_encoder_dict, global_token, mask_token, temporal_modality_embedding, total_modality_embedding)
    
    def forward(self, data_input, device):
        data_dict, idx_dict, padding_mask_dict = self.to_gpu(data_input, device)
        
        encoding_dict, encoding_weight_dict, idx_dict, padding_mask_dict = self.encoder(data_dict, idx_dict, padding_mask_dict, device)
        decoding_output_dict, decoding_weight_dict= self.decoder(encoding_dict, idx_dict, padding_mask_dict, device)
        return decoding_output_dict, encoding_weight_dict, decoding_weight_dict, idx_dict, padding_mask_dict
    
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

1==1