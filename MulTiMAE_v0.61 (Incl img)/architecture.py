import torch
import copy

class Transformer(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict,
                d_model, num_layers, nhead, d_ff, dropout, activation):
        super().__init__()
        self.data_info, self.label_encoder_dict = data_info, label_encoder_dict
        self.temporal_cols = data_info.modality_info["target"] + data_info.modality_info["temporal"]
        self.total_cols = self.temporal_cols

        # 1. Embedding
        self.embedding_dict = torch.nn.ModuleDict({col:Embedding(col, self.data_info, self.label_encoder_dict, d_model["encoder"]) for col in self.total_cols})

        # 2-1. Apply temporal remain
        remain_pos_enc = torch.nn.Parameter(get_positional_encoding(d_model["encoder"]), requires_grad=False)
        self.temporal_remain_dict = torch.nn.ModuleDict({col:TemporalRemain(remain_pos_enc) for col in self.temporal_cols})

        # 3. Apply modality embedding
        num_modality = len(self.total_cols)
        self.encoder_modality_embedding = ModalityEmbedding(num_modality, d_model["encoder"])

        # 4. Encoding
        self.encoder = Encoder(d_model["encoder"], nhead, d_ff["encoder"], dropout, activation, num_layers["encoder"])
        self.to_decoder_dim = torch.nn.Linear(d_model["encoder"], d_model["decoder"])

        # 5. Split
        self.split_dict = Split()

        # 6. Revert
        mask_token = torch.nn.Parameter(torch.rand(1, d_model["decoder"]))
        revert_pos_enc = torch.nn.Parameter(get_positional_encoding(d_model["decoder"]), requires_grad=False)
        self.revert_dict = torch.nn.ModuleDict({col:Revert(mask_token, revert_pos_enc) for col in self.total_cols})

        # 7. Apply modality embedding
        self.decoder_modality_embedding = ModalityEmbedding(num_modality, d_model["decoder"])

        # 8. Decoding
        self.temporal_decoder = torch.nn.ModuleDict({col:TemporalDecoder(d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"]) for col in self.temporal_cols})

        # 9. Output
        patch_size = 16
        self.temporal_output = torch.nn.ModuleDict({col:TemporalOutput(col, self.data_info, self.label_encoder_dict, d_model["decoder"]) for col in self.temporal_cols})
    
    def forward(self, data_input, remain_rto, device):
        data_dict, idx_dict, padding_mask_dict = self._to_gpu(data_input, device)

        # 1. Embedding
        embedding_dict = {key:self.embedding_dict[key](key, val) for key, val in data_dict.items()}

        # 2-1. Apply temporal remain
        temporal_remain_dict = {key:self.temporal_remain_dict[key](key, val, idx_dict[f"{key}_remain_idx"]) for key, val in embedding_dict.items() if key in self.temporal_cols}
        remain_dict = {}
        remain_dict.update(temporal_remain_dict)

        # 3. Apply modality embedding
        remain_dict = self.encoder_modality_embedding(remain_dict, device)

        # 4. Encoding
        encoding, key_cols = self.encoder(remain_dict, padding_mask_dict, self.temporal_cols, device)
        encoding = self.to_decoder_dim(encoding)

        # 5. Split
        encoding_split_dict = self.split_dict(encoding, remain_dict, key_cols)

        # 6. Revert
        revert_dict = {key:self.revert_dict[key](key, val, padding_mask_dict, idx_dict) for key, val in encoding_split_dict.items()}

        # 7. Apply modality embedding
        revert_dict = self.decoder_modality_embedding(revert_dict, device)

        # 8. Decoding
        temporal_decoding_result = {key:self.temporal_decoder[key](key, revert_dict, padding_mask_dict, device) for key in revert_dict.keys() if key in self.temporal_cols}
        temporal_decoding_dict = {key:val["tgt"] for key, val in temporal_decoding_result.items()}
        self_attn_weight_dict = {key:val["self_attn_weight"] for key, val in temporal_decoding_result.items()}
        cross_attn_weight_dict = {key:val["cross_attn_weight"] for key, val in temporal_decoding_result.items()}

        # 9. Output
        temporal_output_dict = {key:self.temporal_output[key](key, val) for key, val in temporal_decoding_dict.items()}
        
        return temporal_output_dict, self_attn_weight_dict, cross_attn_weight_dict
    
    def _to_gpu(self, data, device):
        data_dict = {}
        idx_dict = {}
        padding_mask_dict = {}

        for col in data.keys():
            if col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
                target_dict = data_dict
            elif col.endswith("idx"):
                target_dict = idx_dict
            elif col.endswith("padding_mask"):
                target_dict = padding_mask_dict
            elif col.endswith("scaler"):
                continue
            else: raise Exception(col)
            
            target_dict[col] = data[col].to(device)
        
        return data_dict, idx_dict, padding_mask_dict

import math
def get_positional_encoding(d_model, seq_len=1000):
    position = torch.arange(seq_len).reshape(-1,1)
    i = torch.arange(d_model)//2
    exp_term = 2*i/d_model
    div_term = torch.pow(10000, exp_term).reshape(1, -1)
    pos_encoded = position / div_term

    pos_encoded[:, 0::2] = torch.sin(pos_encoded[:, 0::2])
    pos_encoded[:, 1::2] = torch.cos(pos_encoded[:, 1::2])

    return pos_encoded

class MultiheadBlockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split head
        batch_size, seq_len, _, d_model = Q.shape
        Q = Q.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        K = K.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        V = V.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)

        # Scaled dot product attention
        ### 1. Q·K^t
        QK = Q @ K.permute(0,1,2,4,3)

        ### 2. Softmax
        attn = torch.nn.functional.softmax(QK/math.sqrt(self.d_model//self.nhead), dim=-1)
        
        ### 3. Matmul V
        attn_output = attn @ V
        
        # Concat heads
        attn_output = attn_output.permute(0,2,3,1,4).reshape(batch_size, -1, seq_len, d_model)
        attn_output = attn_output.squeeze()

        return attn_output, attn

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, activation):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout()

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        if activation == "gelu":
            self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)

class Embedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        self.data_info = data_info

        # Numerical
        if col in data_info.processing_info["scaling_cols"]:
            self.embedding = torch.nn.Linear(1, d_model)
        
        # Categorical
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].get_num_cls()
            self.embedding = torch.nn.Embedding(num_cls, d_model)
        
        else: raise Exception(col)
    
    def forward(self, key, val):
        # Numerical, categorical
        if key in list(self.data_info.processing_info["scaling_cols"].keys()) + self.data_info.processing_info["embedding_cols"]:
            res = self.embedding(val)

        else: raise Exception(key)

        return res

class TemporalRemain(torch.nn.Module):
    def __init__(self, pos_enc):
        super().__init__()
        self.pos_enc = pos_enc

    def forward(self, key, val, remain_idx):
        # Positional encoding
        pos_enc = self.pos_enc[:val.shape[1], :]
        pos_enc = pos_enc.unsqueeze(0).repeat(val.shape[0], 1, 1)
        val += pos_enc

        # Apply remain
        remain_idx = remain_idx.unsqueeze(-1).repeat(1, 1, val.shape[-1])

        data_remain = torch.gather(val, index=remain_idx, dim=1)
        return data_remain

class NonTemporalRemain(torch.nn.Module):
    def __init__(self, col, pos_enc):
        super().__init__()
        self.pos_enc = pos_enc
    
    def forward(self, key, val, remain_rto, device):
        # Positional encoding
        pos_enc = self.pos_enc[:val.shape[1], :]
        pos_enc = pos_enc.unsqueeze(0).repeat(val.shape[0], 1, 1)
        val += pos_enc

        # Apply remain
        num_remain = int(val.shape[1] * remain_rto)
        noise = torch.rand(val.shape[:-1]).to(device)
        shuffle_idx = torch.argsort(noise, dim=1)

        remain_idx = shuffle_idx[:, :num_remain]
        masked_idx = shuffle_idx[:, num_remain:]
        revert_idx = torch.argsort(shuffle_idx, dim=1)

        data_remain = torch.gather(val, index=remain_idx.unsqueeze(-1).repeat(1, 1, val.shape[-1]), dim=1)

        remain_padding_mask = torch.ones(remain_idx.shape).to(device)
        masked_padding_mask = torch.ones(masked_idx.shape).to(device)
        revert_padding_mask = torch.ones(revert_idx.shape).to(device)

        return data_remain, remain_idx, masked_idx, revert_idx, remain_padding_mask, masked_padding_mask, revert_padding_mask

class ModalityEmbedding(torch.nn.Module):
    def __init__(self, num_modality, d_model):
        super().__init__()
        self.modality_embedding = torch.nn.Embedding(num_modality, d_model)
    
    def forward(self, remain_dict, device):
        result_dict = {}

        for modality_idx, (key, val) in enumerate(remain_dict.items()):
            modality_val = torch.zeros(val.shape[1]) + modality_idx
            modality = self.modality_embedding(modality_val.to(torch.int).to(device))
            result_dict[key] = val + modality

        return result_dict

class Encoder(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation=activation, norm_first=True), num_layers)
    
    def forward(self, remain_dict, padding_mask_dict, temporal_cols, device):
        # Padding mask
        padding_mask_li = []
        ### Temporal padding mask
        for col in temporal_cols:
            padding_mask_li.append(padding_mask_dict[f"{col}_remain_padding_mask"])
        
        padding_mask = torch.cat(padding_mask_li, dim=1)
        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)

        # Concat data
        key_cols = list(remain_dict.keys())
        concat = torch.cat(list(remain_dict.values()), dim=1)

        # Encoding
        encoding = self.encoder(concat, src_key_padding_mask=padding_mask)

        return encoding, key_cols

class Split(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tobe_split, original_dict, key_cols):
        result_dict = {}
        start_idx = 0
        
        for col in key_cols:
            length = original_dict[col].shape[1]
            result_dict[col] = tobe_split[:, start_idx:start_idx+length, :]
            start_idx += length
        
        assert start_idx == tobe_split.shape[1]
        return result_dict 

class Revert(torch.nn.Module):
    def __init__(self, mask_token, pos_enc):
        super().__init__()
        self.mask_token = mask_token
        self.pos_enc = pos_enc
    
    def forward(self, key, val, padding_mask_dict, idx_dict):
        # Replace remain padding to mask token
        remain_padding_mask = padding_mask_dict[f"{key}_remain_padding_mask"].unsqueeze(-1).repeat(1, 1, val.shape[-1])
        val = torch.where(remain_padding_mask==1, val, self.mask_token)

        # Append mask token
        revert_idx = idx_dict[f"{key}_revert_idx"].unsqueeze(-1).repeat(1, 1, val.shape[-1])
        mask_token = self.mask_token.unsqueeze(0).repeat(val.shape[0], revert_idx.shape[1] - val.shape[1], 1)
        val = torch.cat([val, mask_token], dim=1)
        assert val.shape == revert_idx.shape

        # Apply revert
        val = torch.gather(val, index=revert_idx, dim=1)

        # Pos enc
        pos_enc = self.pos_enc[:val.shape[1], :]
        pos_enc = pos_enc.unsqueeze(0).repeat(val.shape[0], 1, 1)
        val += pos_enc

        return val

class TemporalDecoder(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.decoder = torch.nn.ModuleList([TemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, key, revert_dict, padding_mask_dict, device):
        tgt = revert_dict[key]
        
        temporal_memory = {k:v for k, v in revert_dict.items() if k not in [key]}
        key_cols = temporal_memory.keys()

        memory = torch.stack(list(temporal_memory.values()), dim=-2)
        
        for mod in self.decoder:
            tgt, self_attn_weight, cross_attn_weight = mod(key, tgt, memory, padding_mask_dict, device)
        
        return {"tgt":tgt, "self_attn_weight":self_attn_weight, "cross_attn_weight":cross_attn_weight}

class __TemporalDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation):
        super().__init__()
        # self.cross_attn = torch.nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.cross_attn = MultiheadBlockAttention(d_model, num_heads, dropout)
        self.mlp = torch.nn.Linear(d_model, d_model)
        self.mha = torch.nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, activation)

        self.layernorm_tgt = torch.nn.LayerNorm(d_model)
        self.layernorm_mem = torch.nn.LayerNorm(d_model)

        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)

    
    def _get_padding_mask(self, padding_mask, memory, device):
        padding_mask_li = []
        for col, val in memory.items():
            padding_mask_li.append(padding_mask[f"{col}_revert_padding_mask"])
        
        result = torch.cat(padding_mask_li, dim=1).to(device)
        result = torch.where(result == 1, 0, -torch.inf)
        return result
    
    def forward(self, col, tgt, memory, padding_mask, device):
        tgt = tgt.unsqueeze(-2)
        # memory = torch.stack(list(memory.values()), dim=-2)
        
        tgt = self.layernorm_tgt(tgt)
        memory = self.layernorm_mem(memory)
        
        cross_attn, cross_attn_weight = self.cross_attn(query=tgt, key=memory, value=memory)

        cross_attn = tgt.squeeze() + cross_attn

        padding_mask = padding_mask[f"{col}_revert_padding_mask"]
        padding_mask = torch.where(padding_mask == 1, 0, -torch.inf)
        
        cross_attn = self.layernorm1(cross_attn)
        self_attn, self_attn_weight = self.mha(cross_attn, cross_attn, cross_attn, key_padding_mask=padding_mask) # multiheadattention

        self_attn = self_attn + cross_attn

        # ff
        self_attn = self.layernorm2(self_attn)
        ff = self_attn + self.ff(self_attn)
        
        return ff, self_attn_weight, cross_attn_weight

class TemporalDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation):
        super().__init__()
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)
        self.layernorm4 = torch.nn.LayerNorm(d_model)

        self.cross_attn = MultiheadBlockAttention(d_model, nhead, dropout)
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, activation)
    
    def forward(self, key, tgt, memory, padding_mask_dict, device):
        # Cross attention
        cross_attn, cross_attn_weight = self.cross_attn(query=self.layernorm1(tgt.unsqueeze(-2)), key=memory, value=memory)
        cross_attn = tgt + cross_attn

        # Self attention
        padding_mask = padding_mask_dict[f"{key}_revert_padding_mask"]
        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
        self_attn, self_attn_weight = self.self_attn(self.layernorm2(cross_attn), self.layernorm2(cross_attn), self.layernorm2(cross_attn), key_padding_mask=padding_mask)
        self_attn = cross_attn + self_attn

        # Feed forward
        ff = self.ff(self.layernorm3(self_attn))
        ff = self_attn + ff

        return ff, self_attn_weight, cross_attn_weight

class NonTemporalDecoder(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.decoder = torch.nn.ModuleList([NonTemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, key, revert_dict, padding_mask_dict, temporal_cols):
        tgt = revert_dict[key]
        memory = {k:v for k, v in revert_dict.items() if k in temporal_cols}
        memory = torch.cat(list(memory.values()), dim=1)
        
        for mod in self.decoder:
            tgt, self_attn_weight, cross_attn_weight = mod(key, tgt, memory, padding_mask_dict, temporal_cols)
        
        return {"tgt":tgt, "self_attn_weight":self_attn_weight, "cross_attn_weight":cross_attn_weight}

class NonTemporalDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation):
        super().__init__()
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)

        self.cross_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, activation)

    def forward(self, key, tgt, memory, padding_mask_dict, temporal_cols):
        # Cross attention
        padding_mask_li = []
        for col in temporal_cols:
            padding_mask_li.append(padding_mask_dict[f"{col}_revert_padding_mask"])
        padding_mask = torch.cat(padding_mask_li, dim=1)

        cross_attn, cross_attn_weight = self.cross_attn(query=self.layernorm1(tgt), key=memory, value=memory, key_padding_mask=padding_mask)
        cross_attn = tgt + cross_attn

        # Self attention
        self_attn, self_attn_weight = self.self_attn(self.layernorm2(cross_attn), self.layernorm2(cross_attn), self.layernorm2(cross_attn))
        self_attn = cross_attn + self_attn

        # Feed forward
        ff = self.ff(self.layernorm3(self_attn))
        ff = self_attn + ff

        return ff, self_attn_weight, cross_attn_weight

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
        else: raise Exception(col)
    
    def forward(self, key, val):
        return self.output(val)


1==1