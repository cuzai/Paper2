import math
import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input = [batch, seq_len, d_model]
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.permute(x, (1,0,2))
        x = x + self.pe[:x.size(0)]
        x = torch.permute(x, (1,0,2))
        return self.dropout(x)

class MultiheadBlockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead

        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        self.output_linear = torch.nn.Linear(d_model, d_model)
    
    def forward(self, query_dict, key_dict, value_dict, key_padding_mask_dict):
        attn_output_dict, attn_weight_dict = {}, {}

        if "temporal" in query_dict.keys(): 
            attn_output_dict["temporal"], attn_weight_dict["temporal"] = self.temporal_side_attention(query_dict["temporal"], key_dict, value_dict, key_padding_mask_dict)
        
        if "static" in query_dict.keys():
            attn_output_dict["static"], attn_weight_dict["static"] = self.static_side_attention(query_dict["static"], key_dict, value_dict, key_padding_mask_dict)
        
        return attn_output_dict, attn_weight_dict

    
    def temporal_side_attention(self, query, key_dict, value_dict, key_padding_mask_dict):
        # Compute attention
        if "temporal" in key_dict.keys():
            modality = "temporal"
            Qt_Kt, Vt = self.temp_temp(query, key_dict[modality], value_dict[modality], key_padding_mask_dict[modality])
            QK, V = Qt_Kt, Vt
        
        if "static" in key_dict.keys():
            modality = "static"
            Qt_Ks, Vs = self.temp_static(query, key_dict[modality], value_dict[modality], key_padding_mask_dict[modality])
            QK, V = Qt_Ks, Vs
        
        if "temporal" in key_dict.keys() and "static" in key_dict.keys():
            QK = torch.cat([Qt_Kt, Qt_Ks], dim=-1)
            temporal_attn_weight = torch.nn.functional.softmax(QK, dim=-1)
            assert torch.isnan(temporal_attn_weight).sum() == 0

            Qt_Kt_weight = temporal_attn_weight[:, :, :, :, :Qt_Kt.shape[-1]]
            Qt_Ks_weight = temporal_attn_weight[:, :, :, :, Qt_Kt.shape[-1]:]

            # Qt_Kt_weight = torch.nn.functional.softmax(Qt_Kt, dim=-1)
            # Qt_Kt_weight = torch.where(torch.isnan(Qt_Kt_weight), 0, Qt_Kt_weight)
            # assert torch.isnan(Qt_Kt_weight).sum() == 0
            # Qt_Ks_weight = torch.nn.functional.softmax(Qt_Ks, dim=-1)
            # assert torch.isnan(Qt_Ks_weight).sum() == 0
            # temporal_attn_weight = torch.cat([Qt_Kt_weight, Qt_Ks_weight], dim=-1)

            Qt_Kt_Vt = Qt_Kt_weight @ Vt
            Qt_Ks_Vs = Qt_Ks_weight @ Vs

            QKV = Qt_Kt_Vt + Qt_Ks_Vs
        else:
            temporal_attn_weight = torch.nn.functional.softmax(QK, dim=-1)
            assert torch.isnan(temporal_attn_weight).sum() == 0
            QKV = temporal_attn_weight @ V
        
        # Concat head
        batch_size, nhead, seq_len, num_modality, _ = QKV.shape
        temporal_attn_output = QKV.permute(0,2,3,1,4).reshape(batch_size, seq_len, num_modality, -1)
        temporal_attn_output = self.output_linear(temporal_attn_output)
        
        return temporal_attn_output, temporal_attn_weight

    def static_side_attention(self, query, key_dict, value_dict, key_padding_mask_dict):
        # Compute attention
        if "temporal" in key_dict.keys():
            modality = "temporal"
            Qs_Kt, Vt = self.static_temp(query, key_dict[modality], value_dict[modality], key_padding_mask_dict[modality])
            QK, V = Qs_Kt, Vt
        if "static" in key_dict.keys():
            modality = "static"
            Qs_Ks, Vs = self.static_static(query, key_dict[modality], value_dict[modality], key_padding_mask_dict[modality])
            QK, V = Qs_Ks, Vs
        
        if "temporal" in key_dict.keys() and "static" in key_dict.keys():
            QK = torch.cat([Qs_Kt, Qs_Ks], dim=-1)
            static_attn_weight = torch.nn.functional.softmax(QK, dim=-1)
            assert torch.isnan(static_attn_weight).sum() == 0

            Qs_Kt_weight = static_attn_weight[:, :, :, :Qs_Kt.shape[-1]]
            Qs_Ks_weight = static_attn_weight[:, :, :, Qs_Kt.shape[-1]:]
            
            # Qs_Kt_weight = torch.nn.functional.softmax(Qs_Kt, dim=-1)
            # Qs_Kt_weight = torch.where(torch.isnan(Qs_Kt_weight), 0, Qs_Kt_weight)
            # assert torch.isnan(Qs_Kt_weight).sum() == 0
            # Qs_Ks_weight = torch.nn.functional.softmax(Qs_Ks, dim=-1)
            # assert torch.isnan(Qs_Ks_weight).sum() == 0
            # static_attn_weight = torch.cat([Qs_Kt_weight, Qs_Ks_weight], dim=-1)

            Qs_Kt_Vt = Qs_Kt_weight @ Vt
            Qs_Ks_Vs = Qs_Ks_weight @ Vs

            QKV = Qs_Kt_Vt + Qs_Ks_Vs
        else:
            static_attn_weight = torch.nn.functional.softmax(QK, dim=-1)
            assert torch.isnan(static_attn_weight).sum() == 0
            QKV = static_attn_weight @ V
        
        # Concat head
        batch_size, nhead, seq_len, _ = QKV.shape
        static_attn_output = QKV.permute(0,2,1,3).reshape(batch_size, seq_len, -1)
        static_attn_output = self.output_linear(static_attn_output)

        return static_attn_output, static_attn_weight


    
    def temp_temp(self, query, key, value, padding_mask):
        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split head
        batch_size, Q_seq_len, Q_num_modality, d_model = Q.shape
        batch_size, K_seq_len, K_num_modality, d_model = K.shape
        Q = Q.view(batch_size, Q_seq_len, Q_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        K = K.view(batch_size, K_seq_len, K_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        V = V.view(batch_size, K_seq_len, K_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)

        # Scaled dot product attention
        QK = Q @ K.permute(0,1,2,4,3)
        logits = QK / math.sqrt(d_model//self.nhead)

        # Padding mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(-2)
        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
        logits += padding_mask
        
        return logits, V

    def temp_static(self, query, key, value, padding_mask):
        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key.unsqueeze(1))
        V = self.v_linear(value.unsqueeze(1))

        # Split head
        batch_size, Q_seq_len, Q_num_modality, d_model = Q.shape
        batch_size, K_seq_len, K_num_modality, d_model = K.shape
        Q = Q.view(batch_size, Q_seq_len, Q_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        K = K.view(batch_size, K_seq_len, K_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        V = V.view(batch_size, K_seq_len, K_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)

        # Scaled dot product attention
        QK = Q @ K.permute(0,1,2,4,3)
        logits = QK / math.sqrt(d_model//self.nhead)
        
        # Padding mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
        logits += padding_mask
        
        return logits, V

    def static_temp(self, query, key, value, padding_mask):
        batch_size, seq_len, d_model = query.shape

        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key.view(batch_size, -1, d_model))
        V = self.v_linear(value.view(batch_size, -1, d_model))

        # Split head
        Q = Q.view(batch_size, seq_len, self.nhead, d_model//self.nhead).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.nhead, d_model//self.nhead).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.nhead, d_model//self.nhead).permute(0,2,1,3)
        
        # Scaled dot product attention
        QK = Q @ K.permute(0,1,3,2)
        logits = QK / math.sqrt(d_model//self.nhead)

        # Padding mask
        padding_mask = padding_mask.view(batch_size, -1)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
        logits += padding_mask

        return logits, V

    def static_static(self, query, key, value, padding_mask):
        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split head
        batch_size, _, d_model = Q.shape
        Q = Q.view(batch_size, -1, self.nhead, d_model//self.nhead).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.nhead, d_model//self.nhead).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.nhead, d_model//self.nhead).permute(0,2,1,3)

        # Scaled dot product attention
        QK = Q @ K.permute(0,1,3,2)
        logits = QK / math.sqrt(d_model//self.nhead)

        # Padding mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
        logits += padding_mask

        return logits, V

class PosModEmb(torch.nn.Module):
    def __init__(self, d_model, dropout, temporal_cols, img_cols, nlp_cols):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols
        assert "global" in self.temporal_cols
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        # self.temporal_pos_enc = torch.nn.Parameter(torch.zeros(1, 1000, d_model))
        # self.img_pos_enc = torch.nn.Parameter(torch.zeros(1, 14*14+1, d_model))
        # self.nlp_pos_enc = torch.nn.Parameter(torch.zeros(1, 1000, d_model))

        self.num_modality = len(temporal_cols + img_cols + nlp_cols)
        self.modality_embedding = torch.nn.Embedding(self.num_modality, d_model)

    def forward(self, data_dict, device):
        result_dict = {}
        assert self.num_modality == len(data_dict)

        for modality_idx, (key, val) in enumerate(data_dict.items()):
            # Positional encoding
            data = self.pos_enc(val)
            # data = val
            # if key in self.temporal_cols:
            #     data += self.temporal_pos_enc[:, :data.shape[1], :]
            # elif key in self.img_cols:
            #     data += self.img_pos_enc
            # elif key in self.nlp_cols:
            #     data += self.nlp_pos_enc[:, :data.shape[1], :]

            # Modality embedding
            seq_len = data.shape[1]
            modality = torch.zeros(seq_len).to(torch.int).to(device) + modality_idx
            modality = self.modality_embedding(modality).unsqueeze(0)
            
            result_dict[key] = data + modality
        
        return result_dict


1==1