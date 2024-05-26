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

class MultiheadBlockSelfAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.nhead = nhead

        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

        self.static_attention = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

    def forward(self, query, key, value, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask):
        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        temporal_attn_output, temporal_attn_weight = self.temporal_side_attention(Q, K, V, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask)
        static_attn_output, static_attn_weight = self.static_side_attention(Q, K, V, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask)

        # Concat all
        temporal_attn_output = temporal_attn_output.view(temporal_attn_output.shape[0], -1, temporal_attn_output.shape[-1])
        attn_output = torch.cat([temporal_attn_output, static_attn_output], dim=1)

        return attn_output

    def temporal_side_attention(self, Q, K, V, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask):
        # Obtain QKV
        temporal_idx = temporal_idx.unsqueeze(0).unsqueeze(-1).repeat(Q.shape[0], 1, Q.shape[-1])
        static_idx = static_idx.unsqueeze(0).unsqueeze(-1).repeat(Q.shape[0], 1, Q.shape[-1])

        Qt = torch.gather(Q, index=temporal_idx, dim=1).view(temporal_shape)

        Ktt = torch.gather(K, index=temporal_idx, dim=1).view(temporal_shape)
        Kts = torch.gather(K, index=static_idx, dim=1).unsqueeze(1).repeat(1, Ktt.shape[1], 1, 1)
        Kt = torch.cat([Ktt, Kts], dim=-2)

        Vtt = torch.gather(V, index=temporal_idx, dim=1).view(temporal_shape)
        Vts = torch.gather(V, index=static_idx, dim=1).unsqueeze(1).repeat(1, Vtt.shape[1], 1, 1)
        Vt = torch.cat([Vtt, Vts], dim=-2)

        # Obtain key padding mask
        static_padding_mask = static_padding_mask.unsqueeze(1).repeat(1, temporal_padding_mask.shape[1], 1)

        key_padding_mask = torch.cat([temporal_padding_mask, static_padding_mask], dim=-1)
        key_padding_mask = torch.where(key_padding_mask==1, 0, -torch.inf)

        # Attention
        temporal_attn_output, temporal_attn_weight = self.multihead_block_attention(Qt, Kt, Vt, key_padding_mask=key_padding_mask)
        return temporal_attn_output, temporal_attn_weight
    
    def static_side_attention(self, Q, K, V, temporal_shape, temporal_idx, static_idx, temporal_padding_mask, static_padding_mask):
        # Obtaion QKV
        temporal_idx = temporal_idx.unsqueeze(0).unsqueeze(-1).repeat(Q.shape[0], 1, Q.shape[-1])
        static_idx = static_idx.unsqueeze(0).unsqueeze(-1).repeat(Q.shape[0], 1, Q.shape[-1])

        Qs = torch.gather(Q, index=static_idx, dim=1)
        
        Kst = torch.gather(K, index=temporal_idx, dim=1)
        Kss = torch.gather(K, index=static_idx, dim=1)
        Ks = torch.cat([Kst, Kss], dim=1)

        Vst = torch.gather(V, index=temporal_idx, dim=1)
        Vss = torch.gather(V, index=static_idx, dim=1)
        Vs = torch.cat([Vst, Vss], dim=1)

        # Obtain key padding mask
        temporal_padding_mask = temporal_padding_mask.view(temporal_shape[0], -1)
        static_padding_mask = static_padding_mask
        
        key_padding_mask = torch.cat([temporal_padding_mask, static_padding_mask], dim=1)
        key_padding_mask = torch.where(key_padding_mask==1, 0, -torch.inf)

        # Attention
        static_attn_output, static_attn_weight = self.static_attention(Qs, Ks, Vs, key_padding_mask=key_padding_mask)

        return static_attn_output, static_attn_weight 


    def multihead_block_attention(self, Q, K, V, key_padding_mask):
        # Split head
        batch_size, seq_len, _, d_model = Q.shape
        Q = Q.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)
        K = K.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)
        V = V.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)

        # Scaled dot product attention
        ### 1. Q·K^t
        QK = Q @ K.permute(0,1,2,4,3)
        logits = QK / math.sqrt(d_model//self.nhead)
        
        ### #. Padding_mask
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(-2).repeat(1, logits.shape[1], 1, logits.shape[-2], 1)
        logits += key_padding_mask

        ### 2. Softmax
        attn_weight = torch.nn.functional.softmax(logits, dim=-1)

        ### 3. Matmul V
        attn_output = attn_weight @ V

        ### 4. Concat heads
        attn_output = attn_output.permute(0,2,3,1,4).reshape(batch_size, seq_len, -1, d_model)

        return attn_output, attn_weight

class MultiheadBlockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.nhead = nhead

        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, query, key, value, padding_mask):
        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split head
        batch_size, seq_len, _, d_model = Q.shape
        Q = Q.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)
        K = K.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)
        V = V.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)

        # Scaled dot product attention
        ### 1. Q·K^t
        QK = Q @ K.permute(0,1,2,4,3)
        logits = QK / math.sqrt(d_model//self.nhead)
        
        ### #. Padding_mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(-2).repeat(1, logits.shape[1], 1, logits.shape[-2], 1)
        logits += padding_mask

        ### 2. Softmax
        attn_weight = torch.nn.functional.softmax(logits, dim=-1)

        ### 3. Matmul V
        attn_output = attn_weight @ V

        ### 4. Concat heads
        attn_output = attn_output.permute(0,2,3,1,4).reshape(batch_size, seq_len, -1, d_model)

        return attn_output, attn_weight

1==1