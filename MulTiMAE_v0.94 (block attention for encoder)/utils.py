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
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.permute(x, (1,0,2))
        x = x + self.pe[:x.size(0)]
        x = torch.permute(x, (1,0,2))
        return self.dropout(x)

class MultiheadBlockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.nhead = nhead

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
        Q = Q.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)
        K = K.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)
        V = V.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0, 3, 1, 2, 4)

        # Scaled dot product attention
        ### 1. Q·K^t
        QK = Q @ K.permute(0,1,2,4,3)

        ### 2. Softmax
        attn_weight = torch.nn.functional.softmax(QK / math.sqrt(d_model//self.nhead), dim=-1)

        ### 3. Matmul V
        attn_output = attn_weight @ V

        ### 4. Concat heads
        attn_output = attn_output.permute(0,2,3,1,4).reshape(batch_size, seq_len, -1, d_model)

        return attn_output, attn_weight

1==1