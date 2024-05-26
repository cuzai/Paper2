import copy
import math
import torch
from utils import DynamicEmbedding, ImgEmbedding, OthersRemain, TemporalRemain, ImgRemain, OthersRevert, TemporalRevert, ImgRevert, DynamicOutput, ImgOutput

def get_positional_encoding(d_model, seq_len=1000):
    position = torch.arange(seq_len).reshape(-1,1)
    i = torch.arange(d_model)//2
    exp_term = 2*i/d_model
    div_term = torch.pow(10000, exp_term).reshape(1, -1)
    pos_encoded = position / div_term

    pos_encoded[:, 0::2] = torch.sin(pos_encoded[:, 0::2])
    pos_encoded[:, 1::2] = torch.cos(pos_encoded[:, 1::2])

    return pos_encoded

class Transformer(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict, patch_size,
                        d_model, num_layers, nhead, d_ff, dropout):
        super().__init__()
        activation = "gelu"
        self.data_info, self.label_encoder_dict = data_info, label_encoder_dict

        # Embedding
        self.dynamic_embedding_dict = self._init_dynamic_embedding_dict(data_info, label_encoder_dict, d_model["encoder"])
        self.img_embedding_dict = self._init_img_embedding_dict(d_model["encoder"], patch_size)

        # Remain masking
        self.others_remain_pos_emb = torch.nn.Embedding(len(self.data_info.modality_info["others"])+1, d_model["encoder"])
        self.temporal_remain_pos_enc = torch.nn.Parameter(get_positional_encoding(d_model["encoder"]), requires_grad=True)
        self.img_remain_pos_enc = torch.nn.Parameter(get_positional_encoding(d_model["encoder"]), requires_grad=True)
        
        self.others_remain = OthersRemain(d_model["encoder"])
        self.temporal_remain_dict = self._init_temporal_remain_dict(d_model["encoder"])
        self.img_remain_dict = self._init_img_remain_dict(d_model["encoder"])

        # Concat
        self.encoder_modality_emb = torch.nn.Embedding(self.data_info.num_modality, d_model["encoder"])

        # Encoding
        self.encoding = Encoder(EncoderLayer(d_model["encoder"], nhead, d_ff["encoder"], dropout, activation, batch_first=True), num_layers["encoder"])
        self.linear = torch.nn.Linear(d_model["encoder"], d_model["decoder"])

        # Revert
        self.others_revert_pos_emb = torch.nn.Embedding(len(self.data_info.modality_info["others"])+1, d_model["decoder"])
        self.temporal_revert_pos_enc = torch.nn.Parameter(get_positional_encoding(d_model["decoder"]), requires_grad=True)
        self.img_revert_pos_enc = torch.nn.Parameter(get_positional_encoding(d_model["decoder"]), requires_grad=True)
        self.decoder_modality_emb = torch.nn.Embedding(self.data_info.num_modality, d_model["decoder"])

        self.others_revert = OthersRevert(d_model["decoder"])
        self.temporal_revert_dict = self._init_temporal_revert(d_model["decoder"])
        self.img_revert_dict = self._init_img_revert(d_model["decoder"])

        # Decoding
        self.decoding = Encoder(EncoderLayer(d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, batch_first=True), num_layers["decoder"])
        # decoder = BlockEncoderLayer(d_model["decoder"], d_ff["decoder"], nhead, activation, dropout)
        # self.decoding = torch.nn.ModuleList([copy.deepcopy(decoder) for i in range(num_layers["decoder"])])

        # Output
        self.output_dict = self._init_output(label_encoder_dict, patch_size, d_model["decoder"])

    def _init_dynamic_embedding_dict(self, data_info, label_encoder_dict, d_model):
        dynamic_embedding_dict = {}
        target_cols = self.data_info.modality_info["others"] + self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        for col in target_cols:
            dynamic_embedding_dict[col] = DynamicEmbedding(col, data_info, label_encoder_dict, d_model)
        
        return torch.nn.ModuleDict(dynamic_embedding_dict)
    
    def _init_img_embedding_dict(self, d_model, patch_size):
        img_embedding_dict = {}
        target_cols = self.data_info.modality_info["img"]
        for col in target_cols:
            img_embedding_dict[f"{col}_img_input"] = ImgEmbedding(d_model, patch_size)
        
        return torch.nn.ModuleDict(img_embedding_dict)

    
    def _init_temporal_remain_dict(self, d_model):
        temporal_remain_dict = {}
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        for col in target_cols:
            temporal_remain_dict[col] = TemporalRemain(d_model)
        return torch.nn.ModuleDict(temporal_remain_dict)

    def _init_img_remain_dict(self, d_model):
        img_remain_dict = {}
        target_cols = self.data_info.modality_info["img"]
        for col in target_cols:
            img_remain_dict[f"{col}_img_input"] = ImgRemain(d_model)
        return torch.nn.ModuleDict(img_remain_dict)

   
    def _init_temporal_revert(self, d_model):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        temporal_revert_dict = {}
        for col in target_cols:
            temporal_revert_dict[col] = TemporalRevert(d_model)
        
        return torch.nn.ModuleDict(temporal_revert_dict)

    def _init_img_revert(self, d_model):
        target_cols = self.data_info.modality_info["img"]
        img_revert_dict = {}
        for col in target_cols:
            img_revert_dict[f"{col}_img_input"] = ImgRevert(d_model)
        
        return torch.nn.ModuleDict(img_revert_dict)

    def _init_output(self, label_encoder_dict, patch_size, d_model):
        output_dict = {}
        # Others and temporal
        for col in self.data_info.modality_info["others"] + self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            output_dict[col] = DynamicOutput(col, self.data_info, label_encoder_dict, d_model)

        # Img
        for col in self.data_info.modality_info["img"]:
            output_dict[f"{col}_img_input"] = ImgOutput(patch_size, d_model)
        
        return torch.nn.ModuleDict(output_dict)


    def _data_to_gpu(self, data_input_dict, device):
        data_dict = {}
        idx_dict = {}
        padding_mask_dict = {}
        # Others data
        for col in self.data_info.modality_info["others"]:
            data_dict[col] = data_input_dict[col].to(device)
        
        # Temporal data
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            data_dict[col] = data_input_dict[col].to(device)
            idx_dict[f"{col}_remain_idx"] = data_input_dict[f"{col}_remain_idx"].to(device)
            idx_dict[f"{col}_masked_idx"] = data_input_dict[f"{col}_masked_idx"].to(device)
            idx_dict[f"{col}_revert_idx"] = data_input_dict[f"{col}_revert_idx"].to(device)
        
            padding_mask_dict[f"{col}_remain_padding_mask"] = data_input_dict[f"{col}_remain_padding_mask"].to(device)
            padding_mask_dict[f"{col}_masked_padding_mask"] = data_input_dict[f"{col}_masked_padding_mask"].to(device)
            padding_mask_dict[f"{col}_revert_padding_mask"] = data_input_dict[f"{col}_revert_padding_mask"].to(device)

        # Img data
        for col in self.data_info.modality_info["img"]:
            data_dict[f"{col}_img_input"] = data_input_dict[f"{col}_img_input"].to(device)
        
        return data_dict, idx_dict, padding_mask_dict

    def _embed_dynamic(self, data_dict):
        dynamic_embedding_dict = {}
        target_cols = self.data_info.modality_info["others"] + self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        for col in target_cols:
            dynamic_embedding_dict[col] = self.dynamic_embedding_dict[col](data_dict[col])
        
        return dynamic_embedding_dict

    def _embed_img(self, data_dict):
        img_embedding_dict = {}
        target_cols = self.data_info.modality_info["img"]
        for col in target_cols:
            img_embedding_dict[f"{col}_img_input"] = self.img_embedding_dict[f"{col}_img_input"](data_dict[f"{col}_img_input"])
        
        return img_embedding_dict


    def _mask_others_remain(self, dynamic_embedding_dict, idx_dict, remain_rto, device):
        others_data = [val for key, val in dynamic_embedding_dict.items() if key in self.data_info.modality_info["others"]]
        others_data = torch.stack(others_data, dim=1)
        
        # if self.training:
        if True:
            remain_rto_ = remain_rto["cat"]
        else:
            remain_rto_ = 1
        others_remain_data, others_idx = self.others_remain(others_data, self.others_remain_pos_emb, remain_rto_, device)
        idx_dict.update(others_idx)

        return others_remain_data, idx_dict
        
    def _mask_temporal_remain(self, dynamic_embedding_dict, idx_dict):
        temporal_remain_dict = {}
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        for col in target_cols:
            temporal_remain_dict[col] = self.temporal_remain_dict[col](dynamic_embedding_dict[col], self.temporal_remain_pos_enc, idx_dict[f"{col}_remain_idx"])
        return temporal_remain_dict

    def _mask_img_remain(self, img_embedding_dict, idx_dict, remain_rto, device):
        img_remain_dict = {}
        target_cols = self.data_info.modality_info["img"]
        for col in target_cols:
            # if self.training:
            if True:
                remain_rto_ = remain_rto["general"]
            else:
                remain_rto_ = 1
            img_remain_dict[f"{col}_img_input"], img_idx_dict = self.img_remain_dict[f"{col}_img_input"](img_embedding_dict[f"{col}_img_input"], self.img_remain_pos_enc, remain_rto_, col, device)
            idx_dict.update(img_idx_dict)

        return img_remain_dict, idx_dict

    
    def _concat_all(self, others_remain_data, temporal_remain_dict, img_remain_dict, modality_emb, device):
        concat_li = []
        modality_idx = 0
        
        # Others
        modality = (torch.zeros(others_remain_data.shape[:-1]) + modality_idx).to(torch.long).to(device)
        others_remain_data += modality_emb(modality)
        concat_li.append(others_remain_data)
        modality_idx += 1

        # Temporal
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            modality = (torch.zeros(temporal_remain_dict[col].shape[:-1]) + modality_idx).to(torch.long).to(device)
            temporal_remain_dict[col] += modality_emb(modality)
            concat_li.append(temporal_remain_dict[col])
            modality_idx += 1
        
        # Img
        for col in self.data_info.modality_info["img"]:
            modality = (torch.zeros(img_remain_dict[f"{col}_img_input"].shape[:-1]) + modality_idx).to(torch.long).to(device)
            modality_emb = modality_emb(modality)
            img_remain_dict[f"{col}_img_input"] += modality_emb
            concat_li.append(img_remain_dict[f"{col}_img_input"])
            modality_idx += 1

        return torch.cat(concat_li, dim=1)

    def _concat_block(self, others_data, temporal_dict, img_dict, modality_emb, device):
        modality_idx = 0
        
        # Others
        modality = (torch.zeros(others_data.shape[:-1]) + modality_idx).to(torch.long).to(device)
        others_data += modality_emb(modality)
        modality_idx += 1

        # Temporal
        temporal_li = []
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            modality = (torch.zeros(temporal_dict[col].shape[:-1]) + modality_idx).to(torch.long).to(device)
            temporal_dict[col] += modality_emb(modality)
            temporal_li.append(temporal_dict[col])
            modality_idx += 1

        # Img
        img_li = []
        for col in self.data_info.modality_info["img"]:
            modality = (torch.zeros(img_dict[f"{col}_img_input"].shape[:-1]) + modality_idx).to(torch.long).to(device)
            modality_emb = modality_emb(modality)
            img_dict[f"{col}_img_input"] += modality_emb
            img_li.append(img_dict[f"{col}_img_input"])
            modality_idx += 1
        
        temporal_concat = torch.stack(temporal_li, dim=-2)
        others_concat = torch.cat([others_data] + img_li, dim=1)
        
        return temporal_concat, others_concat
    
    def _get_padding_mask(self, others_remain_data, padding_mask_dict, img_remain_dict, mode, device):
        padding_mask_li = []

        # Others
        others_padding_mask = torch.ones(others_remain_data.shape[:-1]).to(device)
        padding_mask_li.append(others_padding_mask)

        # Temporal
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            mask = padding_mask_dict[f"{col}_{mode}_padding_mask"]
            global_token_mask = torch.ones(mask.shape[0], 1).to(device)
            mask = torch.cat([global_token_mask, mask], dim=1)
            padding_mask_li.append(mask)
        
        # Img
        for col in self.data_info.modality_info["img"]:
            mask = torch.ones(img_remain_dict[f"{col}_img_input"].shape[:-1]).to(device)
            padding_mask_li.append(mask)
        
        padding_mask = torch.cat(padding_mask_li, dim=1)
        return torch.where(padding_mask == 1, 0, -torch.inf)


    def _split_full(self, concat_data, last_others_data, last_temporal_dict, last_img_dict):
        start_idx = 0

        # Others
        length = last_others_data.shape[1]
        others_split_data = concat_data[..., start_idx: start_idx+length, :]
        start_idx += length

        # Temporal
        temporal_split_dict = {}
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            length = last_temporal_dict[col].shape[1]
            temporal_split_dict[col] = concat_data[..., start_idx:start_idx+length, :]
            start_idx += length

        # Img
        img_split_dict = {}
        for col in self.data_info.modality_info["img"]:
            length = last_img_dict[f"{col}_img_input"].shape[1]
            img_split_dict[f"{col}_img_input"] = concat_data[..., start_idx:start_idx+length, :]
            start_idx += length
        
        return others_split_data, temporal_split_dict, img_split_dict

    def _split_block(self, temporal_decoding, others_decoding, others_revert_data, temporal_revert_dict, img_revert_dict):
        # Others
        len_others = others_revert_data.shape[1]
        others_split_data = others_decoding[:, :len_others, :]
        
        # Temporal
        temporal_split_dict = {}
        temporal_idx = 0
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            temporal_split_dict[col] = temporal_decoding[:, :, temporal_idx, :]
            temporal_idx += 1

        # Img
        img_split_dict = {}
        img_idx = 0
        img_decoding = others_decoding[:, len_others:, :]
        start_idx = 0
        for col in self.data_info.modality_info["img"]:
            img_len = img_revert_dict[f"{col}_img_input"].shape[1]
            img_split_dict[f"{col}_img_input"] = img_decoding[:, start_idx: start_idx+img_len, :]
            start_idx += img_len
            
        return others_split_data, temporal_split_dict, img_split_dict

    def _revert_others(self, others_data, idx_dict, device):
        others_revert_data = self.others_revert(others_data, idx_dict["others_revert_idx"], self.others_revert_pos_emb, device)
        return others_revert_data

    def _revert_temporal(self, temporal_data, idx_dict, padding_mask_dict, device):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        temporal_revert_dict = {}
        for col in target_cols:
            temporal_revert_dict[col] = self.temporal_revert_dict[col](temporal_data[col], idx_dict[f"{col}_revert_idx"], self.temporal_revert_pos_enc, padding_mask_dict[f"{col}_remain_padding_mask"], device)
        return temporal_revert_dict
    
    def _revert_img(self, img_data, idx_dict, device):
        target_cols = self.data_info.modality_info["img"]
        img_revert_dict = {}
        for col in target_cols:
            img_revert_dict[f"{col}_img_input"] = self.img_revert_dict[f"{col}_img_input"](img_data[f"{col}_img_input"], idx_dict[f"{col}_img_input_revert_idx"], self.img_revert_pos_enc, device)
        return img_revert_dict

    def _apply_output(self, others_split, temporal_split_dict, img_split_dict):
        output_dict = {}
        # Others
        target_cols = self.data_info.modality_info["others"]
        others_idx = 1 # Skip global token
        for col in target_cols:
            output_dict[col] = self.output_dict[col](others_split[:, others_idx, :])
            others_idx += 1
        
        # Temporal
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        for col in target_cols:
            output_dict[col] = self.output_dict[col](temporal_split_dict[col])
        
        # Img
        target_cols = self.data_info.modality_info["img"]
        for col in target_cols:
            output_dict[f"{col}_img_input"] = self.output_dict[f"{col}_img_input"](img_split_dict[f"{col}_img_input"])

        return output_dict


    def forward(self, data_input_dict, remain_rto, device):
        # Data to GPU
        data_dict, idx_dict, padding_mask_dict = self._data_to_gpu(data_input_dict, device)

        # Embedding
        dynamic_embedding_dict = self._embed_dynamic(data_dict)
        img_embedding_dict = self._embed_img(data_dict)

        # Remain masking (Pos encoding)
        others_remain_data, idx_dict = self._mask_others_remain(dynamic_embedding_dict, idx_dict, remain_rto, device)
        temporal_remain_dict = self._mask_temporal_remain(dynamic_embedding_dict, idx_dict)
        img_remain_dict, idx_idct = self._mask_img_remain(img_embedding_dict, idx_dict, remain_rto, device)

        # Concat (Modality embedding)
        concat_data = self._concat_all(others_remain_data, temporal_remain_dict, img_remain_dict, self.encoder_modality_emb, device)

        # Encoding
        encoder_padding_mask = self._get_padding_mask(others_remain_data, padding_mask_dict, img_remain_dict, "remain", device)
        encoding, encoding_weight = self.encoding(concat_data, src_key_padding_mask=encoder_padding_mask)
        encoding = self.linear(encoding)
        
        # Split
        others_split_data, temporal_split_dict, img_split_dict = self._split_full(encoding, others_remain_data, temporal_remain_dict, img_remain_dict)

        # Revert
        others_revert_data = self._revert_others(others_split_data, idx_dict, device)
        temporal_revert_dict = self._revert_temporal(temporal_split_dict, idx_dict, padding_mask_dict, device)
        img_revert_dict = self._revert_img(img_split_dict, idx_dict, device)

        # Concat (Modality embedding)
        concat_data = self._concat_all(others_revert_data, temporal_revert_dict, img_revert_dict, self.decoder_modality_emb, device)
        # temporal_concat, others_concat = self._concat_block(others_revert_data, temporal_revert_dict, img_revert_dict, self.decoder_modality_emb, device)

        # Decoding
        decoder_padding_mask = self._get_padding_mask(others_revert_data, padding_mask_dict, img_revert_dict, "revert", device)
        decoding, decoding_weight = self.decoding(concat_data, src_key_padding_mask=decoder_padding_mask)
        # temporal_decoding, others_decoding = temporal_concat, others_concat
        # for mod in self.decoding:
        #     temporal_decoding, others_decoding, temporal_attn, others_attn = mod(temporal_decoding, others_decoding)

        # Split
        others_split_data, temporal_split_dict, img_split_dict = self._split_full(decoding, others_revert_data, temporal_revert_dict, img_revert_dict)
        # others_split_data, temporal_split_dict, img_split_dict = self._split_block(temporal_decoding, others_decoding, others_revert_data, temporal_revert_dict, img_revert_dict)

        # Output
        output_dict = self._apply_output(others_split_data, temporal_split_dict, img_split_dict)

        return output_dict, idx_dict


class MultiheadBlockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def _print_shape(self, Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss):
        print("Qtt:", Qtt.shape)
        print("Qss:", Qss.shape)
        print("Ktt:", Ktt.shape)
        print("Kts:", Kts.shape)
        print("Kss:", Kss.shape)
        print("Vtt:", Vtt.shape)
        print("Vts:", Vts.shape)
        print("Vss:", Vss.shape)
        print("_"*100)

    def _linear_transform(self, temporal_concat, others_concat):
        Qtt = temporal_concat[:, 1:, :, :]
        Qss = torch.cat([temporal_concat[:, 0, :, :], others_concat], dim=1)
        
        Ktt = temporal_concat[:, 1:, :, :]
        Kts = others_concat
        Kss = torch.cat([temporal_concat[:, 0, :, :], others_concat], dim=1)

        Vtt = temporal_concat[:, 1:, :, :]
        Vts = others_concat
        Vss = torch.cat([temporal_concat[:, 0, :, :], others_concat], dim=1)

        Qtt = self.q_linear(Qtt)
        Qss = self.q_linear(Qss)

        Ktt = self.k_linear(Ktt)
        Kts = self.k_linear(Kts)
        Kss = self.k_linear(Kss)

        Vtt = self.v_linear(Vtt)
        Vts = self.v_linear(Vts)
        Vss = self.v_linear(Vss)

        return Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss

    def _split_head(self, Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss):
        batch_size, seq_len, _, d_model = Qtt.shape
        Qtt_ = Qtt.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        Ktt_ = Ktt.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        Vtt_ = Vtt.view(batch_size, seq_len, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)

        Qss_ = Qss.view(batch_size, 1, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        Kts_ = Kts.view(batch_size, 1, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        Kss_ = Kss.view(batch_size, 1, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        Vts_ = Vts.view(batch_size, 1, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        Vss_ = Vss.view(batch_size, 1, -1, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        
        return Qtt_, Qss_, Ktt_, Kts_, Kss_, Vtt_, Vts_, Vss_

    def _scaled_dot_product_attn(self, Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss):
        ### 1. Q·K^t ###
        # Temporal attention
        temp_temp = Qtt @ Ktt.permute(0,1,2,4,3)
        temp_others = Qtt @ Kts.permute(0,1,2,4,3)

        # Others attention
        others_others = Qss @ Kss.permute(0,1,2,4,3)

        ### 2. Softmax ###
        temporal_attn = torch.cat([temp_temp, temp_others], dim=-1)
        temporal_attn = torch.nn.functional.softmax(temporal_attn/math.sqrt(self.d_model//self.nhead), dim=-1)
        
        temp_temp_attn = temporal_attn[..., :temp_temp.shape[-1]]
        temp_others_attn = temporal_attn[..., -temp_others.shape[-1]:]

        others_attn = torch.nn.functional.softmax(others_others/math.sqrt(self.d_model//self.nhead), dim=-1)

        ### 3. Matmul V ###
        temporal = (temp_temp_attn @ Vtt) + (temp_others_attn @ Vts)
        others = others_attn @ Vss

        return temporal, others, temporal_attn, others_attn
        
    def forward(self, temporal_concat, others_concat, block_mask=None):
        Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss = self._linear_transform(temporal_concat, others_concat)
        Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss = self._split_head(Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss)

        # Scaled dot product attention
        temporal_, others_, temporal_attn, others_attn = self._scaled_dot_product_attn(Qtt, Qss, Ktt, Kts, Kss, Vtt, Vts, Vss)

        # Concat heads
        batch_size, _, num_temporal_modality, d_model = temporal_concat.shape
        temporal_ = temporal_.permute(0,2,3,1,4).reshape(batch_size, -1, num_temporal_modality, d_model)

        batch_size, _, d_model = others_concat.shape
        others_ = others_.reshape(batch_size, -1, d_model)

        # Give global temporal from others to tmeporal
        num_temporal_modality = temporal_.shape[-2]
        global_temporal = others_[..., :num_temporal_modality, :].unsqueeze(1)

        temporal = torch.cat([global_temporal, temporal_], dim=1)
        others = others_[..., num_temporal_modality:, :]

        assert (temporal.shape == temporal_concat.shape) and (others.shape == others_concat.shape)

        return temporal, others, temporal_attn, others_attn

class BlockEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, nhead, activation, dropout):
        super().__init__()

        self.self_attn = MultiheadBlockAttention(d_model, nhead, dropout)
        self._ff_block = FeedForward(d_model, d_ff, activation)

        self.temporal_dropout = torch.nn.Dropout(dropout)
        self.others_dropout = torch.nn.Dropout(dropout)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, temporal_concat, others_concat):
        temporal_attnoutput, others_attnoutput, temporal_attnweight, others_attnweight = self._sa_block(temporal_concat, others_concat)

        # Flatten
        batch_size, _, d_model = others_concat.shape
        flattened_others = others_concat.view(batch_size, -1, d_model)
        flattened_temporal = temporal_concat.view(batch_size, -1, d_model)
        
        flattened_input = torch.cat([flattened_others, flattened_temporal], dim=1)
        flattened_output = torch.cat([flattened_others, flattened_temporal], dim=1)

        x = flattened_input
        x = self.norm1(x + flattened_output)
        x = self.norm2(x + self._ff_block(x))

        others_output = x[:, :flattened_others.shape[1], :].view(others_attnoutput.shape)
        temporal_output = x[:, -flattened_temporal.shape[1]:, :].view(temporal_attnoutput.shape)

        return temporal_output, others_output, temporal_attnweight, others_attnweight

    # self-attention block
    def _sa_block(self, temporal_concat, others_concat):
        temporal_attnoutput, others_attnoutput, temporal_attnweight, others_attnweight = self.self_attn(temporal_concat, others_concat)
        
        temporal_attnoutput = self.temporal_dropout(temporal_attnoutput)
        others_attnoutput = self.others_dropout(others_attnoutput)

        return temporal_attnoutput, others_attnoutput, temporal_attnweight, others_attnweight

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, activation):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout()

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)

from torch.nn import functional as F

def _generate_square_subsequent_mask(sz, device, dtype):
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )

def _get_seq_len(src, batch_first):
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

def _detect_is_causal_mask(mask, is_causal=None,size=None):
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

class EncoderLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src, pos_enc, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        attn_output, attn_weight = self._sa_block(x, pos_enc, src_mask, src_key_padding_mask, is_causal=is_causal)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self._ff_block(x))

        return x, attn_weight

    # self-attention block
    def _sa_block(self, x, pos_enc, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weight = self.self_attn(x+pos_enc, x+pos_enc, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal, average_attn_weights=False)
        return self.dropout1(x), attn_weight

class Encoder(torch.nn.TransformerEncoder):
    def forward(self, src, pos_enc=0, mask=None, src_key_padding_mask=None, is_causal=None):
       ################################################################################################################
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
       ################################################################################################################

        for mod in self.layers:
            output, attn_weight = mod(output, pos_enc, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weight