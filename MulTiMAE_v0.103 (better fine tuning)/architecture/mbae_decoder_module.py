import torch
from architecture.shared_module import *

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn = MultiheadBlockAttention(d_model, nhead)

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
    
    def forward(self, tgt_dict, memory_dict, tgt_key_padding_mask_dict, memory_key_padding_mask_dict, fine_tuning=False):
        tgt_modality = list(tgt_dict.keys()); assert len(tgt_modality)==1
        tgt_modality = tgt_modality[0]

        x = tgt_dict[tgt_modality]
        
        # Cross attention
        tgt_dict = {tgt_modality: x.unsqueeze(-2) if tgt_modality=="temporal" else x}
        cross_attn_output_dict, cross_attn_weight_dict = self._ca_block(self.flatten_apply(tgt_dict, module=self.ca_norm), memory_dict, memory_key_padding_mask_dict)
        x = x + cross_attn_output_dict[tgt_modality].squeeze()

        # # Self attention
        # if tgt_modality == "static" or tgt_modality == "global":
        #     self_attn_output, self_attn_weight = self._sa_block(self.sa_norm(x), tgt_key_padding_mask_dict[tgt_modality])
        #     x = x + self_attn_output

        self_attn_weight = None
        
        # Feed forward
        x = x + self._ff_block(self.ff_norm(x))

        return {tgt_modality:x}, self_attn_weight, cross_attn_weight_dict, tgt_modality

    def flatten_apply(self, data_dict, module):
        # Flaten
        flatten_li = []
        temporal_length = 0
        for key, val in data_dict.items():
            if key == "temporal":
                temporal_block_shape = val.shape
                temporal_flattened = val.view(temporal_block_shape[0], -1, temporal_block_shape[-1])
                temporal_length = temporal_flattened.shape[-2]
                flatten_li.append(temporal_flattened)
                
            elif key == "static":
                flatten_li.append(val)
            
        flattened_data = torch.cat(flatten_li, dim=1)

        # Apply module
        moduled_data = module(flattened_data)
        
        # Un-flatten
        result_dict = {}
        for key, val in data_dict.items():
            if key == "temporal":
                result_dict[key] = moduled_data[:, :temporal_length, :].view(temporal_block_shape)
            elif key == "static":
                result_dict[key] = moduled_data[:, temporal_length:, :]
        
        return result_dict

    def _sa_block(self, src, padding_mask):
        padding_mask = torch.where(padding_mask==1, 0, -torch.inf)
        x, attn_weight = self.self_attn(src, src, src, key_padding_mask=padding_mask, average_attn_weights=False)
        return self.sa_dropout(x), attn_weight
    
    def _ca_block(self, tgt_dict, memory_dict, padding_mask_dict):
        attn_output_dict, attn_weight_dict = self.cross_attn(tgt_dict, memory_dict, memory_dict, padding_mask_dict)
        return self.flatten_apply(attn_output_dict, module=self.ca_dropout), attn_weight_dict

    def _ff_block(self, x):
        x = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(x)

class Decoder(torch.nn.Module):
    def __init__(self, temporal_cols, img_cols, nlp_cols, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols
        self.total_cols = temporal_cols + img_cols + nlp_cols

        self.decoder_layer_dict = torch.nn.ModuleDict()
        for col in self.total_cols:
            self.decoder_layer_dict[col] = torch.nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, data_dict, padding_mask_dict, device):
        result_dict, self_attn_weight_dict, cross_attn_weight_dict_dict = {}, {}, {}

        for col in self.total_cols:
            tgt_dict, tgt_key_padding_mask_dict, memory_dict, memory_key_padding_mask_dict= self.get_tgt_memory(col, data_dict, padding_mask_dict, device)

            for mod in self.decoder_layer_dict[col]:
                tgt_dict, self_attn_weight, cross_attn_weight_dict, tgt_modality = mod(tgt_dict, memory_dict, tgt_key_padding_mask_dict, memory_key_padding_mask_dict)
        
            result_dict[col] = tgt_dict[tgt_modality]
            self_attn_weight_dict[col] = self_attn_weight
            cross_attn_weight_dict_dict[col] = cross_attn_weight_dict
        
        return result_dict, self_attn_weight_dict, cross_attn_weight_dict_dict
        
    def get_tgt_memory(self, col, data_dict, padding_mask_dict, device):
        tgt_dict, tgt_key_padding_mask_dict, memory_dict, memory_key_padding_mask_dict = {}, {}, {}, {}
        modality = "temporal" if col in self.temporal_cols else "static"

        # Obtain target
        tgt_dict[modality] = data_dict[col]
        tgt_key_padding_mask_dict[modality] = padding_mask_dict["temporal_padding_mask"] if modality=="temporal" else padding_mask_dict[f"{col}_revert_padding_mask"]
        assert tgt_dict[modality].shape[:-1] == tgt_key_padding_mask_dict[modality].shape

        # Memory
        ### Temporal memory
        # temporal_memory_li = [val for key, val in data_dict.items() if key!=col and key in ["global"]+self.temporal_cols]
        temporal_memory_li = [val for key, val in data_dict.items() if key in ["global"]+self.temporal_cols]
        temporal_memory_block = torch.stack(temporal_memory_li, dim=-2)
        memory_dict["temporal"] = temporal_memory_block

        num_modality = temporal_memory_block.shape[-2]
        temporal_key_padding_mask = padding_mask_dict["temporal_padding_mask"].unsqueeze(-1).repeat(1, 1, num_modality)
        temporal_key_padding_mask[:, :, 1] = padding_mask_dict["target_fcst_mask"]
        # temporal_key_padding_mask = torch.ones(padding_mask_dict["temporal_padding_mask"].shape).unsqueeze(-1).repeat(1, 1, num_modality).to(device)

        memory_key_padding_mask_dict["temporal"] = temporal_key_padding_mask
        assert temporal_memory_block.shape[:-1] == temporal_key_padding_mask.shape

        ### Static memory
        # static_memory_li = [val for key, val in data_dict.items() if key!=col and key in self.img_cols+self.nlp_cols]
        static_memory_li = [val for key, val in data_dict.items() if key in self.img_cols+self.nlp_cols]
        if len(static_memory_li) > 0:
            static_memory = torch.cat(static_memory_li, dim=1)
            memory_dict["static"] = static_memory
            
            static_key_padding_mask_li = [padding_mask_dict[f"{key}_revert_padding_mask"] for key in data_dict.keys() if key in self.img_cols+self.nlp_cols]
            static_key_padding_mask = torch.cat(static_key_padding_mask_li, dim=1)
            memory_key_padding_mask_dict["static"] = static_key_padding_mask
            assert static_memory.shape[:-1] == static_key_padding_mask.shape

        return tgt_dict, tgt_key_padding_mask_dict, memory_dict, memory_key_padding_mask_dict


class TemporalOutput(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.processing_info["scaling_cols"] or col == "global":
            self.output = torch.nn.Sequential(
                                    # torch.nn.Linear(d_model, d_model),
                                    torch.nn.Linear(d_model, 1))
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].get_num_cls()
            self.output = torch.nn.Sequential(
                                    # torch.nn.Linear(d_model, d_model),
                                    torch.nn.Linear(d_model, num_cls))

    def forward(self, data):
        return self.output(data)

class ImgOutput(torch.nn.Module):
    def __init__(self, d_model, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # self.output = torch.nn.Sequential(
        #     torch.nn.Linear(d_model, d_model),
        #     torch.nn.Linear(d_model, 3*patch_size*patch_size))

        self.output1 = torch.nn.Sequential(
            # torch.nn.Linear(d_model, d_model),
            torch.nn.Linear(d_model, 3*patch_size*patch_size))
        # self.output2 = torch.nn.ConvTranspose2d(3*patch_size*patch_size, 3, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, data):
        # data = self.output1(data)
        # data = data.permute(0,2,1)[:, :, 1:]
        # data = data.reshape(data.shape[0], data.shape[1], int(224//self.patch_size), int(224//self.patch_size))
        # return self.output2(data)
        return self.output1(data)

class NlpOutput(torch.nn.Module):
    def __init__(self, d_model, num_vocab=30522):
        super().__init__()
        self.output = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Linear(d_model, num_vocab))
    
    def forward(self, data):
        return self.output(data)

class Output(torch.nn.Module):
    def __init__(self, data_info, d_model, nhead, d_ff, dropout, activation, num_layers, label_encoder_dict, temporal_cols, img_cols, nlp_cols, patch_size):
        super().__init__()
        self.output_dict, self.pre_output_dict, self.encoder_dict = torch.nn.ModuleDict(), torch.nn.ModuleDict(), torch.nn.ModuleDict()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols
        for col in temporal_cols:
            self.pre_output_dict[col] = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, d_model),)
            self.encoder_dict[col] = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation, batch_first=True, norm_first=True), num_layers)
            self.output_dict[col] = TemporalOutput(col, data_info, label_encoder_dict, d_model)
        for col in img_cols:
            self.pre_output_dict[col] = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, d_model),)
            self.encoder_dict[col] = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation, batch_first=True, norm_first=True), num_layers)
            self.output_dict[col] = ImgOutput(d_model, patch_size)
        for col in nlp_cols:
            self.pre_output_dict[col] = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, d_model),)
            self.encoder_dict[col] = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation, batch_first=True, norm_first=True), num_layers)
            self.output_dict[col] = NlpOutput(d_model)
    
    def forward(self, data_dict, padding_mask_dict):
        result_dict = {}
        for key, val in data_dict.items():
            if key in self.temporal_cols+["global"]:
                padding_mask = padding_mask_dict["temporal_padding_mask"]
            else:
                padding_mask = padding_mask_dict[f"{key}_revert_padding_mask"]
            padding_mask = torch.where(padding_mask==1, 0, -torch.inf)

            val = self.pre_output_dict[key](val)
            if key in self.img_cols + self.nlp_cols + ["global"]:
                val = self.encoder_dict[key](val, src_key_padding_mask=padding_mask)
            result_dict[key] = self.output_dict[key](val)
        
        return result_dict
1==1