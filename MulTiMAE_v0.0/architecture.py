import torch

from utils import DynamicEmbedding, ImgEmbedding, TemporalRemain, ImgRemain, OthersRemain, TemporalRevert, ImgRevert, OthersRevert,\
                    DynamicOutput, ImgOutput

class Transformer(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict, patch_size,
                        d_model, num_layers, nhead, d_ff, dropout):
        super().__init__()
        activation = "gelu"
        self.data_info, self.label_encoder_dict = data_info, label_encoder_dict
        
        # Embedding
        self.dynamic_embedding_dict = self._initiate_dynamic_embedding(label_encoder_dict, d_model)
        self.img_embedding_dict = self._initiate_img_embedding(d_model, patch_size)

        # Remain masking
        self.temporal_remain_li = self._initiate_temporal_remain(d_model)
        self.img_remain_li = self._initiate_img_remain(d_model)
        self.others_remain = OthersRemain(data_info, d_model)

        # Encoding
        self.encoding = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation, batch_first=True), num_layers)

        # Mask token
        self.mask_token_dict = self._initialize_mask_token(d_model)

        # Revert
        self.temporal_revert_li = self._initiate_temporal_revert(d_model)
        self.img_revert_li = self._initiate_img_revert(d_model)
        self.others_revert = OthersRevert(data_info, d_model)

        # Decoding
        self.decoding = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation, batch_first=True), num_layers)

        # Output
        self.output_dict = self._initiate_output(label_encoder_dict, patch_size, d_model)
    
    def _initiate_dynamic_embedding(self, label_encoder_dict, d_model):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"] + self.data_info.modality_info["others"]
        result_dict = {}
        for col in target_cols:
            result_dict[col] = DynamicEmbedding(col, self.data_info, label_encoder_dict, d_model)
        return torch.nn.ModuleDict(result_dict)

    def _initiate_img_embedding(self, d_model, patch_size):
        target_cols = self.data_info.modality_info["img"]
        result_dict = {}
        for col in target_cols:
            result_dict[f"{col}_img_input"] = ImgEmbedding(d_model, patch_size)
        return torch.nn.ModuleDict(result_dict)
    
    def _initiate_temporal_remain(self, d_model):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        result_li = []
        for col in target_cols:
            result_li.append(TemporalRemain(d_model))
        return torch.nn.ModuleList(result_li)

    def _initiate_img_remain(self, d_model):
        target_cols = self.data_info.modality_info["img"]
        result_li = []
        for col in target_cols:
            result_li.append(ImgRemain(d_model))
        return torch.nn.ModuleList(result_li)

    def _initialize_mask_token(self, d_model):
        result_dict = {}
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"] + self.data_info.modality_info["others"]:
            result_dict[col] = torch.nn.Parameter(torch.zeros(1, d_model))
        
        for col in self.data_info.modality_info["img"]:
            result_dict[f"{col}_img_input"] = torch.nn.Parameter(torch.zeros(1, d_model))
        
        result_dict["others"] = torch.nn.Parameter(torch.zeros(1, d_model))

        return torch.nn.ParameterDict(result_dict)

    def _initiate_temporal_revert(self, d_model):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        result_li = []
        for col in target_cols:
            result_li.append(TemporalRevert(d_model))
        return torch.nn.ModuleList(result_li)

    def _initiate_img_revert(self, d_model):
        target_cols = self.data_info.modality_info["img"]
        result_li = []
        for col in target_cols:
            result_li.append(ImgRevert(d_model))
        return torch.nn.ModuleList(result_li)

    def _initiate_output(self, label_encoder_dict, patch_size, d_model):
        output_dict = {}
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            output_dict[col] = DynamicOutput(col, self.data_info, label_encoder_dict, d_model)
        
        for col in self.data_info.modality_info["img"]:
            output_dict[f"{col}_img_input"] = ImgOutput(patch_size, d_model)

        for col in self.data_info.modality_info["others"]:
            output_dict[col] = DynamicOutput(col, self.data_info, label_encoder_dict, d_model)
        
        return torch.nn.ModuleDict(output_dict)


    def _embed_dynamic(self, data_input, device):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"] + self.data_info.modality_info["others"]
        result_dict = {}
        for col in target_cols:
            result_dict[col] = self.dynamic_embedding_dict[col](data_input[col].to(device))
        
        return result_dict

    def _embed_img(self, data_input, device):
        target_cols = self.data_info.modality_info["img"]
        result_dict = {}
        for col in target_cols:
            result_dict[f"{col}_img_input"] = self.img_embedding_dict[f"{col}_img_input"](data_input[f"{col}_img_input"].to(device))
        
        return result_dict


    def _mask_temporal_remain(self, embedding_dict, data_input, device):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        result_dict = {}
        idx_dict = {}
        for col, temporal_remain in zip(target_cols, self.temporal_remain_li):
            result_dict[col], remain_idx = temporal_remain(embedding_dict[col], data_input[f"{col}_remain_idx"].to(device), device)
            idx_dict[f"{col}_remain_idx"] = remain_idx
            idx_dict[f"{col}_masked_idx"] = data_input[f"{col}_masked_idx"].to(device)
            idx_dict[f"{col}_revert_idx"] = data_input[f"{col}_revert_idx"].to(device)
        return result_dict, idx_dict
    
    def _mask_img_remain(self, embedding_dict, remain_rto, device):
        target_cols = self.data_info.modality_info["img"]
        result_dict = {}
        idx_dict = {}
        for col, img_remain in zip(target_cols, self.img_remain_li):
            data, remain_idx, masked_idx, revert_idx = img_remain(embedding_dict[f"{col}_img_input"], remain_rto, device)

            result_dict[f"{col}_img_input"] = data
            idx_dict[f"{col}_img_remain_idx"] = remain_idx
            idx_dict[f"{col}_img_masked_idx"] = masked_idx
            idx_dict[f"{col}_img_revert_idx"] = revert_idx
            
        return result_dict, idx_dict

    def _mask_others_remain(self, dynamic_embedding_dict, temporal_remain_dict, img_remain_dict, remain_rto, device):
        others_data_dict = {key:val for key, val in dynamic_embedding_dict.items() if key in self.data_info.modality_info["others"]}
        temporal_remain_dict, img_remain_dict, others_data, others_idx_dict = self.others_remain(temporal_remain_dict, img_remain_dict, others_data_dict, remain_rto, device)

        return temporal_remain_dict, img_remain_dict, others_data, others_idx_dict

    def _get_padding_mask(self, data_input, img_dict, others_data, mode, device):
        temporal_padding_mask_li = []
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            mask = data_input[f"temporal_{mode}_padding_mask"]
            mask = torch.cat([torch.ones(mask.shape[0], 1), mask], dim=1)
            temporal_padding_mask_li.append(mask)
        temporal_padding_mask = torch.cat(temporal_padding_mask_li, dim=1).to(device)

        img_padding_mask_li = []
        for col in self.data_info.modality_info["img"]:
            mask = torch.ones(img_dict[f"{col}_img_input"].shape[:-1])
            img_padding_mask_li.append(mask)
        if len(img_padding_mask_li) > 0:
            img_padding_mask = torch.cat(img_padding_mask_li, dim=1).to(device)
        else: img_padding_mask = torch.tensor([]).to(device)

        others_padding_mask = torch.ones(others_data.shape[:-1]).to(device)

        total_padding_mask = torch.cat([temporal_padding_mask, img_padding_mask, others_padding_mask], dim=1)
        total_padding_mask = torch.where(total_padding_mask==1, 0, -torch.inf)

        return total_padding_mask


    def _split_each(self, data, last_temporal_embedding_dict, last_img_embedding_dict, last_others_embedding):
        start_idx = 0
        
        # Temporal
        temporal_split_dict = {}
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            # Select specific feature
            length = last_temporal_embedding_dict[col].shape[1]
            selected_embedding = data[..., start_idx: start_idx+length, :]
            temporal_split_dict[col] = selected_embedding
            start_idx += length
        
        # Image
        img_split_dict = {}
        for col in self.data_info.modality_info["img"]:
            # Select specific feature
            length = last_img_embedding_dict[f"{col}_img_input"].shape[1]
            selected_embedding = data[..., start_idx: start_idx+length, :]
            img_split_dict[f"{col}_img_input"] = selected_embedding
            start_idx += length

        # Others
        length = last_others_embedding.shape[1]
        selected_embedding = data[..., start_idx: start_idx+length, :]
        others_split = selected_embedding

        return temporal_split_dict, img_split_dict, others_split

    def _apply_temporal_revert(self, remain_dict, data_input, idx_dict, device):
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        result_dict = {}
        for col, temporal_revert in zip(target_cols, self.temporal_revert_li):
            result_dict[col] = temporal_revert(remain_dict[col], self.mask_token_dict[col], idx_dict[f"{col}_revert_idx"], device, data_input["temporal_remain_padding_mask"].to(device))
        return result_dict

    def _apply_img_revert(self, remain_dict, idx_dict, device):
        target_cols = self.data_info.modality_info["img"]
        result_dict = {}
        for col, img_revert in zip(target_cols, self.img_revert_li):
            result_dict[f"{col}_img_input"] = img_revert(remain_dict[f"{col}_img_input"], self.mask_token_dict[f"{col}_img_input"], idx_dict[f"{col}_img_revert_idx"], device)
        return result_dict

    def _apply_others_revert(self, temporal_revert_dict, img_revert_dict, others_remain_data, others_idx_dict, device):
        temporal_revert_dict, img_revert_dict, others_revert = self.others_revert(temporal_revert_dict, img_revert_dict, others_remain_data, self.mask_token_dict["others"], others_idx_dict["others_revert_idx"], self.data_info, device)
        return temporal_revert_dict, img_revert_dict, others_revert

    def _apply_output(self, temporal_split_dict, img_split_dict, others_split):
        output_dict = {}
        # Temporal
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        for col in target_cols:
            output_dict[col] = self.output_dict[col](temporal_split_dict[col])
        
        # Img
        target_cols = self.data_info.modality_info["img"]
        for col in target_cols:
            output_dict[f"{col}_img_input"] = self.output_dict[f"{col}_img_input"](img_split_dict[f"{col}_img_input"])

        # Others
        target_cols = self.data_info.modality_info["others"]
        others_idx = 0
        for col in target_cols:
            output_dict[col] = self.output_dict[col](others_split[:, others_idx, :])
            others_idx += 1
        
        return output_dict


    def forward(self, data_input, remain_rto, device):
        # Embedding: Just embedding
        dynamic_embedding_dict = self._embed_dynamic(data_input, device)
        img_embedding_dict = self._embed_img(data_input, device)

        # Remain masking: 1.pos_enc, 2.shuffle&remain, 3.global_token&pos_enc
        temporal_remain_dict, temporal_idx_dict = self._mask_temporal_remain(dynamic_embedding_dict, data_input, device)
        img_remain_dict, img_idx_dict = self._mask_img_remain(img_embedding_dict, remain_rto["general"], device)
        ### Others remain masking: 1.modlity embedding
        temporal_remain_dict, img_remain_dict, others_remain_data, others_idx_dict = self._mask_others_remain(dynamic_embedding_dict, temporal_remain_dict, img_remain_dict, remain_rto["cat"], device)

        # Concat
        encoder_padding_mask = self._get_padding_mask(data_input, img_remain_dict, others_remain_data, mode="remain", device=device)
        encoder_input = torch.cat(list(temporal_remain_dict.values()) + list(img_remain_dict.values()) + [others_remain_data], dim=1)
        encoding = self.encoding(encoder_input, src_key_padding_mask=encoder_padding_mask)

        # Split
        temporal_split_dict, img_split_dict, others_split = self._split_each(encoding, temporal_remain_dict, img_remain_dict, others_remain_data)

        # Revert: 1.Revert 2. pos_enc
        temporal_revert_dict = self._apply_temporal_revert(temporal_split_dict, data_input, temporal_idx_dict, device)
        img_revert_dict = self._apply_img_revert(img_split_dict, img_idx_dict, device)
        # ### Others revert: 1.Modality embedding
        temporal_revert_dict, img_revert_dict, others_revert = self._apply_others_revert(temporal_revert_dict, img_revert_dict, others_split, others_idx_dict, device)

        # Concat
        decoder_padding_mask = self._get_padding_mask(data_input, img_revert_dict, others_revert, mode="revert", device=device)
        decoder_input = torch.cat(list(temporal_revert_dict.values()) + list(img_revert_dict.values()) + [others_revert], dim=1)
        decoding = self.decoding(decoder_input, src_key_padding_mask=decoder_padding_mask)

        # Split
        temporal_split_dict, img_split_dict, others_split = self._split_each(decoding, temporal_revert_dict, img_revert_dict, others_revert)

        # Output
        output_dict = self._apply_output(temporal_split_dict, img_split_dict, others_split)

        return output_dict, others_idx_dict