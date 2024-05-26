import torch
from architecture.mbae_encoder_module import *
from architecture.mbae_decoder_module import *

class MBAEEncoder(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols,
                    d_model, num_layers, nhead, d_ff, dropout, activation,
                    patch_size, is_from_pretrained):
        super().__init__()
        self.data_info, self.is_from_pretrained = data_info, is_from_pretrained

        # 1. Embedding
        self.embedding = Embedding(self.data_info, label_encoder_dict, d_model["encoder"], temporal_cols, img_cols, nlp_cols, patch_size, is_from_pretrained)
        # 2. Encoder positional encoding & Modality embedding
        self.posmod_emb = PosModEmb(d_model["encoder"], dropout, ["global"]+temporal_cols, img_cols, nlp_cols)
        # 3. Remain masking
        self.remain_mask = Remain(["global"]+temporal_cols, img_cols, nlp_cols)
        # 4. Encoder
        self.encoder = Encoder(d_model["encoder"], nhead, d_ff["encoder"], dropout, activation, num_layers["encoder"])
        # 5. To decoder dimension
        self.decoder_dim_linear = torch.nn.Linear(d_model["encoder"], d_model["decoder"])
        # 6. Revert
        self.revert = Revert(d_model["decoder"], ["global"]+temporal_cols, img_cols, nlp_cols, is_from_pretrained)
    
    def forward(self, data_input, remain_rto, device):
        data_dict, idx_dict, padding_mask_dict = self.to_gpu(data_input, device)
        # 1. Embedding
        embedding_dict = self.embedding(data_dict, padding_mask_dict, device)
        # 2. Encoder positional encoding & Modality embedding
        posmod_emb_dict = self.posmod_emb(embedding_dict, device)
        # 3. Remain masking
        temporal_block_remain, img_remain_dict, nlp_remain_dict, idx_dict, padding_mask_dict = self.remain_mask(posmod_emb_dict, idx_dict, padding_mask_dict, remain_rto, device)
        # 4. Encoder
        temporal_encoding_block, img_encoding_dict, nlp_encoding_dict, encoding_weight_dict = self.encoder(temporal_block_remain, img_remain_dict, nlp_remain_dict, padding_mask_dict)
        # 5. To decoder dimension
        temporal_encoding_block, img_encoding_dict, nlp_encoding_dict = self.to_decoder_dim(temporal_encoding_block, img_encoding_dict, nlp_encoding_dict)
        # 6. Revert
        revert_dict = self.revert(temporal_encoding_block, img_encoding_dict, nlp_encoding_dict, idx_dict)
        
        return revert_dict, encoding_weight_dict,\
                data_dict, idx_dict, padding_mask_dict

    def to_gpu(self, data_input, device):
        data_dict, idx_dict, padding_mask_dict = {}, {}, {}
        data_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"] + self.data_info.modality_info["img"] + self.data_info.modality_info["nlp"]
        for key, val in data_input.items():
            if key in data_cols:
                data_dict[key] = data_input[key].to(device)
            elif key.endswith("idx"):
                idx_dict[key] = data_input[key].to(device)
            elif key.endswith("mask"):
                padding_mask_dict[key] = data_input[key].to(device)
            
        return data_dict, idx_dict, padding_mask_dict

    def to_decoder_dim(self, temporal_encoding_block, img_encoding_dict, nlp_encoding_dict):
        if self.is_from_pretrained:
            temporal_encoding_block = self.decoder_dim_linear(temporal_encoding_block)
            for key, val in img_encoding_dict.items():
                img_encoding_dict[key] = self.decoder_dim_linear(val)
            
            for key, val in nlp_encoding_dict.items():
                nlp_encoding_dict[key] = self.decoder_dim_linear(val)
        
        return temporal_encoding_block, img_encoding_dict, nlp_encoding_dict

class MBAEDecoder(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols, 
                    d_model, nhead, d_ff, dropout, activation, num_layers,
                    patch_size):
        super().__init__()
        
        # 7. Decoder positional encoding & Modality embedding
        self.posmod_emb = PosModEmb(d_model["decoder"], dropout, ["global"]+temporal_cols, img_cols, nlp_cols)
        # 8. Decoding
        self.decoder = Decoder(temporal_cols, img_cols, nlp_cols, d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"])
        # 9. Output
        self.output = Output(data_info, d_model["decoder"], label_encoder_dict, temporal_cols, img_cols, nlp_cols, patch_size)
    
    def forward(self, data_dict, padding_mask_dict, device):
        # 7. Decoder positional encoding & Modality embedding
        posmod_emb_dict = self.posmod_emb(data_dict, device)
        # 8. Decoding
        decoding_dict, self_attn_weight_dict, cross_attn_weight_dict_dict = self.decoder(posmod_emb_dict, padding_mask_dict)
        # 9. Output
        output_dict = self.output(decoding_dict)

        return output_dict

class MaskedBlockAutoEncoder(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict,
                    d_model, num_layers, nhead, d_ff, dropout, activation,
                    patch_size, is_from_pretrained):
        super().__init__()
        temporal_cols, img_cols, nlp_cols = self.define_col_modalities(data_info)

        self.mbae_encoder = MBAEEncoder(data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols,
                                        d_model, num_layers, nhead, d_ff, dropout, activation,
                                        patch_size, is_from_pretrained)
        
        self.mbae_decoder = MBAEDecoder(data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols,
                                        d_model, nhead, d_ff, dropout, activation, num_layers,
                                        patch_size)
    
    def forward(self, data_input, remain_rto, device):
        encoding_dict, encoding_weight_dict,\
            data_dict, idx_dict, padding_mask_dict = self.mbae_encoder(data_input, remain_rto, device)

        decoding_dict = self.mbae_decoder(encoding_dict, padding_mask_dict, device)

        return decoding_dict, data_dict, idx_dict, padding_mask_dict
    
    def define_col_modalities(self, data_info):
        temporal_cols = data_info.modality_info["target"] + data_info.modality_info["temporal"]
        img_cols = data_info.modality_info["img"]
        nlp_cols = data_info.modality_info["nlp"]

        return temporal_cols, img_cols, nlp_cols
1==1