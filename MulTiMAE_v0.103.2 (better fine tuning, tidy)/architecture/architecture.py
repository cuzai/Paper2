import torch
from architecture.mbae_encoder_module import *
from architecture.mbae_decoder_module import *

class MBAEEncoder(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols,
                    d_model, num_layers, nhead, d_ff, dropout, activation,
                    patch_size):
        super().__init__()
        self.data_info = data_info
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols

        # 1. Embedding
        self.embedding = Embedding(self.data_info, label_encoder_dict, d_model["encoder"], temporal_cols, img_cols, nlp_cols, patch_size)
        # 2. Encoder positional encoding & Modality embedding
        self.posmod_emb = PosModEmb(d_model["encoder"], dropout, ["global"]+temporal_cols, img_cols, nlp_cols)
        # 3. Remain masking
        self.remain_mask = Remain(["global"]+temporal_cols, img_cols, nlp_cols)
        self.norm1 = torch.nn.LayerNorm(d_model["encoder"])
        # 4. Encoder
        self.encoder = Encoder(d_model["encoder"], nhead, d_ff["encoder"], dropout, activation, num_layers["encoder"])
        self.norm2 = torch.nn.LayerNorm(d_model["encoder"])
        # 6. Revert
        self.revert = Revert(d_model["encoder"], ["global"]+temporal_cols, img_cols, nlp_cols)
        # 5. To decoder dimension
        self.decoder_dim_linear = torch.nn.Linear(d_model["encoder"], d_model["decoder"])
    
    def forward(self, data_input, remain_rto, device):
        data_dict, idx_dict, padding_mask_dict = self.to_gpu(data_input, device)
        # 1. Embedding
        embedding_dict = self.embedding(data_dict, padding_mask_dict, device)
        # 2. Encoder positional encoding & Modality embedding
        posmod_emb_dict = self.posmod_emb(embedding_dict, device)
        # 3. Remain masking
        temporal_block_remain, img_remain_dict, nlp_remain_dict, idx_dict, padding_mask_dict = self.remain_mask(posmod_emb_dict, idx_dict, padding_mask_dict, remain_rto, device)

        flattened = temporal_block_remain.reshape(temporal_block_remain.shape[0], -1, temporal_block_remain.shape[-1])
        for col in self.img_cols:
            flattened = torch.cat([flattened, img_remain_dict[col]], dim=-2)
        for col in self.nlp_cols:
            flattened = torch.cat([flattened, nlp_remain_dict[col]], dim=-2)
        
        flattened = self.norm1(flattened)
        
        idx = temporal_block_remain.shape[1]*temporal_block_remain.shape[2]
        temporal_block_remain = flattened[:, :idx, :].reshape(temporal_block_remain.shape)
        for col in self.img_cols:
            length = img_remain_dict[col].shape[1]
            img_remain_dict[col] = flattened[:, idx:idx+length, :]
            idx += length
        
        for col in self.nlp_cols:
            length = nlp_remain_dict[col].shape[1]
            nlp_remain_dict[col] = flattened[:, idx:idx+length, :]
            idx += length

        # 4. Encoder
        temporal_encoding_block, img_encoding_dict, nlp_encoding_dict, encoding_weight_dict = self.encoder(temporal_block_remain, img_remain_dict, nlp_remain_dict, padding_mask_dict)

        flattened = temporal_encoding_block.reshape(temporal_encoding_block.shape[0], -1, temporal_encoding_block.shape[-1])
        for col in self.img_cols:
            flattened = torch.cat([flattened, img_encoding_dict[col]], dim=-2)
        for col in self.nlp_cols:
            flattened = torch.cat([flattened, nlp_encoding_dict[col]], dim=-2)
        
        flattened = self.norm2(flattened)
        
        idx = temporal_encoding_block.shape[1]*temporal_encoding_block.shape[2]
        temporal_encoding_block = flattened[:, :idx, :].reshape(temporal_encoding_block.shape)
        for col in self.img_cols:
            length = img_encoding_dict[col].shape[1]
            img_encoding_dict[col] = flattened[:, idx:idx+length, :]
            idx += length
        
        for col in self.nlp_cols:
            length = nlp_encoding_dict[col].shape[1]
            nlp_encoding_dict[col] = flattened[:, idx:idx+length, :]
            idx += length

        # 6. Revert
        revert_dict = self.revert(temporal_encoding_block, img_encoding_dict, nlp_encoding_dict, idx_dict, padding_mask_dict)
        
        flattened = torch.stack([val for key, val in revert_dict.items() if key in self.temporal_cols], dim=-2)
        batch_size, seq_len, num_modality, d_model = flattened.shape
        flattened = flattened.reshape(flattened.shape[0], -1, flattened.shape[-1])

        for col in self.img_cols:
            flattened = torch.cat([flattened, revert_dict[col]], dim=-2)

        for col in self.nlp_cols:
            flattened = torch.cat([flattened, revert_dict[col]], dim=-2)
        
        flattened = self.norm2(flattened)

        temporal_block = flattened[:, :seq_len*num_modality, :].reshape(batch_size, seq_len, num_modality, d_model)
        for n, col in enumerate(self.temporal_cols):
            revert_dict[col] = temporal_block[:, :, n, :]
        
        idx = seq_len*num_modality
        for col in self.img_cols:
            length = revert_dict[col].shape[1]
            revert_dict[col] = flattened[:, idx:idx+length, :]
            idx += length
        
        for col in self.nlp_cols:
            length = revert_dict[col].shape[1]
            revert_dict[col] = flattened[:, idx:idx+length, :]
            idx += length

        # 7. Decoder positional encoding & Modality embedding
        posmod_emb_dict = self.posmod_emb(revert_dict, device)
        # 5. To decoder dimension   
        posmod_emb_dict = self.to_decoder_dim(posmod_emb_dict)
        
        return posmod_emb_dict,\
                encoding_weight_dict, data_dict, idx_dict, padding_mask_dict


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


    def to_decoder_dim(self, posmod_emb_dict):
        for key, val in posmod_emb_dict.items():
            posmod_emb_dict[key] = self.decoder_dim_linear(val)
    
        return posmod_emb_dict

class MBAEDecoder(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols, 
                    d_model, nhead, d_ff, dropout, activation, num_layers,
                    patch_size):
        super().__init__()
        self.temporal_cols, self.img_cols, self.nlp_cols = temporal_cols, img_cols, nlp_cols, 
        # 8. Decoding
        # self.decoder = Encoder(d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"])
        self.decoder = Decoder(temporal_cols, img_cols, nlp_cols, d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"])
        self.norm = torch.nn.LayerNorm(d_model["decoder"])
        # 9. Output
        self.output = Output(data_info, d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"], label_encoder_dict, temporal_cols, img_cols, nlp_cols, patch_size)
    
    def forward(self, posmod_emb_dict, idx_dict, padding_mask_dict, device):
        # 8. Decoding
        decoding_dict, self_attn_weight_dict, cross_attn_weight_dict_dict = self.decoder(posmod_emb_dict, padding_mask_dict, device)
        
        # temporal_block = torch.stack([val for key, val in posmod_emb_dict.items() if key in ["global"]+self.temporal_cols], dim=-2)
        # img_dict = {key:val for key, val in posmod_emb_dict.items() if key in self.img_cols}
        # nlp_dict = {key:val for key, val in posmod_emb_dict.items() if key in self.nlp_cols}

        # temporal_decoding_block, img_decoding_dict, nlp_decoding_dict, encoding_weight_dict = self.decoder(temporal_block, img_dict, nlp_dict, padding_mask_dict, mode="revert")
        # decoding_dict = {}
        # for n, col in enumerate(self.temporal_cols):
        #     decoding_dict[col] = temporal_decoding_block[:, :, n+1, :]
        # decoding_dict.update(img_decoding_dict)
        # decoding_dict.update(nlp_decoding_dict)

        # flattened = torch.stack([val for key, val in decoding_dict.items() if key in self.temporal_cols], dim=-2)
        # batch_size, seq_len, num_modality, d_model = flattened.shape
        # flattened = flattened.reshape(flattened.shape[0], -1, flattened.shape[-1])

        # for col in self.img_cols:
        #     flattened = torch.cat([flattened, decoding_dict[col]], dim=-2)

        # for col in self.nlp_cols:
        #     flattened = torch.cat([flattened, decoding_dict[col]], dim=-2)
        
        # flattened = self.norm(flattened)

        # temporal_block = flattened[:, :seq_len*num_modality, :].reshape(batch_size, seq_len, num_modality, d_model)
        # for n, col in enumerate(self.temporal_cols):
        #     decoding_dict[col] = temporal_block[:, :, n, :]
        
        # idx = seq_len*num_modality
        # for col in self.img_cols:
        #     length = decoding_dict[col].shape[1]
        #     decoding_dict[col] = flattened[:, idx:idx+length, :]
        #     idx += length
        
        # for col in self.nlp_cols:
        #     length = decoding_dict[col].shape[1]
        #     decoding_dict[col] = flattened[:, idx:idx+length, :]
        #     idx += length

        # 9. Output
        output_dict = self.output(decoding_dict, padding_mask_dict)

        return output_dict


class MaskedBlockAutoEncoder(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict,
                    d_model, num_layers, nhead, d_ff, dropout, activation,
                    patch_size):
        super().__init__()
        temporal_cols, img_cols, nlp_cols = self.define_col_modalities(data_info)

        self.mbae_encoder = MBAEEncoder(data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols,
                                        d_model, num_layers, nhead, d_ff, dropout, activation,
                                        patch_size)
        
        self.mbae_decoder = MBAEDecoder(data_info, label_encoder_dict, temporal_cols, img_cols, nlp_cols,
                                        d_model, nhead, d_ff, dropout, activation, num_layers,
                                        patch_size)

    def forward(self, data_input, remain_rto, device):
        posmod_emb_dict,\
                encoding_weight_dict, data_dict, idx_dict, padding_mask_dict = self.mbae_encoder(data_input, remain_rto, device)

        decoding_dict = self.mbae_decoder(posmod_emb_dict, idx_dict, padding_mask_dict, device)

        return decoding_dict, data_dict, idx_dict, padding_mask_dict
    
    def define_col_modalities(self, data_info):
        temporal_cols = data_info.modality_info["target"] + data_info.modality_info["temporal"]
        img_cols = data_info.modality_info["img"]
        nlp_cols = data_info.modality_info["nlp"]

        return temporal_cols, img_cols, nlp_cols
1==1