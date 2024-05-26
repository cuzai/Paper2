import torch
from architecture_detail import *

class Transformer(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict,
                d_model, num_layers, nhead, d_ff, dropout, activation):
        super().__init__()
        self.data_info, self.label_encoder_dict = data_info, label_encoder_dict
        self.temporal_cols, self.img_cols, self.nlp_cols, self.total_cols = self.define_cols()
        self.self_attn_weight_dict, self.cross_attn_weight_dict = {}, {}
        
        # 1. Embedding
        global_token = torch.nn.Parameter(torch.rand(1, d_model["encoder"]))
        self.embedding_dict = self.init_process(mod=Embedding, args=[self.data_info, self.label_encoder_dict, d_model["encoder"], global_token])
        # 2. Pos encoding and modality embedding
        num_modality = len(self.total_cols)
        modality = {col:n for n, col in enumerate(self.total_cols)}
        encoder_pos_enc = PositionalEncoding(d_model["encoder"], dropout)
        self.enc_pos_mod_encoding_dict = self.init_process(mod=PosModEncoding, args=[encoder_pos_enc, num_modality, modality, d_model["encoder"]])
        # 3. Remain mask
        global_token = torch.nn.Parameter(torch.rand(1, d_model["encoder"]))
        self.remain_dict = Remain(global_token)
        # 4. Encoding
        to_decoder_dim = torch.nn.Linear(d_model["encoder"], d_model["decoder"])
        self.encoding = Encoder(d_model["encoder"], nhead, d_ff["encoder"], dropout, activation, num_layers["encoder"], to_decoder_dim)
        # 5. Revert
        mask_token = torch.nn.Parameter(torch.rand(1, d_model["decoder"]))
        self.revert = Revert(mask_token)
        # 6. Pos encoding and modality embedding
        decoder_pos_enc = PositionalEncoding(d_model["decoder"], dropout)
        self.dec_pos_mod_encoding_dict = self.init_process(mod=PosModEncoding, args=[decoder_pos_enc, num_modality, modality, d_model["decoder"]])
        # 7. Decoding
        self.decoding = self.init_process(mod=Decoder, args=[d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"]])

        # 9. Output
        self.temporal_output = self.init_process(mod=TemporalOutput, args=[self.data_info, self.label_encoder_dict, d_model["decoder"]], target_cols=self.temporal_cols)
        self.img_output = self.init_process(mod=ImgOutput, args=[d_model["decoder"]], target_cols=self.img_cols)
        self.nlp_output = self.init_process(mod=NlpOutput, args=[d_model["decoder"]], target_cols=self.nlp_cols)

    def forward(self, data_input, remain_rto, device):
        data_dict, self.idx_dict, self.padding_mask_dict = self.to_gpu(data_input, device)
        
        # 1. Embedding
        embedding_dict = self.apply_process(data=data_dict, mod=self.embedding_dict, args=[self.padding_mask_dict, device])
        # 2. Pos encoding and modality embedding
        enc_pos_mod_encoding_dict = self.apply_process(data=embedding_dict, mod=self.enc_pos_mod_encoding_dict, args=[device])
        # 3. Remain mask
        temporal_dict, img_dict, nlp_dict, self.idx_dict, self.padding_mask_dict = self.remain_dict(enc_pos_mod_encoding_dict, self.idx_dict, self.padding_mask_dict, remain_rto, self.temporal_cols, self.img_cols, self.nlp_cols, device)
        # 4. Encoding
        temporal_encoding_dict, img_encoding_dict, nlp_encoding_dict = self.encoding(temporal_dict, img_dict, nlp_dict, self.padding_mask_dict, self.img_cols, self.nlp_cols, device)
        # 5. Revert
        revert_dict = self.revert(temporal_encoding_dict, img_encoding_dict, nlp_encoding_dict, self.idx_dict, self.temporal_cols, self.img_cols, self.nlp_cols)
        # 6. Pos encoding and modality embedding
        dec_pos_mod_encoding_dict = self.apply_process(data=revert_dict, mod=self.dec_pos_mod_encoding_dict, args=[device])
        # 7. Decoder
        decoding = self.apply_process(data=dec_pos_mod_encoding_dict, mod=self.decoding, args=[dec_pos_mod_encoding_dict, self.padding_mask_dict, self.temporal_cols, self.img_cols, self.nlp_cols, device], collate_fn=self.tidy_decoding)

        # 9. Output
        temporal_output = self.apply_process(data=decoding, mod=self.temporal_output, target_cols=self.temporal_cols)
        img_output = self.apply_process(data=decoding, mod=self.img_output, target_cols=self.img_cols)
        nlp_output = self.apply_process(data=decoding, mod=self.nlp_output, target_cols=self.nlp_cols)

        return temporal_output, img_output, nlp_output, self.self_attn_weight_dict, self.cross_attn_weight_dict, self.idx_dict, self.padding_mask_dict


    
    def define_cols(self):
        temporal_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        img_cols = self.data_info.modality_info["img"]
        nlp_cols = self.data_info.modality_info["nlp"]
        total_cols = temporal_cols + img_cols + nlp_cols

        return temporal_cols, img_cols, nlp_cols, total_cols

    def to_gpu(self, data_input, device):
        data_dict, idx_dict, padding_mask_dict = {}, {}, {}
        for key, val in data_input.items():
            if key in self.temporal_cols + self.img_cols + self.nlp_cols:
                data_dict[key] = data_input[key].to(device)
            elif key.endswith("idx"):
                idx_dict[key] = data_input[key].to(device)
            elif key.endswith("padding_mask"):
                padding_mask_dict[key] = data_input[key].to(device)
            
        return data_dict, idx_dict, padding_mask_dict

    def init_process(self, mod, args=[], target_cols=None):
        result_dict = {}
        target_cols = self.total_cols if target_cols is None else target_cols
        for col in target_cols:
            result_dict[col] = mod(col, *args)
        
        return torch.nn.ModuleDict(result_dict)

    def apply_process(self, data, mod, args=[], target_cols=None, collate_fn=None):
        result_dict = {}
        target_cols = self.total_cols if target_cols is None else target_cols
        for col in target_cols:
            result_dict[col] = mod[col](col, data[col], *args)
        
        if collate_fn is not None:
            return collate_fn(result_dict)
        else:
            return result_dict

    def tidy_decoding(self, decoding_dict):
        result_dict = {}
        for key, val in decoding_dict.items():
            result_dict[key], self_attn_weight, cross_attn_weight = val
            self.self_attn_weight_dict.update({key:self_attn_weight})
            self.cross_attn_weight_dict.update({key:cross_attn_weight})

        return result_dict
1==1