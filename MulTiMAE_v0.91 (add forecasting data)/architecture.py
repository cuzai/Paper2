import torch
from architecture_detail import *
from utils import *

class Transformer(torch.nn.Module):
    def __init__(self, data_info, label_encoder_dict,
                d_model, num_layers, nhead, d_ff, dropout, activation):
        super().__init__()
        self.data_info, self.label_encoder_dict = data_info, label_encoder_dict
        self.temporal_cols, self.img_cols, self.nlp_cols, self.total_cols = self.define_cols()
        self.self_attn_weight_dict, self.cross_attn_weight_dict = {}, {}

        # 1. Embedding
        self.embedding_dict = self.init_process(mod=Embedding, args=[self.data_info, self.label_encoder_dict, d_model["encoder"]])
        # 2.Remain mask
        self.remain_dict = self.init_process(mod=Remain, args=[self.data_info, d_model["encoder"], dropout])
        # 3. Modality embedding
        num_modality = len(self.total_cols)
        modality = {col:n for n, col in enumerate(self.total_cols)}
        self.encoder_modality_embedding_dict = self.init_process(mod=ModalityEmbedding, args=[num_modality, modality, d_model["encoder"]])
        # 4. Encoding
        self.encoding = Encoder(d_model["encoder"], nhead, d_ff["encoder"], dropout, activation, num_layers["encoder"])
        self.to_decoder_dim = torch.nn.Linear(d_model["encoder"], d_model["decoder"])
        # 5. Split
        self.split = Split()
        # 6. Revert
        self.revert_dict = self.init_process(mod=Revert, args=[d_model["decoder"], dropout])
        # 7. Modality embeding
        self.decoder_modality_embedding_dict = self.init_process(mod=ModalityEmbedding, args=[num_modality, modality, d_model["decoder"]])
        # 8. Decoding
        self.temporal_decoding_dict = self.init_process(mod=TemporalDecoder, args=[d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"]], target_cols=self.temporal_cols)
        self.non_temporal_decoding_dict = self.init_process(mod=NonTemporalDecoder, args=[d_model["decoder"], nhead, d_ff["decoder"], dropout, activation, num_layers["decoder"]], target_cols=self.img_cols+self.nlp_cols)

        # 9. Output
        self.temporal_output = self.init_process(mod=TemporalOutput, args=[self.data_info, self.label_encoder_dict, d_model["decoder"]], target_cols=self.temporal_cols)
        self.img_output = self.init_process(mod=ImgOutput, args=[d_model["decoder"]], target_cols=self.img_cols)
        self.nlp_output = self.init_process(mod=NlpOutput, args=[d_model["decoder"]], target_cols=self.nlp_cols)

    def forward(self, data_input, remain_rto, device):
        data_dict, self.idx_dict, self.padding_mask_dict = self.to_gpu(data_input, device)
        
        # 1. Embedding
        embedding_dict = self.apply_process(data=data_dict, mod=self.embedding_dict, args=[self.padding_mask_dict, device])
        # 2. Remain mask
        remain_dict = self.apply_process(data=embedding_dict, mod=self.remain_dict, args=[self.idx_dict, remain_rto, device], collate_fn=self.tidy_remain)
        # 3. Modality embedding
        encoder_modality_embedding_dict = self.apply_process(data=remain_dict, mod=self.encoder_modality_embedding_dict, args=[device])
        # 4. Encoding
        encoding = self.encoding(encoder_modality_embedding_dict, self.padding_mask_dict, self.total_cols, device)
        encoding = self.to_decoder_dim(encoding)
        # 5. Split
        split_dict = self.split(encoding, encoder_modality_embedding_dict, self.total_cols)
        # 6. Revert
        revert_dict = self.apply_process(data=split_dict, mod=self.revert_dict, args=[self.idx_dict, self.padding_mask_dict])
        # 7. Modality embedding
        decoder_modality_embedding_dict = self.apply_process(data=revert_dict, mod=self.decoder_modality_embedding_dict, args=[device])
        # 8. Decoding
        temporal_decoding_dict = self.apply_process(data=decoder_modality_embedding_dict, mod=self.temporal_decoding_dict, args=[decoder_modality_embedding_dict, self.padding_mask_dict, self.temporal_cols, self.img_cols, self.nlp_cols, device], collate_fn=self.tidy_decoding, target_cols=self.temporal_cols)
        non_temporal_decoding_dict = self.apply_process(data=decoder_modality_embedding_dict, mod=self.non_temporal_decoding_dict, args=[decoder_modality_embedding_dict, self.padding_mask_dict, self.temporal_cols, self.img_cols, self.nlp_cols, device], collate_fn=self.tidy_decoding, target_cols=self.img_cols+self.nlp_cols)
        # 9. Output
        temporal_output = self.apply_process(data=temporal_decoding_dict, mod=self.temporal_output, target_cols=self.temporal_cols)
        img_output = self.apply_process(data=non_temporal_decoding_dict, mod=self.img_output, target_cols=self.img_cols)
        nlp_output = self.apply_process(data=non_temporal_decoding_dict, mod=self.nlp_output, target_cols=self.nlp_cols)

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
    
    def tidy_remain(self, remain_dict):
        result_dict = {}
        for col in self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]:
            result_dict[col] = remain_dict[col]

        for col in self.data_info.modality_info["img"]:
            val, remain_idx, masked_idx, revert_idx, remain_padding_mask, revert_padding_mask = remain_dict[col]
            result_dict[col] = val
            self.idx_dict.update({f"{col}_remain_idx":remain_idx,
                                f"{col}_masked_idx":masked_idx,
                                f"{col}_revert_idx":revert_idx})
            self.padding_mask_dict.update({f"{col}_remain_padding_mask":remain_padding_mask,
                                            f"{col}_revert_padding_mask":revert_padding_mask})
        
        for col in self.data_info.modality_info["nlp"]:
            result_dict[col] = remain_dict[col]
            
        return result_dict

    def tidy_decoding(self, decoding_dict):
        result_dict = {}
        for key, val in decoding_dict.items():
            result_dict[key], self_attn_weight, cross_attn_weight = val
            self.self_attn_weight_dict.update({key:self_attn_weight})
            self.cross_attn_weight_dict.update({key:cross_attn_weight})

        return result_dict