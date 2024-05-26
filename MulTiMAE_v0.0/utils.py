import torch

def get_positional_encoding(d_model, seq_len=1000):
    position = torch.arange(seq_len).reshape(-1,1)
    i = torch.arange(d_model)//2
    exp_term = 2*i/d_model
    div_term = torch.pow(10000, exp_term).reshape(1, -1)
    pos_encoded = position / div_term

    pos_encoded[:, 0::2] = torch.sin(pos_encoded[:, 0::2])
    pos_encoded[:, 1::2] = torch.cos(pos_encoded[:, 1::2])

    return pos_encoded

def apply_mask(data, remain_rto, device):
    num_remain = int(data.shape[1] * remain_rto)

    # Index for shuffle and revert
    noise = torch.rand(data.shape[:-1]).to(device)
    shuffle_idx = torch.argsort(noise, dim=-1)

    remain_idx = shuffle_idx[:, :num_remain]
    masked_idx = shuffle_idx[:, num_remain:]
    revert_idx = torch.argsort(shuffle_idx, dim=-1)

    # Apply mask
    remain_idx_ = remain_idx.unsqueeze(-1).repeat(1, 1, data.shape[-1])
    data = torch.gather(data, index=remain_idx_, dim=1)

    return data, remain_idx, masked_idx, revert_idx

def _apply_revert(remain_encoding, mask_token, revert_idx, device, remain_padding_mask=None):
    # Process for revert index
    idx_for_global_token = torch.zeros(revert_idx.shape[0], 1).to(device).to(torch.int)
    revert_idx += 1 # Shift one step for global token
    revert_idx = torch.cat([idx_for_global_token, revert_idx], dim=-1)
    revert_idx = revert_idx.unsqueeze(-1).repeat(1, 1, remain_encoding.shape[-1])

    # Append mask tokens for missing positions
    if remain_padding_mask is not None:
        ### Replace remain padding to mask token
        idx_for_global_token = torch.ones(remain_padding_mask.shape[0], 1).to(device).to(torch.int)
        remain_padding_mask = torch.cat([idx_for_global_token, remain_padding_mask], dim=-1)
        remain_padding_mask = remain_padding_mask.unsqueeze(-1).repeat(1, 1, remain_encoding.shape[-1])
        remain_encoding = torch.where(remain_padding_mask==1, remain_encoding, mask_token)

    ### Append missing
    mask_tokens = mask_token.unsqueeze(0).repeat(revert_idx.shape[0],
                                                revert_idx.shape[1] - remain_encoding.shape[1],
                                                1)
    full_embedding = torch.cat([remain_encoding, mask_tokens], dim=1)

    # Revert
    reverted_embedding = torch.gather(full_embedding, index=revert_idx, dim=1)
                                                
    return reverted_embedding

class DynamicEmbedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, 
                        d_model):
        super().__init__()

        if col in data_info.processing_info["scaling_cols"]:
            self.embedding = torch.nn.Linear(1, d_model)
        
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].idx
            self.embedding = torch.nn.Embedding(num_cls, d_model)
    
    def forward(self, data):
        return self.embedding(data)

class ImgEmbedding(torch.nn.Module):
    def __init__(self, d_model, patch_size, img_size=224):
        super().__init__()
        # Embedding
        self.conv = torch.nn.Conv2d(3, d_model, patch_size, patch_size)
    
    def forward(self, img):
        # Embedding
        patches = self.conv(img).permute(0, 2, 3, 1)
        bs, img_h, img_w, d_model = patches.shape
        embedding = patches.view(bs, -1, d_model)
        
        return embedding

class TemporalRemain(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.pos_enc = torch.nn.Parameter(get_positional_encoding(d_model))
        self.pos_enc = get_positional_encoding(d_model)
        self.global_token = torch.nn.Parameter(torch.zeros(1, d_model))
    
    def forward(self, data, remain_idx, device):
        # Positional encoding
        pos_enc = self.pos_enc[:data.shape[1]+1, :].to(device)
        data += pos_enc[1:, :]

        # Get remain
        remain_idx = remain_idx.unsqueeze(-1).repeat(1, 1, data.shape[-1])
        data = torch.gather(data, index=remain_idx, dim=1)
        
        # Global token
        global_token = self.global_token.unsqueeze(0).repeat(data.shape[0], 1, 1)
        global_token += pos_enc[0, :]
        data = torch.cat([global_token, data], dim=1)

        return data, remain_idx

class ImgRemain(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.pos_enc = torch.nn.Parameter(get_positional_encoding(d_model))
        self.pos_enc = get_positional_encoding(d_model)
        self.global_token = torch.nn.Parameter(torch.zeros(1, d_model))
    
    def forward(self, data, remain_rto, device):
        # Positional encoding
        pos_enc = self.pos_enc[:data.shape[1]+1].to(device)
        data += pos_enc[1:, :]
        
        # Get remain
        data, remain_idx, masked_idx, revert_idx = apply_mask(data, remain_rto, device)

        # Global token
        global_token = self.global_token.unsqueeze(0).repeat(data.shape[0], 1, 1)
        global_token += pos_enc[0, :]
        data = torch.cat([global_token, data], dim=1)

        return data, remain_idx, masked_idx, revert_idx

class OthersRemain(torch.nn.Module):
    def __init__(self, data_info, d_model):
        super().__init__()
        self.num_modality = data_info.num_modality
        self.pos_emb = torch.nn.Embedding(data_info.num_modality+1, d_model)
        self.global_token = torch.nn.Parameter(torch.zeros(1, d_model))
    
    def forward(self, temporal_remain_dict, img_remain_dict, others_dict, remain_rto, device):
        # Modality embedding
        target_cols = list(temporal_remain_dict.keys()) + list(img_remain_dict.keys()) + list(others_dict.keys())
        modality_idx = 1
        for col in target_cols:
            if col in temporal_remain_dict.keys():
                modality = torch.zeros(temporal_remain_dict[col].shape[:-1]).to(device) + modality_idx
                temporal_remain_dict[col] += self.pos_emb(modality.to(torch.int))
            elif col in img_remain_dict.keys():
                modality = torch.zeros(img_remain_dict[col].shape[:-1]).to(device) + modality_idx
                img_remain_dict[col] += self.pos_emb(modality.to(torch.int))
            elif col in others_dict.keys():
                modality = torch.zeros(others_dict[col].shape[:-1]).to(device) + modality_idx
                others_dict[col] += self.pos_emb(modality.to(torch.int))

            modality_idx += 1
        
        # Get remain
        if len(others_dict.values()) > 0 :
            others_data = torch.stack(list(others_dict.values()), dim=1)
            others_remain_data, others_remain_idx, others_masked_idx, others_revert_idx = apply_mask(others_data, remain_rto, device)
        
            # Global token
            global_token = self.global_token.unsqueeze(0).repeat(others_remain_data.shape[0], 1, 1)
            modality = torch.zeros(global_token.shape[:-1]).to(device)
            global_token += self.pos_emb(modality.to(torch.int))
            others_remain_data = torch.cat([global_token, others_remain_data], dim=1)
        
        else: 
            others_remain_data, others_remain_idx, others_masked_idx = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
            global_token = self.global_token.unsqueeze(0).repeat(temporal_remain_dict[list(temporal_remain_dict.keys())[0]].shape[0], 1, 1)
            others_revert_idx = torch.zeros(global_token.shape[:-1]).to(device).to(torch.long)

            modality = torch.zeros(global_token.shape[:-1]).to(device)
            global_token += self.pos_emb(modality.to(torch.int))
            others_remain_data = torch.cat([global_token, others_remain_data], dim=1)

        others_idx_dict = {"others_remain_idx": others_remain_idx, "others_masked_idx": others_masked_idx, "others_revert_idx": others_revert_idx}

        return temporal_remain_dict, img_remain_dict, others_remain_data, others_idx_dict

class TemporalRevert(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.pos_enc = torch.nn.Parameter(get_positional_encoding(d_model))
        self.pos_enc = get_positional_encoding(d_model)
    
    def forward(self, data, mask_token, revert_idx, device, padding_mask):
        revert_embedding = _apply_revert(data, mask_token, revert_idx, device, padding_mask)

        # Positional encdoing
        pos_enc = self.pos_enc[:revert_embedding.shape[1], :].to(device)
        revert_embedding += pos_enc

        return revert_embedding

class ImgRevert(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.pos_enc = torch.nn.Parameter(get_positional_encoding(d_model))
        self.pos_enc = get_positional_encoding(d_model)
    
    def forward(self, data, mask_token, revert_idx, device):
        # Revert
        revert_embedding = _apply_revert(data, mask_token, revert_idx, device)

        # Positional encdoing
        pos_enc = self.pos_enc[:revert_embedding.shape[1], :].to(device)
        revert_embedding += pos_enc

        return revert_embedding
    
class OthersRevert(torch.nn.Module):
    def __init__(self, data_info, d_model):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(data_info.num_modality+1, d_model)
    
    def forward(self, temporal_revert_dict, img_revert_dict, others_remain_data, mask_token, revert_idx, data_info, device):
        # Revert
        others_revert = _apply_revert(others_remain_data, mask_token, revert_idx, device)

        # Modality embedding
        target_cols = data_info.modality_info["target"] + data_info.modality_info["temporal"] + data_info.modality_info["img"] + data_info.modality_info["others"]
        modality_idx = 1
        others_idx = 0
        for col in target_cols:
            if col in temporal_revert_dict.keys():
                modality = torch.zeros(temporal_revert_dict[col].shape[:-1]).to(device) + modality_idx
                temporal_revert_dict[col] += self.pos_emb(modality.to(torch.int))
            elif col in img_revert_dict.keys():
                modality = torch.zeros(img_revert_dict[col].shape[:-1]).to(device) + modality_idx
                img_revert_dict[col] += self.pos_emb(modality.to(torch.int))
            elif col in data_info.modality_info["others"]:
                modality = torch.zeros(others_revert[:, others_idx, :].shape[:-1]).to(device) + modality_idx
                others_revert[:, others_idx, :] += self.pos_emb(modality.to(torch.int))
                others_idx += 1

            modality_idx += 1

        return temporal_revert_dict, img_revert_dict, others_revert

class DynamicOutput(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.processing_info["scaling_cols"]:
            self.output = torch.nn.Sequential(
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, 1)
                            )
        if col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].idx
            self.output = torch.nn.Sequential(
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, num_cls)
                            )

    def forward(self, data):
        return self.output(data)

class ImgOutput(torch.nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.output = torch.nn.Sequential(
                        torch.nn.Linear(d_model, d_model), 
                        torch.nn.Linear(d_model, d_model), 
                        torch.nn.Linear(d_model, 3*patch_size*patch_size)
                        )
    
    def forward(self, data):
        return self.output(data)