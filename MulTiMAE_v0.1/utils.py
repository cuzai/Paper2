import torch

def apply_remain_mask(data, remain_rto, device):
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

def _apply_revert(remain_data, mask_token, revert_idx, device, remain_padding_mask=None):
    # Revert index
    revert_idx += 1
    global_token_idx = torch.zeros(revert_idx.shape[0], 1).to(torch.int).to(device)
    revert_idx = torch.cat([global_token_idx, revert_idx], dim=1)
    revert_idx = revert_idx.unsqueeze(-1).repeat(1, 1, remain_data.shape[-1])

    # Replace padding_mask to mask_token
    if remain_padding_mask is not None:
        remain_padding_mask = torch.cat([global_token_idx, remain_padding_mask], dim=1)
        remain_padding_mask = remain_padding_mask.unsqueeze(-1).repeat(1, 1, remain_data.shape[-1])
        remain_data = torch.where(remain_padding_mask==1, remain_data, mask_token)

    # Append missing
    mask_tokens = mask_token.unsqueeze(0).repeat(revert_idx.shape[0], revert_idx.shape[1] - remain_data.shape[1], 1)
    full_embedding = torch.cat([remain_data, mask_tokens], dim=1)
    
    return torch.gather(full_embedding, index=revert_idx, dim=1)


class DynamicEmbedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
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

class OthersRemain(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.global_token = torch.nn.Parameter(torch.rand(1, d_model))
    
    def forward(self, data, others_pos_emb, remain_rto, device):
        # Positional embedding
        modality = (torch.arange(data.shape[1]+1)).to(device)
        pos_emb = others_pos_emb(modality)
        data += pos_emb[1:, :]

        # Get remain
        data, remain_idx, masked_idx, revert_idx = apply_remain_mask(data, remain_rto, device)
        idx_dict = {"others_remain_idx": remain_idx, "others_masked_idx": masked_idx, "others_revert_idx": revert_idx}

        # Global token
        global_token = self.global_token.unsqueeze(0).repeat(data.shape[0], 1, 1)
        global_token += pos_emb[0, :]
        data = torch.cat([global_token, data], dim=1)
        return data, idx_dict

class TemporalRemain(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.global_token = torch.nn.Parameter(torch.rand(1, d_model))
    
    def forward(self, data, temporal_pos_enc, remain_idx):
        # Positional encoding
        pos_enc = temporal_pos_enc[:data.shape[1]+1, :]
        data += pos_enc[1:, :]

        # Get remain
        remain_idx = remain_idx.unsqueeze(-1).repeat(1, 1, data.shape[-1])
        data = torch.gather(data, index=remain_idx, dim=1)
        
        # Global token
        global_token = self.global_token.unsqueeze(0).repeat(data.shape[0], 1, 1)
        global_token += pos_enc[0, :]
        data = torch.cat([global_token, data], dim=1)

        return data

class ImgRemain(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.global_token = torch.nn.Parameter(torch.rand(1, d_model))
    
    def forward(self, data, img_pos_enc, remain_rto, col, device):
        # Positional encoding
        pos_enc = img_pos_enc[:data.shape[1]+1]
        data += pos_enc[1:, :]
        
        # Get remain
        data, remain_idx, masked_idx, revert_idx = apply_remain_mask(data, remain_rto["general"], device)
        idx_dict = {f"{col}_img_input_remain_idx": remain_idx, f"{col}_img_input_masked_idx": masked_idx, f"{col}_img_input_revert_idx": revert_idx}

        # Global token
        global_token = self.global_token.unsqueeze(0).repeat(data.shape[0], 1, 1)
        global_token += pos_enc[0, :]
        data = torch.cat([global_token, data], dim=1)

        return data, idx_dict

class OthersRevert(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.rand(1, d_model))
    
    def forward(self, others_data, revert_idx, others_pos_emb, device):
        reverted_data = _apply_revert(others_data, self.mask_token, revert_idx, device)
    
        # Positional embedding
        modality = (torch.arange(reverted_data.shape[1])).to(device)
        pos_emb = others_pos_emb(modality).unsqueeze(0).repeat(reverted_data.shape[0] , 1, 1)
        reverted_data += pos_emb
        
        return reverted_data
    
class TemporalRevert(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.rand(1, d_model))
    
    def forward(self, temporal_data, revert_idx, temporal_pos_enc, remain_padding_mask, device):
        reverted_data = _apply_revert(temporal_data, self.mask_token, revert_idx, device, remain_padding_mask=remain_padding_mask)

        # Positional encoding
        pos_enc = temporal_pos_enc[:reverted_data.shape[1], :]
        reverted_data += pos_enc

        return reverted_data

class ImgRevert(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.rand(1, d_model))
    
    def forward(self, img_data, revert_idx, img_pos_enc, device):
        reverted_data = _apply_revert(img_data, self.mask_token, revert_idx, device)
        
        # Positional encoding
        pos_enc = img_pos_enc[:reverted_data.shape[1], :]
        reverted_data += pos_enc

        return reverted_data

class DynamicOutput(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.processing_info["scaling_cols"]:
            self.output = torch.nn.Sequential(
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, 1)
                            )
        elif col in data_info.processing_info["embedding_cols"]:
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