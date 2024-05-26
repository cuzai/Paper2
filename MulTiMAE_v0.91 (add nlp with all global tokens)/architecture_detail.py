import copy
import torch
from architecture_detail_detail import *
from utils import *

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

class _EncoderLayer(torch.nn.TransformerEncoderLayer):
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

class _Encoder(torch.nn.TransformerEncoder):
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



class Embedding(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.modality_info["target"] + data_info.modality_info["temporal"]:
            self.embedding = TemporalEmbedding(col, data_info, label_encoder_dict, d_model)
        elif col in data_info.modality_info["img"]:
            self.embedding = ImgEmbedding(d_model)
        elif col in data_info.modality_info["nlp"]:
            self.embedding = NlpEmbedding(d_model)
    
    def forward(self, key, val, padding_mask, device):
        return self.embedding(key, val, padding_mask, device)

class Remain(torch.nn.Module):
    def __init__(self, col, data_info, d_model, dropout):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.global_token = torch.nn.Parameter(torch.rand(1, d_model))
        
        if col in data_info.modality_info["target"] + data_info.modality_info["temporal"]:
            self.remain = TemporalRemain()
        elif col in data_info.modality_info["img"]:
            self.remain = ImgRemain()
        elif col in data_info.modality_info["nlp"]:
            self.remain = NlpRemain()
    
    def forward(self, key, val, idx_dict, remain_rto, device):
        return self.remain(key, val, self.global_token, self.pos_enc, idx_dict, remain_rto, device)

class ModalityEmbedding(torch.nn.Module):
    def __init__(self, col, num_modality, modality, d_model):
        super().__init__()
        self.modality = modality[col]
        self.modality_embedding = torch.nn.Embedding(num_modality, d_model)
    
    def forward(self, key, val, device):
        modality = torch.zeros(val.shape[1]).to(torch.int).to(device) + self.modality
        modality = self.modality_embedding(modality)
        
        return val + modality

class Encoder(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        # self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation=activation, norm_first=True), num_layers)
        self.encoder = _Encoder(_EncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation=activation, norm_first=True), num_layers)
    
    def forward(self, data_dict, padding_mask_dict, total_cols, device):
        # Concat data and padding_mask
        data_concat_li, padding_mask_concat_li = [], []
        for col in total_cols:
            data = data_dict[col]
            
            padding_mask = padding_mask_dict[f"{col}_remain_padding_mask"]
            padding_mask_for_global_token = torch.ones(data.shape[0], 1).to(device)
            new_padding_mask = torch.cat([padding_mask_for_global_token, padding_mask], dim=1)

            data_concat_li.append(data)
            padding_mask_concat_li.append(new_padding_mask)
        
        data_concat = torch.cat(data_concat_li, dim=1)
        padding_mask_concat = torch.cat(padding_mask_concat_li, dim=1)
        padding_mask_concat = torch.where(padding_mask_concat==1, 0, -torch.inf)
        assert data_concat.shape[:-1] == padding_mask_concat.shape

        encoding, attn_weight = self.encoder(data_concat, src_key_padding_mask=padding_mask_concat)

        return encoding, attn_weight

class Split(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data_to_split, data_for_length, total_cols):
        split_dict = {}
        start_idx = 0

        for col in total_cols:
            length = data_for_length[col].shape[1]
            split_dict[col] = data_to_split[:, start_idx:start_idx+length, :]
            start_idx += length
        assert start_idx == data_to_split.shape[1]

        return split_dict

class Revert(torch.nn.Module):
    def __init__(self, col, d_model, dropout):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.mask_token = torch.nn.Parameter(torch.rand(1, d_model))
        self.revert = DynamicRevert()
        
    def forward(self, key, val, idx_dict, padding_mask_dict):
        return self.revert(key, val, self.mask_token, self.pos_enc, idx_dict, padding_mask_dict)


class TemporalDecoder(torch.nn.Module):
    def __init__(self, col, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(TemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation)) for _ in range(num_layers)])
    
    def forward(self, key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device):
        tgt = val

        # Get memory
        memory_li = []
        ### Temporal memory
        for col in temporal_cols:
            if col == key: continue
            memory_li.append(data_dict[col])
        memory = torch.stack(memory_li, dim=-2)
        
        ### Img memory & Nlp memory
        for col in img_cols + nlp_cols:
            img_data = data_dict[col].unsqueeze(1).repeat(1, memory.shape[1], 1, 1)
            memory = torch.cat([memory, img_data], dim=-2)
        
        # Obtain self_attn_padding_mask
        self_attn_padding_mask = padding_mask_dict[f"{key}_revert_padding_mask"]
        padding_mask_for_global_token = torch.ones(val.shape[0], 1).to(device)
        self_attn_padding_mask = torch.cat([padding_mask_for_global_token, self_attn_padding_mask], dim=-1)
        self_attn_padding_mask = torch.where(self_attn_padding_mask==1, 0, -torch.inf)

        # Apply decoder
        for mod in self.layers:
            tgt, self_attn_weight, cross_attn_weight = mod(tgt, memory, self_attn_padding_mask)
            
        return tgt, self_attn_weight, cross_attn_weight

class NonTemporalDecoder(torch.nn.Module):
    def __init__(self, col, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(NonTemporalDecoderLayer(d_model, nhead, d_ff, dropout, activation)) for _ in range(num_layers)])
    
    def forward(self, key, val, data_dict, padding_mask_dict, temporal_cols, img_cols, nlp_cols, device):
        tgt = val

        # Get memory and cross_attn_padding_mask
        memory_li, cross_attn_padding_mask_li = [], []
        ### Temporal memory & Nlp memory
        for col in temporal_cols + nlp_cols:
            temporal_data = data_dict[col]
            padding_mask = padding_mask_dict[f"{col}_revert_padding_mask"]
            padding_mask_for_global_token = torch.ones(val.shape[0], 1).to(device)
            new_padding_mask = torch.cat([padding_mask_for_global_token, padding_mask], dim=-1)
            
            memory_li.append(temporal_data)
            cross_attn_padding_mask_li.append(new_padding_mask)
        
        memory = torch.cat(memory_li, dim=1)
        cross_attn_padding_mask = torch.cat(cross_attn_padding_mask_li, dim=1)
        cross_attn_padding_mask = torch.where(cross_attn_padding_mask == 1, 0, -torch.inf)

        # Obtain self_attn_padding_mask
        self_attn_padding_mask = padding_mask_dict[f"{key}_revert_padding_mask"]
        padding_mask_for_global_token = torch.ones(val.shape[0], 1).to(device)
        self_attn_padding_mask = torch.cat([padding_mask_for_global_token, self_attn_padding_mask], dim=-1)
        self_attn_padding_mask = torch.where(self_attn_padding_mask==1, 0, -torch.inf)

        # Apply decoder
        for mod in self.layers:
            tgt, self_attn_weight, cross_attn_weight = mod(tgt, memory, cross_attn_padding_mask, self_attn_padding_mask)
            
        return tgt, self_attn_weight, cross_attn_weight


class TemporalOutput(torch.nn.Module):
    def __init__(self, col, data_info, label_encoder_dict, d_model):
        super().__init__()
        if col in data_info.processing_info["scaling_cols"]:
            self.output = torch.nn.Sequential(
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, 1))
        
        elif col in data_info.processing_info["embedding_cols"]:
            num_cls = label_encoder_dict[col].get_num_cls()
            self.output = torch.nn.Sequential(
                            torch.nn.Linear(d_model, d_model),
                            torch.nn.Linear(d_model, num_cls))

    
    def forward(self, key, val):
        return self.output(val)

class ImgOutput(torch.nn.Module):
    def __init__(self, col, d_model, patch_size):
        super().__init__()
        self.output = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Linear(d_model, 3*patch_size*patch_size))
    
    def forward(self, key, val):
        return self.output(val)
        
class NlpOutput(torch.nn.Module):
    def __init__(self, col, d_model, num_vocab=30522):
        super().__init__()
        self.output = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Linear(d_model, num_vocab))
    
    def forward(self, key, val):
        return self.output(val)
1==1