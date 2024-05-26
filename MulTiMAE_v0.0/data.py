import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import torch
from transformers import AutoImageProcessor
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, data_info, remain_rto_general, is_train=True, label_encoder_dict=None):
        self.data_info, self.remain_rto_general, self.is_train = data_info, remain_rto_general, is_train

        # Fit label encoder
        if self.is_train:
            label_encoder_dict = self.encode_label(data, self.data_info.processing_info["embedding_cols"])
        else:
            assert label_encoder_dict is not None, "label_encoder_dict must be provided in test mode"
        self.label_encoder_dict = label_encoder_dict

        # Iterate each product
        self.data_li = []
        data.groupby(data_info.modality_info["group"]).progress_apply(lambda x: self.data_li.append(x))

    def encode_label(self, data, embedding_cols):
        label_encoder_dict = {}
        for col in embedding_cols:
            encoder = CustomLabelEncoder()
            encoder.fit(data[col])
            label_encoder_dict[col] = encoder
        return label_encoder_dict

    def __len__(self):
        return len(self.data_li)

    def _label_encode(self, data):
        result_dict = {}
        # Label encode
        for col in self.data_info.processing_info["embedding_cols"]:
            encoded = self.label_encoder_dict[col].transform(data[col])
            if col in self.data_info.modality_info["others"]:
                result_dict[col] = torch.IntTensor(encoded[[0]]).squeeze()
            else:
                result_dict[col] = torch.IntTensor(encoded)
            
        return result_dict
    
    def _scale(self, data):
        result_dict = {}
        for col, scaler in self.data_info.processing_info["scaling_cols"].items():
            scaler = scaler()
            result_dict[col] = torch.Tensor(scaler.fit_transform(data[col].values.reshape(-1,1)))
            result_dict[f"{col}_scaler"] = scaler
        
        return result_dict
    
    def _process_img(self, data):
        result_dict = {}
        transform = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        for col in self.data_info.processing_info["img_cols"]:
            img_path = data[f"{col}"].values[0]
            img_raw = Image.open(img_path).convert("RGB")
            result_dict[f"{col}_img_raw"] = img_raw
            result_dict[f"{col}_img_input"] = transform(img_raw, return_tensors="pt", train=False)["pixel_values"].squeeze(0)
        return result_dict

    def _get_mask(self, data):
        num_remain = int(len(data) * self.remain_rto_general) if self.is_train else -90

        # Get index for shuffled remain and revert
        if self.is_train:
            noise = torch.rand(len(data))
            shuffle_idx = torch.argsort(noise, dim=-1)
        else:
            shuffle_idx = torch.arange(data.shape[0])
        
        remain_idx = shuffle_idx[:num_remain]
        masked_idx = shuffle_idx[num_remain:]
        revert_idx = torch.argsort(shuffle_idx, dim=-1)

        remain_padding_mask = torch.ones(remain_idx.shape)
        masked_padding_mask = torch.ones(masked_idx.shape)
        revert_padding_mask = torch.ones(revert_idx.shape)

        return remain_idx, masked_idx, revert_idx, remain_padding_mask, masked_padding_mask, revert_padding_mask

    def _mask_temporal(self, data, col_li, is_target):
        result_dict = {}
        for col in col_li:
            remain_idx, masked_idx, revert_idx, remain_padding_mask, masked_padding_mask, revert_padding_mask = self._get_mask(data[col])

            result_dict[f"{col}_remain_idx"] = remain_idx
            result_dict[f"{col}_masked_idx"] = masked_idx
            result_dict[f"{col}_revert_idx"] = revert_idx

            if is_target :
                result_dict[f"temporal_remain_padding_mask"] = remain_padding_mask
                result_dict[f"temporal_masked_padding_mask"] = masked_padding_mask
                result_dict[f"temporal_revert_padding_mask"] = revert_padding_mask
    
        return result_dict
    
    def __getitem__(self, idx):
        data = self.data_li[idx]
        data_input = {}

        # Label encode
        data_input.update(self._label_encode(data))
        # Scale
        data_input.update(self._scale(data))
        # Process image
        data_input.update(self._process_img(data))

        # Mask temporal
        data_input.update(self._mask_temporal(data_input, self.data_info.modality_info["target"], is_target=True))
        data_input.update(self._mask_temporal(data_input, self.data_info.modality_info["temporal"], is_target=False))

        return data_input

def collate_fn(batch_li, data_info):
    result_dict = {}
    # Respective temporal features
    for col in data_info.modality_info["target"] + data_info.modality_info["temporal"]:
        data = [batch[col] for batch in batch_li]
        data_remain_idx = [batch[f"{col}_remain_idx"] for batch in batch_li]
        data_masked_idx = [batch[f"{col}_masked_idx"] for batch in batch_li]
        data_revert_idx = [batch[f"{col}_revert_idx"] for batch in batch_li]

        result_dict[col] = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        result_dict[f"{col}_remain_idx"] = torch.nn.utils.rnn.pad_sequence(data_remain_idx, batch_first=True)
        result_dict[f"{col}_masked_idx"] = torch.nn.utils.rnn.pad_sequence(data_masked_idx, batch_first=True)
        result_dict[f"{col}_revert_idx"] = torch.nn.utils.rnn.pad_sequence(data_revert_idx, batch_first=True)
    
    # Global temporal padding mask
    for col in data_info.modality_info["target"]:
        temporal_remain_padding_mask = [batch[f"temporal_remain_padding_mask"] for batch in batch_li]
        temporal_masked_padding_mask = [batch[f"temporal_masked_padding_mask"] for batch in batch_li]
        temporal_revert_padding_mask = [batch[f"temporal_revert_padding_mask"] for batch in batch_li]

        result_dict[f"temporal_remain_padding_mask"] = torch.nn.utils.rnn.pad_sequence(temporal_remain_padding_mask, batch_first=True)
        result_dict[f"temporal_masked_padding_mask"] = torch.nn.utils.rnn.pad_sequence(temporal_masked_padding_mask, batch_first=True)
        result_dict[f"temporal_revert_padding_mask"] = torch.nn.utils.rnn.pad_sequence(temporal_revert_padding_mask, batch_first=True)
    
    # Process img feature
    for col in data_info.modality_info["img"]:
        result_dict[f"{col}_img_input"] = torch.stack([batch[f"{col}_img_input"] for batch in batch_li])
        result_dict[f"{col}_img_raw"] = [batch[f"{col}_img_raw"] for batch in batch_li]
    
    # Process others features
    for col in data_info.modality_info["others"]:
        result_dict[col] = torch.stack([batch[col] for batch in batch_li]).squeeze()
    
    return result_dict

class DataInfo():
    def __init__(self, modality_info, processing_info):
        self.modality_info, self.processing_info = modality_info, processing_info
        print(self.modality_info)
        self.check_validity(self.modality_info, self.processing_info)
        
        self.num_modality = self.get_num_modality(modality_info)
    
    def check_validity(self, modality_info, processing_info):
        # Modality info
        num_target = len(modality_info["target"])
        num_temporal = len(modality_info["temporal"])
        num_img_path = len(modality_info["img"])
        num_static = len(modality_info["others"])
        num_modality_info = num_target + num_temporal + num_static + num_img_path

        # Processing info
        num_scaling_cols = len(processing_info["scaling_cols"].keys())
        num_embedding_cols = len(processing_info["embedding_cols"])
        num_img_cols = len(processing_info["img_cols"])
        num_processing_info = num_scaling_cols + num_embedding_cols + num_img_cols

        assert num_modality_info == num_processing_info, f"num_modality_info is {num_modality_info} while num_processing_info is {num_processing_info}"

    def get_num_modality(self, data_info):
        num_target = len(data_info["target"])
        num_temporal = len(data_info["temporal"])
        num_img_path = len(data_info["img"])
        num_static = len(data_info["others"])
        
        num_modality = num_target + num_temporal + num_static + num_img_path
        return num_modality

class LogScaler(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.a = 1
        return self
    
    def transform(self, x, y=None):
        return np.log1p(x)
    
    def inverse_transform(self, x, y=None):
        return np.expm1(x)
    
class NoneScaler(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        x = x.copy()
        return x
    
    def inverse_transform(self, x, y=None):
        x = x.copy()
        return x

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapper = {}
        self.idx = 0

    def fit(self, x, y=None):
        x = set(x)
        for val in x:
            if val not in self.mapper.keys():
                self.mapper[val] = self.idx
                self.idx += 1
        self.mapper["unseen"] = self.idx
        self.idx += 1
        return self
    
    def transform(self, x, y=None):
        res = []
        for val in x:
            if val in self.mapper.keys():
                res.append(self.mapper[val])
            else:
                res.append(self.idx)
        return np.array(res)
    
    def inverse_transform(self, idx):
        inverse_mapper = {val:key for key, val in self.mapper.items()}
        return inverse_mapper[idx]