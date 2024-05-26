import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, data_info, remain_rto):
        super().__init__()
        self.data_info, self.remain_rto = data_info, remain_rto

        # Fit label encoder
        self.label_encoder_dict = self._fit_label_encoder(data)

        # Iterate data
        data_li = []
        data.groupby(self.data_info.modality_info["group"]).progress_apply(data_li.append)
        self.data_li = tuple(data_li)
    
    def _fit_label_encoder(self, data):
        result_dict = {}
        target_cols = self.data_info.processing_info["embedding_cols"]
        for col in target_cols:
            encoder = CustomLabelEncoder()
            encoder.fit(data[col])
            result_dict[col] = encoder
        return result_dict
    
    def __len__(self):
        return len(self.data_li)

    def _transform_label_encoder(self, data):
        result_dict = {}
        for col in data.columns:
            result_dict[col] = self.label_encoder_dict[col].transform(data[col].values)
        return result_dict
    
    def _scale_data(self, data):
        result_dict = {}
        for col in data.columns:
            scaler = self.data_info.processing_info["scaling_cols"][col]
            scaler = scaler()
            result_dict[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
            result_dict[f"{col}_scaler"] = scaler
        
        return result_dict
    
    def _apply_temporal_remain(self, data):
        result_dict = {}
        remain_rto = self.remain_rto["temporal"]

        for col in data.columns:
            val = data[col].values

            num_remain = int(len(val) * remain_rto)
            noise = np.random.rand(data[col].shape[0])
            shuffle_idx = np.argsort(noise)

            remain_idx = shuffle_idx[:num_remain]
            masked_idx = shuffle_idx[num_remain:]
            revert_idx = np.argsort(shuffle_idx)

            remain_padding_mask = np.ones(remain_idx.shape)
            masked_padding_mask = np.ones(masked_idx.shape)
            revert_padding_mask = np.ones(revert_idx.shape)

            result_dict[f"{col}_remain_idx"] = remain_idx
            result_dict[f"{col}_masked_idx"] = masked_idx
            result_dict[f"{col}_revert_idx"] = revert_idx
            result_dict[f"{col}_remain_padding_mask"] = remain_padding_mask
            result_dict[f"{col}_masked_padding_mask"] = masked_padding_mask
            result_dict[f"{col}_revert_padding_mask"] = revert_padding_mask

        return result_dict
    
    def __getitem__(self, idx):
        result_dict = {}
        data = self.data_li[idx]

        # Label encode
        target_cols = self.data_info.processing_info["embedding_cols"]
        result_dict.update(self._transform_label_encoder(data[target_cols]))
        
        # Scale
        target_cols = self.data_info.processing_info["scaling_cols"].keys()
        result_dict.update(self._scale_data(data[target_cols]))

        # Temporal remain mask
        target_cols = self.data_info.modality_info["target"] + self.data_info.modality_info["temporal"]
        result_dict.update(self._apply_temporal_remain(data[target_cols]))

        # Others data
        target_cols = self.data_info.modality_info["others"]
        for col in target_cols:
            val = np.unique(result_dict[col])
            result_dict[col] = val
            assert len(val) == 1, f"Length of the unique value is {len(val)}"

        # Convert to tensor type
        result_dict = {key:torch.from_numpy(val) if not key.endswith("scaler") else key for key, val in result_dict.items()}

        return result_dict

def collate_fn(batch_li, data_info):
    result_dict = {}
    
    # Temporal data
    temporal_cols = data_info.modality_info["target"] + data_info.modality_info["temporal"]
    for col in temporal_cols:
        dtype = torch.int if col in data_info.processing_info["embedding_cols"] else torch.float
        
        data = [batch[col].to(dtype) for batch in batch_li]
        data_remain_idx = [batch[f"{col}_remain_idx"].to(torch.int64) for batch in batch_li]
        data_masked_idx = [batch[f"{col}_masked_idx"].to(torch.int64) for batch in batch_li]
        data_revert_idx = [batch[f"{col}_revert_idx"].to(torch.int64) for batch in batch_li]
        data_remain_padding_mask = [batch[f"{col}_remain_padding_mask"].to(dtype) for batch in batch_li]
        data_masked_padding_mask = [batch[f"{col}_masked_padding_mask"].to(dtype) for batch in batch_li]
        data_revert_padding_mask = [batch[f"{col}_revert_padding_mask"].to(dtype) for batch in batch_li]

        result_dict[col] = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        result_dict[f"{col}_remain_idx"] = torch.nn.utils.rnn.pad_sequence(data_remain_idx, batch_first=True)
        result_dict[f"{col}_masked_idx"] = torch.nn.utils.rnn.pad_sequence(data_masked_idx, batch_first=True)
        result_dict[f"{col}_revert_idx"] = torch.nn.utils.rnn.pad_sequence(data_revert_idx, batch_first=True)
        result_dict[f"{col}_remain_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_remain_padding_mask, batch_first=True)
        result_dict[f"{col}_masked_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_masked_padding_mask, batch_first=True)
        result_dict[f"{col}_revert_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_revert_padding_mask, batch_first=True)

    # Others data
    others_cols = data_info.modality_info["others"]
    for col in others_cols:
        dtype = torch.int if col in data_info.processing_info["embedding_cols"] else torch.float
        result_dict[col] = torch.stack([batch[col].to(dtype) for batch in batch_li])


    # Scalers
    for col in data_info.processing_info["scaling_cols"]:
        result_dict[f"{col}_scaler"] = [batch[f"{col}_scaler"] for batch in batch_li]

    return result_dict



class DataInfo():
    def __init__(self, modality_info, processing_info):
        self.modality_info, self.processing_info = modality_info, processing_info
        self._check_modality()
    
    def _check_modality(self):
        # Modality info
        num_target = len(self.modality_info["target"])
        num_temporal = len(self.modality_info["temporal"])
        num_others = len(self.modality_info["others"])
        num_modality = num_target + num_temporal + num_others

        # Processing info
        num_scaling_cols = len(self.processing_info["scaling_cols"])
        num_embedding_cols = len(self.processing_info["embedding_cols"])
        num_processing = num_scaling_cols + num_embedding_cols

        assert num_modality == num_processing, f"num_modality: {num_modality}, num_processing: {num_processing}"

class NoneScaler(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x
    
    def inverse_transform(self, x, y=None):
        return x

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        unique_x = list(set(x))
        unique_x.append("unknown")
        self.mapper = {i:n for n, i in enumerate(unique_x)}
        self.inverse_mapper = {val:key for key, val in self.mapper.items()}

        return self
    
    def return_mapping(self, x, result):
        if isinstance(x, torch.Tensor):
            return torch.tensor(result)
        
        elif isinstance(x, np.ndarray):
            return np.array(result)
        
        else: 
            raise Exception(f"Data type is neither tensor nor numpy but {type(x)}")
    
    def transform(self, x, y=None):
        result = []
        for i in x:
            if i in self.mapper.keys():
                result.append(self.mapper[i])
            else:
                result.append(self.mapper["unknown"])
        
        return self.return_mapping(x, result)
    
    def inverse_transform(self, x, y=None):
        result = []
        for i in x:
            result.append(self.inverse_mapper[i])
        
        return self.return_mapping(x, result)
    
    def get_num_cls(self):
        return len(self.mapper)

class LogScaler(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.a = 1
        return self
    
    def transform(self, x, y=None):
        if isinstance(x, torch.Tensor):
            return torch.loglp(x)
        elif isinstance(x, np.ndarray):
            return np.log1p(x)
        else: raise Exception(f"Data type is neither tensor nor numpy but {type(x)}")
    
    def inverse_transform(self, x, y=None):
        if isinstance(x, torch.Tensor):
            return torch.expm1(x)
        elif isinstance(x, np.ndarray):
            return np.expm1(x)
        else: raise Exception(f"Data type is neither tensor nor numpy but {type(x)}")

1==1