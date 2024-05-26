import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, data_info, remain_rto, is_non_zero):
        super().__init__()
        self.data_info, self.remain_rto, self.is_non_zero = data_info, remain_rto, is_non_zero
        
        # Fit label encoder
        self.label_encoder_dict = self._fit_label_encoder(data)

        # Iterate
        data_li = []
        data.groupby(self.data_info.modality_info["group"]).progress_apply(lambda x: data_li.append(x))
        self.data_li = tuple(data_li)
    
    def __len__(self):
        return len(self.data_li)
    
    def _fit_label_encoder(self, data):
        result_dict = {}
        target_cols = self.data_info.processing_info["embedding_cols"]
        for col in target_cols:
            encoder = CustomLabelEncoder()
            encoder.fit(data[col])
            result_dict[col] = encoder
        return result_dict

    def _transform_label_encoder(self, data):
        result_dict = {}
        target_cols = self.data_info.processing_info["embedding_cols"]
        for col in target_cols:
            result_dict[col] = self.label_encoder_dict[col].transform(data[col].values)
        return result_dict

    def _scale_data(self, data):
        result_dict = {}
        for col, scaler in self.data_info.processing_info["scaling_cols"].items():
            scaler = scaler()
            result_dict[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
        return result_dict

    def _apply_remain(self, data, valid_idx, remain_rto):
        num_remain = int(len(valid_idx) * remain_rto)
        noise = np.random.rand(len(valid_idx))
        shuffle_idx = np.argsort(noise, axis=0)

        remain_idx = shuffle_idx[:num_remain]
        masked_idx = shuffle_idx[num_remain:]
        revert_idx = np.argsort(shuffle_idx, axis=0)

        remain_padding_mask = np.ones(remain_idx.shape)
        masked_padding_mask = np.ones(masked_idx.shape)
        revert_padding_mask = np.ones(revert_idx.shape)

        return remain_idx, masked_idx, revert_idx, remain_padding_mask, masked_padding_mask, revert_padding_mask

    def _apply_target_remain(self, data):
        result_dict = {}
        target_cols = self.data_info.modality_info["target"]

        for col in target_cols:
            value = data[col].values

            valid_idx = np.arange(len(value)) if not self.is_non_zero else np.where(value > 0)[0]
            remain_idx, masked_idx, revert_idx, remain_padding_mask, masked_padding_mask, revert_padding_mask = self._apply_remain(value, valid_idx, self.remain_rto["target"])

            result_dict[f"target_valid_idx"] = valid_idx
            result_dict[f"{col}_remain_idx"] = remain_idx
            result_dict[f"{col}_masked_idx"] = masked_idx
            result_dict[f"{col}_revert_idx"] = revert_idx
            result_dict[f"{col}_remain_padding_mask"] = remain_padding_mask
            result_dict[f"{col}_masked_padding_mask"] = masked_padding_mask
            result_dict[f"{col}_revert_padding_mask"] = revert_padding_mask

        return result_dict

    def _apply_temporal_remain(self, data, valid_idx):
        result_dict = {}
        target_cols = self.data_info.modality_info["temporal"]

        for col in target_cols:
            value = data[col].values
            remain_idx, masked_idx, revert_idx, remain_padding_mask, masked_padding_mask, revert_padding_mask = self._apply_remain(data[col].values, valid_idx, self.remain_rto["temporal"])

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
        result_dict.update(self._transform_label_encoder(data))

        # Scale
        result_dict.update(self._scale_data(data))

        # Masking
        result_dict.update(self._apply_target_remain(data))
        result_dict.update(self._apply_temporal_remain(data, result_dict["target_valid_idx"]))

        return result_dict

def collate_fn(batch_li, data_info):
    result_dict = {}
    for col in data_info.modality_info["target"]:
        tensor_type = torch.int if col in data_info.processing_info["embedding_cols"] else torch.float
        data = [torch.from_numpy(batch[col]).to(tensor_type) for batch in batch_li]
        data_valid_idx = [torch.from_numpy(batch[f"target_valid_idx"]).to(torch.int64) for batch in batch_li]
        data_remain_idx = [torch.from_numpy(batch[f"{col}_remain_idx"]).to(torch.int64) for batch in batch_li]
        data_masked_idx = [torch.from_numpy(batch[f"{col}_masked_idx"]).to(torch.int64) for batch in batch_li]
        data_revert_idx = [torch.from_numpy(batch[f"{col}_revert_idx"]).to(torch.int64) for batch in batch_li]
        data_remain_padding_mask = [torch.from_numpy(batch[f"{col}_remain_padding_mask"]).to(tensor_type) for batch in batch_li]
        data_masked_padding_mask = [torch.from_numpy(batch[f"{col}_masked_padding_mask"]).to(tensor_type) for batch in batch_li]
        data_revert_padding_mask = [torch.from_numpy(batch[f"{col}_revert_padding_mask"]).to(tensor_type) for batch in batch_li]

        result_dict[col] = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        result_dict[f"target_valid_idx"] = torch.nn.utils.rnn.pad_sequence(data_valid_idx, batch_first=True)
        result_dict[f"{col}_remain_idx"] = torch.nn.utils.rnn.pad_sequence(data_remain_idx, batch_first=True)
        result_dict[f"{col}_masked_idx"] = torch.nn.utils.rnn.pad_sequence(data_masked_idx, batch_first=True)
        result_dict[f"{col}_revert_idx"] = torch.nn.utils.rnn.pad_sequence(data_revert_idx, batch_first=True)
        result_dict[f"{col}_remain_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_remain_padding_mask, batch_first=True)
        result_dict[f"{col}_masked_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_masked_padding_mask, batch_first=True)
        result_dict[f"{col}_revert_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_revert_padding_mask, batch_first=True)

    for col in  data_info.modality_info["temporal"]:
        tensor_type = torch.int if col in data_info.processing_info["embedding_cols"] else torch.float
        data = [torch.from_numpy(batch[col]).to(tensor_type) for batch in batch_li]
        data_remain_idx = [torch.from_numpy(batch[f"{col}_remain_idx"]).to(torch.int64) for batch in batch_li]
        data_masked_idx = [torch.from_numpy(batch[f"{col}_masked_idx"]).to(torch.int64) for batch in batch_li]
        data_revert_idx = [torch.from_numpy(batch[f"{col}_revert_idx"]).to(torch.int64) for batch in batch_li]
        data_remain_padding_mask = [torch.from_numpy(batch[f"{col}_remain_padding_mask"]).to(tensor_type) for batch in batch_li]
        data_masked_padding_mask = [torch.from_numpy(batch[f"{col}_masked_padding_mask"]).to(tensor_type) for batch in batch_li]
        data_revert_padding_mask = [torch.from_numpy(batch[f"{col}_revert_padding_mask"]).to(tensor_type) for batch in batch_li]

        result_dict[col] = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        result_dict[f"{col}_remain_idx"] = torch.nn.utils.rnn.pad_sequence(data_remain_idx, batch_first=True)
        result_dict[f"{col}_masked_idx"] = torch.nn.utils.rnn.pad_sequence(data_masked_idx, batch_first=True)
        result_dict[f"{col}_revert_idx"] = torch.nn.utils.rnn.pad_sequence(data_revert_idx, batch_first=True)

        result_dict[f"{col}_remain_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_remain_padding_mask, batch_first=True)
        result_dict[f"{col}_masked_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_masked_padding_mask, batch_first=True)
        result_dict[f"{col}_revert_padding_mask"] = torch.nn.utils.rnn.pad_sequence(data_revert_padding_mask, batch_first=True)

    return result_dict

class DataInfo():
    def __init__(self, modality_info, processing_info):
        self.modality_info, self.processing_info = modality_info, processing_info
        self._check_modality()
    
    def _check_modality(self):
        # Modality info
        num_target = len(self.modality_info["target"])
        num_temporal = len(self.modality_info["temporal"])
        num_modality = num_target + num_temporal

        # Processing info
        num_scaling_cols = len(self.processing_info["scaling_cols"])
        num_embedding_cols = len(self.processing_info["embedding_cols"])
        num_processing = num_scaling_cols + num_embedding_cols

        assert num_modality == num_processing

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
    
    def transform(self, x, y=None):
        result = []
        for i in x:
            if i in self.mapper.keys():
                result.append(self.mapper[i])
            else:
                result.append(self.mapper["unknown"])
        
        return np.array(result)
    
    def inverse_transform(self, x, y=None):
        result = []
        for i in x:
            result.append(self.inverse_mapper[i])
        return np.array(result)
    
    def get_num_cls(self):
        return len(self.mapper) + 1

class LogScaler(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.a = 1
        return self
    
    def transform(self, x, y=None):
        return np.log1p(x)
    
    def inverse_transform(self, x, y=None):
        return np.expm1(x)

1==1