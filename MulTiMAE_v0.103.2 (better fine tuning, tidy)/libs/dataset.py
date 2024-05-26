from tqdm import tqdm; tqdm.pandas()

# Sklearn-related
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Torch-related
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image

transform_img = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

transform_img = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.7623358, 0.74334157, 0.738284], std=[0.27400097, 0.2852157, 0.28155997])])
augmentation = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.7623358, 0.74334157, 0.738284], std=[0.27400097, 0.2852157, 0.28155997])])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, data_info, remain_rto):
        super().__init__()
        self.data_info, self.remain_rto = data_info, remain_rto
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

        # Fit label encoder
        self.label_encoder_dict = self.fit_label_encoder(data)

        # Iterate data
        data_li = []
        data.groupby(self.data_info.modality_info["group"]).progress_apply(lambda x: data_li.append(x))
        self.dataset = tuple(data_li)
    
    def fit_label_encoder(self, data):
        result_dict = {}
        embedding_cols = self.data_info.processing_info["embedding_cols"]
        for col in embedding_cols:
            label_encoder = CustomLabelEncoder()
            label_encoder.fit(data[col])
            result_dict[col] = label_encoder
        return result_dict

    def transform_label_encoder(self, data):
        result_dict = {}
        embedding_cols = self.data_info.processing_info["embedding_cols"]
        for col in embedding_cols:
            result_dict[col] = self.label_encoder_dict[col].transform(data[col].values)
        return result_dict
    
    def scale_data(self, data):
        result_dict = {}
        scaling_cols = self.data_info.processing_info["scaling_cols"]
        for col, scaler in scaling_cols.items():
            scaler = scaler()
            result_dict[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
            result_dict[f"{col}_scaler"] = scaler
        return result_dict

    def apply_nlp_remain(self, data):
        result_dict = {}
        nlp_cols = self.data_info.modality_info["nlp"]
        remain_rto = self.remain_rto["nlp"]

        for col in nlp_cols:
            nlp_data = set(data[col].values); assert len(nlp_data) == 1
            total_token = self.tokenizer(next(iter(nlp_data)), return_tensors="np")["input_ids"].squeeze()
            
            # Split global token and valid token
            global_token = total_token[:1]
            valid_token = total_token[1:]
            
            # Get remain/masked/revert indices
            valid_token_shape = valid_token.shape; assert len(valid_token_shape)==1
            
            num_remain = int(valid_token_shape[0] * remain_rto)
            noise = np.random.rand(valid_token_shape[0])
            shuffle_idx = np.argsort(noise)
            
            remain_idx = shuffle_idx[:num_remain]
            masked_idx = shuffle_idx[num_remain:]
            revert_idx = np.argsort(shuffle_idx)
            
            remain_padding_mask = np.ones(remain_idx.shape[0]+1)
            masked_padding_mask = np.ones(masked_idx.shape[0]+1)
            revert_padding_mask = np.ones(revert_idx.shape[0]+1)

            result_dict.update({f"{col}":total_token, f"{col}_raw":nlp_data,
                                f"{col}_remain_idx":remain_idx, f"{col}_masked_idx":masked_idx, f"{col}_revert_idx":revert_idx,
                                f"{col}_remain_padding_mask":remain_padding_mask, f"{col}_masked_padding_mask":masked_padding_mask, f"{col}_revert_padding_mask":revert_padding_mask})

        return result_dict

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        result_dict = {}
        data = self.dataset[idx]
        
        # Laben encode and scale data
        embedding_cols = self.transform_label_encoder(data)
        scaling_cols = self.scale_data(data)
        result_dict.update(**embedding_cols, **scaling_cols)
        
        # Temporal forecast/padding mask
        target_col = self.data_info.modality_info["target"]
        target_fcst_mask = np.ones(data[target_col].shape).squeeze()
        
        temporal_padding_mask  = np.ones(data[target_col].shape).squeeze()
        result_dict.update({"target_fcst_mask":target_fcst_mask, "temporal_padding_mask":temporal_padding_mask})

        # Img
        img_cols = self.data_info.modality_info["img"]
        for col in img_cols:
            img_path = set(data[col].values); assert len(img_path) == 1
            img_raw = Image.open(next(iter(img_path))).convert("RGB")
            result_dict[f"{col}_raw"] = img_raw
        
        # Nlp
        nlp_result_dict = self.apply_nlp_remain(data)
        result_dict.update(nlp_result_dict)

        return result_dict

def collate_fn(batch_li, data_info):
    result_dict = {}
    # Temporal
    target_temporal_cols = data_info.modality_info["target"] + data_info.modality_info["temporal"]
    for col in target_temporal_cols:
        tensor_type = torch.int if col in data_info.processing_info["embedding_cols"] else torch.float
        data = [torch.from_numpy(batch[col]).to(tensor_type) for batch in batch_li]
        result_dict[col] = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    ### Temporal masks
    target_fcst_mask = [torch.from_numpy(batch["target_fcst_mask"]) for batch in batch_li]
    temporal_padding_mask = [torch.from_numpy(batch["temporal_padding_mask"]) for batch in batch_li]
    
    result_dict["target_fcst_mask"] = torch.nn.utils.rnn.pad_sequence(target_fcst_mask, batch_first=True, padding_value=1)
    result_dict["temporal_padding_mask"] = torch.nn.utils.rnn.pad_sequence(temporal_padding_mask, batch_first=True)

    # Img
    img_cols = data_info.modality_info["img"]
    for col in img_cols:
        img_raw = [batch[f"{col}_raw"] for batch in batch_li]
        img_data = torch.stack([augmentation(i) for i in img_raw], dim=0)

        result_dict[f"{col}_raw"] = torch.stack([transform_img(i) for i in img_raw], dim=0)
        result_dict[f"{col}"] = img_data

    
    # Nlp
    nlp_cols = data_info.modality_info["nlp"]
    for col in nlp_cols:
        data = [torch.from_numpy(batch[col]).to(torch.int) for batch in batch_li]
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

        result_dict[f"{col}_raw"] = [batch[f"{col}_raw"] for batch in batch_li]

    return result_dict

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        nuique_value = set(x)
        nuique_value.append("unknown")
        self.val_to_idx_dict = {val:idx for idx, val in enumerate(nuique_value)}
        self.idx_to_val_dict = {idx:val for val, idx in self.val_to_idx_dict.items()}
        return self
    
    def transform(self, x, y=None):
        result_li = []
        for val in x:
            if val in self.val_to_idx_dict.keys():
                result_li.append(self.val_to_idx_dict[val])
            else:
                result_li.append(self.val_to_idx_dict["unknown"])
        
        return np.array(result_li)
    
    def inverse_transform(self, x, y=None):
        result_li = []
        if idx in x:
            result_li.append(self.idx_to_val_dict[idx])
        
        return np.array(result_li)

    def get_num_cls(self):
        return len(self.val_to_idx_dict)

1==1