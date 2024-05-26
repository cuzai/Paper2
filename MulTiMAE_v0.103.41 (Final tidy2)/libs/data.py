import os

import numpy as np
import pandas as pd
import holidays
import joblib
from tqdm import tqdm; tqdm.pandas()

# Sklearn-related
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Torch-related
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image

# Custom defined
from architecture.architecture import get_indices

# Raw data
class RawData():
    def __init__(self):
        self.trans_col = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
        self.meta_col = ["article_id", "prod_name", "product_type_name", "product_group_name", "graphical_appearance_name","colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name", "department_name", "index_name", "index_group_name", "section_name", "garment_group_name", "detail_desc"]

    def _read_raw_data(self):
        # Read transaction
        df_trans = pd.read_csv("../HnM/transactions_train.csv", parse_dates=["t_dat"], dtype={"article_id":str})
        df_trans = df_trans[self.trans_col]
        
        df_meta = pd.read_csv("../HnM/articles.csv", dtype={"article_id":str})
        df_meta = df_meta[self.meta_col]

        return df_trans, df_meta
    
    def _get_holiday(self, min_year, max_year):
        holiday = holidays.US(years=(min_year, max_year))
        df_holiday = pd.DataFrame({"t_dat":holiday.keys(), "holiday":holiday.values()})
        df_holiday["t_dat"] = pd.to_datetime(df_holiday["t_dat"])

        return df_holiday
    
    def __call__(self):
        df_trans, df_meta = self._read_raw_data()
        
        min_year = df_trans["t_dat"].dt.year.min()
        max_year = df_trans["t_dat"].dt.year.max()
        df_holiday = self._get_holiday(min_year, max_year)

        return df_trans, df_meta, df_holiday

class Preprocess():
    def __init__(self, df_trans, df_meta, df_holiday):
        self.df_trans = df_trans
        self.df_meta = df_meta
        self.df_holiday = df_holiday

    def _get_img_path(self, data):
        data["img_path"] = data["article_id"].apply(lambda x: f'../HnM/images/{x[:3]}/{x}.jpg')
        return data
    
    def _filter_img_valid(self, data):
        data["is_img_valid"] = data["img_path"].apply(lambda x: 1 if os.path.isfile(x) else 0) # Check whether the article has corresponding image file
        data = data[data["is_img_valid"] == 1].drop("is_img_valid", axis=1) # Valid if having corresponding image
        return data

    def _make_sales(self, data):
        data = data.groupby(["t_dat", "article_id", "img_path", "sales_channel_id"], as_index=False).agg(sales=("customer_id", "count"), price=("price", "mean"))
        data["meaningful_size"] = data.groupby(["article_id", "sales_channel_id"], as_index=False)["sales"].transform("count")
        return data

    def _expand_date(self, data):
        data = data.set_index("t_dat").groupby(["article_id", "sales_channel_id"], as_index=False).resample("1D").asfreq().reset_index()
        data["sales"] = data["sales"].fillna(0)
        data["price"] = data["price"].fillna(method="ffill")
        data["article_id"] = data["article_id"].fillna(method="ffill")
        data["img_path"] = data["img_path"].fillna(method="ffill")
        data["sales_channel_id"] = data["sales_channel_id"].fillna(method="ffill")
        data["meaningful_size"] = data["meaningful_size"].fillna(method="ffill")
        
        data["full_size"] = data.groupby(["article_id", "sales_channel_id"], as_index=False)["sales"].transform("count")
        data = data.sort_values(["article_id", "t_dat", "sales_channel_id"]).reset_index(drop=True)
        return data

    def _generate_date_info(self, data):
        data["day"] = data["t_dat"].dt.day
        data["dow"] = data["t_dat"].dt.dayofweek
        data["month"] = data["t_dat"].dt.month
        # data["year"] = data["t_dat"].dt.year.max() - data["t_dat"].dt.year

        # # Cyclic transformation
        # def append_cyclic(col, cycle):
        #     data[f"{col}_sin"] = np.sin(2 * np.pi * data[col]/cycle)
        #     data[f"{col}_cos"] = np.cos(2 * np.pi * data[col]/cycle)
        # append_cyclic("day", 365)
        # append_cyclic("dow", 7)
        # append_cyclic("month", 12)
        return data
    
    def _generate_time_idx(self, data):
        data["time_idx"] = data.groupby(["article_id", "sales_channel_id"]).cumcount()
        return data

    def _merge_meta(self, data, df_meta):
        data = data.merge(df_meta, on="article_id")
        data["information"] = data[["prod_name", "product_type_name", "product_group_name", "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name", "department_name", "index_name", "index_group_name", "section_name", "garment_group_name"]].values.tolist()
        data["information"] = data["information"].progress_apply(set).str.join(" ")
        return data

    def _merge_holiday(self, data, df_holiday):
        data = pd.merge(data, df_holiday, on="t_dat", how="left")
        data["holiday"] = data["holiday"].fillna("Normal day")
        return data

    def __call__(self):
        df_with_img_path = self._get_img_path(self.df_trans); print("_get_img_path")
        df_img_valid = self._filter_img_valid(df_with_img_path); print("_filter_img_valid")
        df_with_sales = self._make_sales(df_img_valid); print("_make_sales")
        df_with_date_expanded = self._expand_date(df_with_sales); print("_expand_date")
        df_with_date_info = self._generate_date_info(df_with_date_expanded); print("_generate_date_info")
        df_with_time_idx = self._generate_time_idx(df_with_date_info); print("_generate_time_idx")
        df_with_meta = self._merge_meta(df_with_time_idx, self.df_meta); print("_merge_meta")
        df_with_holiday = self._merge_holiday(df_with_meta, self.df_holiday); print("_merge_holiday")

        data = df_with_holiday.sort_values(["sales_channel_id", "article_id", "t_dat"]).reset_index(drop=True)
        data.to_parquet("src/df_prep.parquet")
        data.iloc[:1000].to_parquet("src/df_prep_test.parquet")


# Dataloader
def load_dataset(is_test_mode, is_new_rawdata, config, mode, verbose):
    # Get df_prep
    if is_new_rawdata:
        df_trans, df_meta, df_holiday = RawData()()
        Preprocess(df_trans, df_meta, df_holiday)()
    
    df_prep_test = pd.read_parquet("./src/df_prep_test.parquet")
    df_prep = pd.read_parquet("./src/df_prep.parquet")
    
    # Get df_train
    df_train_test = df_prep_test[(df_prep_test["meaningful_size"] >= config.MIN_MEANINGFUL_SEQ_LEN)
                        &(df_prep_test["time_idx"] <= config.MAX_SEQ_LEN-1 + config.PRED_LEN)
                        &(~pd.isna(df_prep_test["detail_desc"]))]

    df_train = df_prep[(df_prep["meaningful_size"] >= config.MIN_MEANINGFUL_SEQ_LEN)
                        &(df_prep["time_idx"] <= config.MAX_SEQ_LEN-1 + config.PRED_LEN)
                        &(~pd.isna(df_prep["detail_desc"]))]
    
    # Get dataloader
    train_dataset_test = Dataset(df_train_test, config, mode)
    train_dataset = Dataset(df_train, config, mode)

    torch.save(train_dataset_test, f"src/{mode}_dataset_test")
    torch.save(train_dataset, f"src/{mode}_dataset")
    
    if is_test_mode:
        return train_dataset_test
    else:
        return train_dataset

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, config, mode):
        super().__init__()
        self.config, self.mode = config, mode
        # Image augmentation
        self.img_transform = transforms.Compose([transforms.Resize((224,224)),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=[0.7623358, 0.74334157, 0.738284], std=[0.27400097, 0.2852157, 0.28155997])
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])
        
        self.img_augmentation = transforms.Compose([
                                                    # transforms.RandomResizedCrop(224),
                                                    transforms.Resize((224,224)),
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize(mean=[0.7623358, 0.74334157, 0.738284], std=[0.27400097, 0.2852157, 0.28155997]),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                    # transforms.ColorJitter(contrast=(3, 3))
                                                    ])
                                                    
        # Fit label encoder
        if mode == "pre_train":
            self.label_encoder_dict = self.fit_label_encoder(data)
            joblib.dump(self.label_encoder_dict, "./src/label_encoder_dict.pkl")
        elif mode == "fine_tuning":
            self.label_encoder_dict = joblib.load("src/label_encoder_dict.pkl")

        # Iterate data
        data_li = []
        data.groupby(self.config.modality_info["group"]).progress_apply(lambda x: data_li.append(x))
        self.dataset = tuple(data_li)
    
    def fit_label_encoder(self, data):
        result_dict = {}
        for col in self.config.embedding_cols:
            label_encoder = CustomLabelEncoder()
            label_encoder.fit(data[col])
            result_dict[col] = label_encoder
        return result_dict

    def __len__(self):
        return len(self.dataset)
    
    def transform_embeddings(self, data):
        result_dict = {}
        for col in self.config.embedding_cols:
            result_dict[f"{col}_encoder_input"] = self.label_encoder_dict[col].transform(data[col].values)[:-self.config.PRED_LEN]
            result_dict[f"{col}_decoder_input"] = self.label_encoder_dict[col].transform(data[col].values)[-self.config.PRED_LEN:]
        return result_dict

    def scale_scalables(self, data):
        result_dict = {}
        for col in self.config.scaling_cols:
            scaler = self.config.processing_info["scaling_cols"][col]()
            scaler.fit(data[col][:-self.config.PRED_LEN].values.reshape(-1,1))
            
            result_dict[f"{col}_encoder_input"] = scaler.transform(data[col][:-self.config.PRED_LEN].values.reshape(-1,1))
            result_dict[f"{col}_decoder_input"] = scaler.transform(data[col][-self.config.PRED_LEN:].values.reshape(-1,1))
            result_dict[f"{col}_scaler"] = scaler
        return result_dict

    def __getitem__(self, idx):
        result_dict = {}
        data = self.dataset[idx]
        
        # Temporal data
        ### Embedding data
        embedded_data = self.transform_embeddings(data)
        result_dict.update(embedded_data)
        
        ### Scaling data
        scaled_data = self.scale_scalables(data)
        result_dict.update(scaled_data)

        ### Temporal mask
        encoder_input_padding_mask = np.ones(data[self.config.target_col].shape, dtype=np.float32)[:-self.config.PRED_LEN].squeeze()
        decoder_input_padding_mask = np.ones(data[self.config.target_col].shape, dtype=np.float32)[-self.config.PRED_LEN:].squeeze()
        result_dict.update({"encoder_input_padding_mask":encoder_input_padding_mask, "decoder_input_padding_mask":decoder_input_padding_mask})
        
        # Img data
        for col in self.config.img_cols:
            img_path = set(data[col].values); assert len(img_path) == 1
            img_raw = Image.open(next(iter(img_path))).convert("RGB")
            result_dict[f"{col}_raw"] = self.img_transform(img_raw)
            if self.mode == "pre_train":
                result_dict[f"{col}"] = self.img_augmentation(img_raw)
            elif self.mode == "fine_tuning":
                result_dict[f"{col}"] = self.img_transform(img_raw)

        # Nlp data
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        for col in self.config.nlp_cols:
            nlp_raw = list(set(data[col].values)); assert len(nlp_raw) == 1
            nlp_raw = list(nlp_raw)[0]
            # nlp_data = np.array(tokenizer.encode(nlp_raw)[1:-1])[np.newaxis, ...]
            nlp_token = tokenizer(nlp_raw, return_tensors="np", padding=True)
            nlp_data = nlp_token["input_ids"][:, 1:-1]
            nlp_padding_mask = nlp_token["attention_mask"][:, :-1].astype(np.float32)
            num_remain = int(np.ceil(nlp_data.shape[-1] * self.config.remain_rto["nlp"]))
            assert num_remain > 0, f"{nlp_data.shape}, {self.config.remain_rto['nlp']}"
            nlp_remain_idx, nlp_maksed_idx, nlp_revert_idx = get_indices(nlp_data.shape, num_remain)
            
            result_dict.update({f"{col}_raw":nlp_raw, f"{col}":nlp_data.squeeze(), f"{col}_remain_idx":nlp_remain_idx.numpy()[0], f"{col}_masked_idx":nlp_maksed_idx.numpy()[0], f"{col}_revert_idx":nlp_revert_idx.numpy()[0], f"{col}_revert_padding_mask":nlp_padding_mask[0]})
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        return result_dict
    
def collate_fn(batch_li, config):
    result_dict = {}

    # Temporal
    for col in config.temporal_cols:
        tensor_type = torch.int if col in config.embedding_cols else torch.float
        encoder_input = [torch.from_numpy(batch[f"{col}_encoder_input"]).to(tensor_type) for batch in batch_li]
        decoder_input = [torch.from_numpy(batch[f"{col}_decoder_input"]).to(tensor_type) for batch in batch_li]
        
        result_dict[f"{col}_encoder_input"] = torch.nn.utils.rnn.pad_sequence(encoder_input, batch_first=True)
        result_dict[f"{col}_decoder_input"] = torch.nn.utils.rnn.pad_sequence(decoder_input, batch_first=True)

    ### Temporal mask
    encoder_input_padding_mask = [torch.from_numpy(batch["encoder_input_padding_mask"]) for batch in batch_li]
    decoder_input_padding_mask = [torch.from_numpy(batch["decoder_input_padding_mask"]) for batch in batch_li]
    
    result_dict["encoder_input_padding_mask"] = torch.nn.utils.rnn.pad_sequence(encoder_input_padding_mask, batch_first=True)
    result_dict["decoder_input_padding_mask"] = torch.nn.utils.rnn.pad_sequence(decoder_input_padding_mask, batch_first=True)

    # Image
    for col in config.img_cols:
        result_dict[f"{col}_raw"] = [batch[f"{col}_raw"] for batch in batch_li]
        result_dict[col] = torch.stack([batch[f"{col}"] for batch in batch_li])
    
    # Nlp
    for col in config.nlp_cols:
        nlp_raw = [batch[f"{col}_raw"] for batch in batch_li]
        nlp_data = [torch.from_numpy(batch[f"{col}"]) for batch in batch_li]
        nlp_padding_mask = [torch.from_numpy(batch[f"{col}_revert_padding_mask"]) for batch in batch_li]
        
        result_dict[f"{col}_raw"] = nlp_raw
        result_dict[col] = torch.nn.utils.rnn.pad_sequence(nlp_data, batch_first=True)
        result_dict[f"{col}_revert_padding_mask"] = torch.nn.utils.rnn.pad_sequence(nlp_padding_mask, batch_first=True)
        
        for idx_type in ["remain", "masked", "revert"]:
            try:
                nlp_idx = [torch.from_numpy(batch[f"{col}_{idx_type}_idx"]).to(torch.int64) for batch in batch_li]
                result_dict[f"{col}_{idx_type}_idx"] = torch.nn.utils.rnn.pad_sequence(nlp_idx, batch_first=True)
            except:
                print(idx_type)
                raise
            

    return result_dict

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        nuique_value = list(set(x))
        nuique_value.append("__unknown__")
        self.val_to_idx_dict = {val:idx for idx, val in enumerate(nuique_value)}
        self.idx_to_val_dict = {idx:val for val, idx in self.val_to_idx_dict.items()}
        return self
    
    def transform(self, x, y=None):
        result_li = []
        for val in x:
            if val in self.val_to_idx_dict.keys():
                result_li.append(self.val_to_idx_dict[val])
            else:
                result_li.append(self.val_to_idx_dict["__unknown__"])
        
        return np.array(result_li)
    
    def inverse_transform(self, x, y=None):
        result_li = []
        if idx in x:
            result_li.append(self.idx_to_val_dict[idx])
        
        return np.array(result_li)

    def get_num_cls(self):
        return len(self.val_to_idx_dict)

1==1