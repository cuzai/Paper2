import os
import numpy as np
import pandas as pd
import holidays

def get_raw_data(is_test_mode, is_prep_data_exist):
    if is_test_mode:
        df_prep = pd.read_parquet("./src/df_preprocessed_test.parquet")
    else:
        if not is_prep_data_exist:
            raw_data = RawData()
            df_trans, df_meta, df_holiday = raw_data()
            preprocess = Preprocess(df_trans, df_meta, df_holiday)
            df_prep = preprocess()
        else:
            df_prep = pd.read_parquet("./src/df_preprocessed.parquet")
    return df_prep
    
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
    
    def __call__(self, is_exist=False):
        if not is_exist:
            df_trans, df_meta = self._read_raw_data()
            
            min_year = df_trans["t_dat"].dt.year.min()
            max_year = df_trans["t_dat"].dt.year.max()
            df_holiday = self._get_holiday(min_year, max_year)

            df_trans.to_parquet("./src/df_trans.parquet")
            df_meta.to_parquet("./src/df_meta.parquet")
            df_holiday.to_parquet("./src/df_holiday.parquet")
        else:
            df_trans = pd.read_parquet("./src/df_trans.parquet")
            df_meta = pd.read_parquet("./src/df_meta.parquet")
            df_holiday = pd.read_parquet("./src/df_holiday.parquet")

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
        data["year"] = data["t_dat"].dt.year.max() - data["t_dat"].dt.year

        # Cyclic transformation
        def append_cyclic(col, cycle):
            data[f"{col}_sin"] = np.sin(2 * np.pi * data[col]/cycle)
            data[f"{col}_cos"] = np.cos(2 * np.pi * data[col]/cycle)
        append_cyclic("day", 365)
        append_cyclic("dow", 7)
        append_cyclic("month", 12)
        return data
    
    def _generate_time_idx(self, data):
        data["time_idx"] = data.groupby(["article_id", "sales_channel_id"]).cumcount()
        return data

    def _merge_meta(self, data, df_meta):
        data = data.merge(df_meta, on="article_id")
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
        data.to_parquet("src/df_preprocessed.parquet")
        data.iloc[:1000].to_parquet("src/df_preprocessed_test.parquet")
        return data

# Data processing
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
    
1==1