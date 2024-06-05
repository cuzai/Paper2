from sklearn.preprocessing import MinMaxScaler, StandardScaler

class pre_train():
    def get_information():
        modality_info = {
        "group": ["article_id", "sales_channel_id"],
        "target": ["sales"],
        "temporal": ["day", "dow", "month", "holiday", "price"],
        "img": ["img_path"],
        # "nlp": ["detail_desc", "information"]
        "nlp": ["detail_desc"]
        }
        
        processing_info = {
            "scaling_cols": {"sales": StandardScaler, "price": StandardScaler},
            "embedding_cols": ["day", "dow", "month", "holiday"],
            "img_cols": modality_info["img"],
            "nlp_cols": modality_info["nlp"]
        }
        return modality_info, processing_info

    def get_model_hyperparameter():
        batch_size = 16
        dropout = 0.1
        patch_size = 16
        activation = "gelu"

        nhead = {"encoder":6, "decoder":6}
        d_model = {"encoder":64*6, "decoder":32*6}
        d_ff = {"encoder":64*6, "decoder":32*6}
        num_layers = {"encoder":4, "decoder":4, "output":0}
        remain_rto = {"temporal":0.25, "img":0.25, "nlp":0.25}

        # nhead = {"encoder":4, "decoder":4}
        # d_model = {"encoder":256, "decoder":128}
        # d_ff = {"encoder":256, "decoder":128}
        # num_layers = {"encoder":2, "decoder":2, "output":0}
        # remain_rto = {"temporal":0.25, "img":0.25, "nlp":0.25}

        return batch_size, dropout, patch_size, d_model, d_ff, nhead, num_layers, remain_rto, activation
        
    def arrange_cols(modality_info, processing_info):
        target_col = modality_info["target"]
        temporal_cols = target_col + modality_info["temporal"]
        img_cols = modality_info["img"]
        nlp_cols = modality_info["nlp"]

        scaling_cols = list(processing_info["scaling_cols"].keys())
        embedding_cols = processing_info["embedding_cols"]
        return target_col, temporal_cols, img_cols, nlp_cols, scaling_cols, embedding_cols
    
    def validate_data(temporal_cols, img_cols, nlp_cols, scaling_cols, embedding_cols, processing_info):
        num_modality = len(temporal_cols + img_cols + nlp_cols)
        num_processing = len(scaling_cols + embedding_cols + processing_info["img_cols"] + processing_info["nlp_cols"])

        assert num_modality == num_processing, f"num_modality: {num_modality} and num_processing {num_processing} should be the same"
    
    # Data loader
    MIN_MEANINGFUL_SEQ_LEN = 100
    MAX_SEQ_LEN = 300
    PRED_LEN = 0
    
    # Get information
    modality_info, processing_info = get_information()
    # Get model hyperparameter
    batch_size, dropout, patch_size, d_model, d_ff, nhead, num_layers, remain_rto, activation = get_model_hyperparameter()
    # Arrange cols
    target_col, temporal_cols, img_cols, nlp_cols, scaling_cols, embedding_cols = arrange_cols(modality_info, processing_info)
    # Data validation    
    validate_data(temporal_cols, img_cols, nlp_cols, scaling_cols, embedding_cols, processing_info)

class fine_tuning(pre_train):
    # Data loader
    PRED_LEN = 50
    LEAST_HIST_LEN = PRED_LEN

    remain_rto = {"temporal":1, "img":1, "nlp":1}