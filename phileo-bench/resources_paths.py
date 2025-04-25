BASE_PATH = "/path/to/PhilEO_Bench_TESTS"
PATH_EXPERIMENTS = "/path/to/PhilEO_Bench_TESTS/experiments"
PATH_INDICES = "/path/to/phileo-bench-ddp/indices"

DATASETS_PATH = "/path/to/PhilEO_Bench_TESTS/"

PATH_DATA_STATIC = "/path/to/data_static/"
PATH_MINIFM_LABELS = "/path/to/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/"

# ----------------------------- DATASETS ------------------------------------
PATH_DS_MAJORTOM = "/path/to/MajorTOM/"
PATH_DS_MAJORTOM_PATHS = "/path/to/MajorTOM/MajorTOM_Paths.pkl"
PATH_DS_DOWNSTREAM_NP_128_10M = "/path/to/Phileo_downstream/downstream_dataset_patches_np/"
PATH_DS_DOWNSTREAM_NP_224_10M = "/path/to/Phileo_downstream/downstream_dataset_patches_np_224/"
PATH_DS_DOWNSTREAM_NP_224_30M = "/path/to/Phileo_downstream/downstream_dataset_patches_np_HLS/"
PATH_DS_DOWNSTREAM_NP_256_10M = "/path/to/Phileo_downstream/downstream_dataset_patches_np_256/"
PATH_DS_DOWNSTREAM_NP_256_30M = "/path/to/Phileo_downstream/downstream_dataset_patches_np_256_HLS/"
PATH_DS_DOWNSTREAM_NSHOT = "/path/to/Phileo_downstream/downstream_datasets_nshot/"
PATH_DS_DOWNSTREAM_PSPLIT = "/path/to/Phileo_downstream/downstream_datasets_psplit/"


PATHS_DS_DOWNSTREAM = [
    PATH_DS_DOWNSTREAM_NP_128_10M,
    PATH_DS_DOWNSTREAM_NP_224_10M,
    PATH_DS_DOWNSTREAM_NP_224_30M,
    PATH_DS_DOWNSTREAM_NP_256_10M,
    PATH_DS_DOWNSTREAM_NP_256_30M,
]

DATASET_DS_DOWNSTREAM_NAMES = {
    PATH_DS_DOWNSTREAM_NP_128_10M: "128_10M",
    PATH_DS_DOWNSTREAM_NP_224_10M: "224_10M",
    PATH_DS_DOWNSTREAM_NP_224_30M: "224_30M",
    PATH_DS_DOWNSTREAM_NP_256_10M: "256_10M",
    PATH_DS_DOWNSTREAM_NP_256_30M: "256_30M",

}

# ----------------------------- MODELS ------------------------------------

PATH_MODEL_PRECURSOR = "/path/to/phileo-foundation/model/phileo-precursor_v08_e025.pt"
PATH_MODEL_COREENCODER_BEST = "/path/to/Phileo_pretrained_models/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt"
PATH_MODEL_COREENCODER_LAST = "/path/to/Phileo_pretrained_models/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_8.pt"
PATH_MODEL_COREENCODER_MH_BEST = "/path/to/Phileo_pretrained_models/01112023_CoreEncoderMultiHead_geo_pred_geo_reduce_on_plateau/CoreEncoderMultiHead_geo_pred_best.pt"
PATH_MODEL_VITL_PRETRAIN_BEST = "/path/to/Phileo_pretrained_models/SatMAE_pretrain-vit-large-e199.pth"
PATH_MODEL_MAEVIT = "/path/to/Phileo_pretrained_models/31102023_MaskedAutoencoderViT/MaskedAutoencoderViT_ckpt.pt"
PATH_MODEL_MAEGCVIT = "/path/to/Phileo_pretrained_models/03112023_MaskedAutoencoderGroupChannelViT/MaskedAutoencoderGroupChannelViT_ckpt.pt"
PATH_MODEL_RESNET50 = "/path/to/Phileo_pretrained_models/seco_resnet50_1m.ckpt"











