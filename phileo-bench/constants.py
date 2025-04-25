CNN_LIST = [
    'baseline_cnn',
    'core_unet_nano',
    'core_unet_tiny',
    'core_unet_base',
    'core_unet_large',
    'core_unet_huge',
    'core_vae_nano',
    'resnet_imagenet',
    'resnet',
    'core_encoder_nano',
]

VIT_CNN_LIST = [
    'vit_cnn_base',
    'vit_cnn_base_wSkip',
]

VIT_UPERNET_PRETRAINED_LIST = ['vit_upernet_pretrained']

PHIVIT_UPERNET_PRETRAINED_LIST = ['phivit_upernet_pretrained']

MIXER_LIST = [
    'mixer_nano',
    'mixer_tiny',
    'mixer_base',
    'mixer_large',
    'mixer_huge',
]

VIT_LIST = [
    'linear_vit_base',
    'linear_vit_larger',
    'linear_vit_huge',
    'autoencoder_vit_base',
    'autoencoder_vit_large',
    'autoencoder_vit_huge',
]

CNN_PRETRAINED_LIST = [
    'GeoAware_core_nano',
    'GeoAware_core_tiny',
    'GeoAware_mixer_nano',
    'GeoAware_mixer_tiny',
    'GeoAware_contrastive_core_nano',
    'GeoAware_mh_pred_core_nano',
    'GeoAware_combined_core_nano',
    'GeoAware_core_autoencoder_nano',
    'seasonal_contrast',
]

VIT_CNN_PRETRAINED_LIST = [
    'prithvi',
    'vit_cnn',
    'vit_cnn_gc',
    'SatMAE',
    'vit_cnn_wSkip',
    'vit_cnn_gc_wSkip',
]

MODEL_LIST_PRETRAINED = CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST + VIT_UPERNET_PRETRAINED_LIST \
                        + PHIVIT_UPERNET_PRETRAINED_LIST
MODEL_LIST = CNN_LIST + MIXER_LIST + VIT_LIST + VIT_CNN_LIST + MODEL_LIST_PRETRAINED

DOWNSTREAM_LIST = [
    'lc',
    'building',
    'roads',
]

REGIONS = [
    'denmark-1',
    'denmark-2',
    'east-africa',
    'egypt-1',
    'eq-guinea',
    'europe',
    'ghana-1',
    'israel-1',
    'israel-2',
    'japan',
    'nigeria',
    'north-america',
    'senegal',
    'south-america',
    'tanzania-1',
    'tanzania-2',
    'tanzania-3',
    'tanzania-4',
    'tanzania-5',
    'uganda-1',
]

LR_SCHEDULERS = [
    'reduce_on_plateau',
    'cosine_annealing',
]
