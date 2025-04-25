import torch
from constants import *
from torchinfo import summary

from models.model_GeoAwarePretrained import get_mixer_kwargs, get_core_encoder_kwargs
from models.model_AutoEncoderViTPretrained import get_core_decoder_kwargs

import sys
sys.path.append("../../")


def get_trainer(
        model_name: str,
        downstream_task: str,
        epochs: int,
        lr:float,
        model,
        device,
        lr_scheduler,
        es_patience,
        es_delta,
        dl_train,
        dl_val,
        dl_test,
        NAME,
        output_folder,
        vis_val,
        warmup_epochs,
        warmup_gamma,
        RANK,
        wandb,
):
    train_args = {
        "epochs": epochs,
        "lr": lr,
        "model": model,
        "device": device,
        "lr_scheduler": lr_scheduler,
        "es_patience": es_patience,
        "es_delta": es_delta,
        "train_loader": dl_train,
        "val_loader": dl_val,
        "test_loader": dl_test,
        "name": NAME,
        "out_folder": output_folder,
        "visualise_validation": vis_val,
        "warmup_epochs": warmup_epochs,
        "warmup_gamma": warmup_gamma,
        "RANK": RANK,
        "wandb": wandb,
    }

    trainer = None

    if model_name in (CNN_LIST + MIXER_LIST + VIT_CNN_LIST + CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST
                      + VIT_UPERNET_PRETRAINED_LIST + GEODINO_UPERNET_PRETRAINED_LIST + PHIVIT_UPERNET_PRETRAINED_LIST):

        if downstream_task == 'roads' or downstream_task == 'building':
            from trainers.train_base import TrainBase
            trainer = TrainBase(**train_args)
        elif downstream_task == 'lc':
            from trainers.train_land_cover import TrainLandCover
            trainer = TrainLandCover(**train_args)

    elif model_name in VIT_LIST:
        if downstream_task == 'roads' or downstream_task == 'building':
            from trainers.train_vit import TrainViT
            trainer = TrainViT(**train_args)

        elif downstream_task == 'lc':
            from trainers.train_vit_land_cover import TrainViTLandCover
            trainer = TrainViTLandCover(**train_args)

    if model_name == 'core_vae_nano':
        from trainers.train_vae import TrainVAE
        trainer = TrainVAE(**train_args)

    if trainer is None:
        raise ValueError(
            "Something went wrong and the trainer was not initialized correctly. \
            Check the specified downstream task and/or model name."
        )

    return trainer


def get_models(model_name, input_channels, output_channels, input_size):

    # chw = (input_channels, input_size, input_size)

    # Baseline CNN
    if model_name == 'baseline_cnn':
        from models.model_Baseline import BaselineNet
        model = BaselineNet(input_dim=input_channels, output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # Core U-Nets
    elif model_name == 'core_unet_nano':
        from models.model_CoreCNN_versions import CoreUnet_nano
        model = CoreUnet_nano(input_dim=input_channels, output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'core_encoder_nano':
        from models.model_CoreCNN_versions import Core_nano
        model = Core_nano(input_dim=input_channels, output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'core_unet_tiny':
        from models.model_CoreCNN_versions import CoreUnet_tiny
        model = CoreUnet_tiny(input_dim=input_channels, output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'core_unet_base':
        from models.model_CoreCNN_versions import CoreUnet_base
        model = CoreUnet_base(input_dim=input_channels, output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'core_unet_large':
        from models.model_CoreCNN_versions import CoreUnet_large
        model = CoreUnet_large(input_dim=input_channels, output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'core_unet_huge':
        from models.model_CoreCNN_versions import CoreUnet_huge
        model = CoreUnet_huge(input_dim=input_channels, output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # Mixers
    elif model_name == 'mixer_nano':
        from models.model_Mixer_versions import Mixer_nano
        model = Mixer_nano(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'mixer_tiny':
        from models.model_Mixer_versions import Mixer_tiny
        model = Mixer_tiny(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'mixer_base':
        from models.model_Mixer_versions import Mixer_base
        model = Mixer_base(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'mixer_large':
        from models.model_Mixer_versions import Mixer_large
        model = Mixer_large(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'mixer_huge':
        from models.model_Mixer_versions import Mixer_huge
        model = Mixer_huge(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # ViTs
    elif model_name == 'linear_vit_base':
        from models.model_LinearViT_versions import LinearViT_base
        model = LinearViT_base(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'linear_vit_large':
        from models.model_LinearViT_versions import LinearViT_large
        model = LinearViT_large(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'linear_vit_huge':
        from models.model_LinearViT_versions import LinearViT_huge
        model = LinearViT_huge(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # AE-VIts
    elif model_name == 'autoencoder_vit_base':
        from models.model_AutoEncoderViT_versions import AutoencoderViT_base
        model = AutoencoderViT_base(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'autoencoder_vit_large':
        from models.model_AutoEncoderViT_versions import AutoencoderViT_large
        model = AutoencoderViT_large(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'autoencoder_vit_huge':
        from models.model_AutoEncoderViT_versions import AutoencoderViT_huge
        model = AutoencoderViT_huge(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # Core VAEs
    elif model_name == 'core_vae_nano':
        from models.model_CoreVAE import CoreVAE_nano
        model = CoreVAE_nano(input_dim=input_channels, output_dim=10)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # ViT-CNNs
    elif model_name == 'vit_cnn_base':
        from models.model_AutoEncoderViTPretrained import vit_large
        model = vit_large(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'vit_cnn_base_wSkip':
        from models.model_AutoEncoderViTPretrained_wSkip import vit_large_wSkip
        model = vit_large_wSkip(chw=(input_channels, input_size, input_size), output_dim=output_channels)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # ResNets
    elif model_name == 'resnet_imagenet':
        from models.model_Resnet50 import resnet
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = resnet(imagenet_weights=True, **resnet_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))
    elif model_name == 'resnet':
        from models.model_Resnet50 import resnet
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = resnet(imagenet_weights=False, **resnet_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    else:
        raise ValueError(f"Unsupported model {model_name}")

    # model_summary = summary(model, test_input)
    model_summary = None    # TODO implement test_input

    return model, model_summary


def get_models_pretrained(
        model_name,
        input_channels,
        output_channels,
        input_size,
        path_model_weights=None,
        freeze=False,
        device='cuda'
):

    # ----------------- GeoAware-Core ---------------------

    if model_name == 'GeoAware_core_nano' or model_name == 'GeoAware_contrastive_core_nano' or model_name == 'GeoAware_mh_pred_core_nano':
        from models.model_GeoAwarePretrained import CoreEncoderGeoPretrained
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    elif model_name == 'GeoAware_core_autoencoder_nano':
        from models.model_GeoAwarePretrained import CoreEncoderGeoAutoEncoder
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoAutoEncoder(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    elif model_name == 'GeoAware_combined_core_nano':
        from models.model_GeoAwarePretrained import CoreEncoderGeoPretrained_combined
        sd_1 = torch.load(path_model_weights[0])
        sd_2 = torch.load(path_model_weights[1])
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano')
        model = CoreEncoderGeoPretrained_combined(output_channels, checkpoint_1=sd_1, checkpoint_2=sd_2, core_encoder_kwargs=core_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    elif model_name == 'GeoAware_core_tiny':
        from models.model_GeoAwarePretrained import CoreEncoderGeoPretrained
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_tiny', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # ----------------- GeoAware-Mixer ---------------------

    elif model_name == 'GeoAware_mixer_nano':
        from models.model_GeoAwarePretrained import MixerGeoPretrained
        sd = torch.load(path_model_weights)
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels, input_size, input_size), output_dim=output_channels, mixer_size='mixer_nano')
        model = MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    elif model_name == 'GeoAware_mixer_tiny':
        from models.model_GeoAwarePretrained import MixerGeoPretrained
        sd = torch.load(path_model_weights)
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels, input_size, input_size), output_dim=output_channels, mixer_size='mixer_tiny')
        model = MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # ----------------- SatMAE ------------------

    elif model_name == 'SatMAE':
        from models.model_SatMAE import satmae_vit_cnn
        sd = torch.load(path_model_weights)
        satmae_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = satmae_vit_cnn(img_size=96, patch_size=8, in_chans=input_channels, checkpoint=sd, freeze_body=freeze, classifier=False, **satmae_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # ----------------- Prithvi ---------------------

    elif model_name == 'prithvi':
        from models.models_Prithvi import prithvi
        sd = torch.load(path_model_weights, map_location=device)
        prithvi_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = prithvi(checkpoint=sd, freeze_body=freeze, classifier=False, **prithvi_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    # ----------------- ViT-CNN ---------------------

    elif model_name == 'vit_cnn':
        from models.model_AutoEncoderViTPretrained import vit_cnn
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = vit_cnn(checkpoint=sd, freeze_body=freeze, **vit_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    elif model_name == 'vit_cnn_wSkip':
        from models.model_AutoEncoderViTPretrained_wSkip import vit_cnn_wSkip
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = vit_cnn_wSkip(checkpoint=sd, freeze_body=freeze, **vit_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    elif model_name == 'vit_cnn_gc':
        from models.model_AutoEncoderViTPretrained import vit_cnn_gc
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = vit_cnn_gc(checkpoint=sd, freeze_body=freeze, **vit_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    elif model_name == 'vit_cnn_gc_wSkip':
        from models.model_AutoEncoderViTPretrained_wSkip import vit_cnn_gc_wSkip
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = vit_cnn_gc_wSkip(checkpoint=sd, freeze_body=freeze, **vit_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))
        
        # ----------------- ViT-UPerNet ---------------------
        
    elif model_name == 'vit_upernet_pretrained':
        from models.model_ViTUperNet import vit_upernet_pretrained
        sd = torch.load(path_model_weights, map_location = device)
        model = vit_upernet_pretrained(checkpoint= sd, chw =(input_channels, input_size, input_size), 
                                      output_dim= output_channels, freeze_body= freeze)

    elif model_name == 'phivit_upernet_pretrained':
        from models.model_PhiViTUperNet import phivit_upernet_pretrained
        sd = torch.load(path_model_weights, map_location = device, weights_only = False)
        model = phivit_upernet_pretrained(checkpoint=sd, chw=(input_channels, input_size, input_size),
                                          output_dim=output_channels, freeze_body=freeze)
    
    elif model_name == 'geodino_tiny_upernet_pretrained':
        from models.model_GeoDINOUperNet import geodino_upernet_pretrained
        sd = torch.load(path_model_weights, map_location = device, weights_only = False)
        model = geodino_upernet_pretrained('tiny', checkpoint=sd, chw=(input_channels, input_size, input_size),
                                           output_dim=output_channels, freeze_body=freeze)
    elif model_name == 'geodino_small_upernet_pretrained':
        from models.model_GeoDINOUperNet import geodino_upernet_pretrained
        sd = torch.load(path_model_weights, map_location = device, weights_only = False)
        model = geodino_upernet_pretrained('small', checkpoint=sd, chw=(input_channels, input_size, input_size),
                                           output_dim=output_channels, freeze_body=freeze)
    elif model_name == 'geodino_base_upernet_pretrained':
        from models.model_GeoDINOUperNet import geodino_upernet_pretrained
        sd = torch.load(path_model_weights, map_location = device, weights_only = False)
        model = geodino_upernet_pretrained('base', checkpoint=sd, chw=(input_channels, input_size, input_size),
                                           output_dim=output_channels, freeze_body=freeze)
        

    ### TO ADD: geodino_upernet_pretrained :D <---
    ### TO ADD: geodino_upernet_pretrained :D <---
    ### TO ADD: geodino_upernet_pretrained :D <---
    ### TO ADD: geodino_upernet_pretrained :D <---
    ### TO ADD: geodino_upernet_pretrained :D <---

    # ----------------- Seasonal-Contrast ------------------

    elif model_name == 'seasonal_contrast':
        from models.model_Seco import seasonal_contrast
        seco_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        model = seasonal_contrast(checkpoint=path_model_weights, freeze_body=freeze, classifier=False, **seco_kwargs)
        # TODO test_input = torch.rand((?, ?, ?, ?))

    else:
        raise ValueError(f"Unsupported pretrained model {model_name}")

    # model_summary = summary(model, test_input)
    model_summary = None  # TODO implement test_input

    return model, model_summary
