import torch
import torch.nn as nn
import torch.nn.functional as F

mse_loss = torch.nn.MSELoss(reduction="mean")
ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
ce_none_loss = nn.CrossEntropyLoss(reduction="none")


def add_dim(tensor):
    if len(tensor.shape) == 1:
        return tensor.unsqueeze(dim=1)

    return tensor


def basic_mse_loss(pred, labels, weights, scale=1.0):
    return (F.mse_loss(add_dim(pred), add_dim(labels), reduction="none") * add_dim(weights)).mean() * scale


def basic_cross_loss(pred, labels, weights, scale=1.0):
    return (ce_none_loss(pred, labels) * weights).mean() * scale


def coord_loss(pred, labels, weights, scale=1.0):
    mse_lat = basic_mse_loss(pred[:, :2], labels[:, :2], weights, scale=1.0)
    mse_lon = basic_mse_loss(pred[:, 2:], labels[:, 2:], weights, scale=1.0)

    return ((mse_lat + mse_lon) / 2.0) * scale


def cossim_loss(embeddings, embeddings_aug, alpha=0.5):
    loss_cos_positive = F.cosine_embedding_loss(embeddings, embeddings_aug, torch.ones(embeddings.shape[0]).cuda())
    loss_cos_negative = F.cosine_embedding_loss(
        torch.flip(embeddings, dims=(0,)), embeddings_aug,
        torch.zeros(embeddings.shape[0]).cuda(),
        )

    euclidean_distance = torch.sqrt(torch.sum((embeddings - embeddings_aug) ** 2, dim=1))

    return loss_cos_positive * (1 + alpha) + loss_cos_negative * (1 - alpha) + euclidean_distance.mean()


def foundation_loss(og_recon, og_emb, og_pred, aug_recon, aug_emb, aug_pred, inputs, inputs_aug, labels):
    loss_og_recon = mse_loss(og_recon, inputs)
    loss_aug_recon = mse_loss(aug_recon, inputs_aug)

    og_pred_coords, og_pred_clouds, og_pred_buildings, og_pred_landcover = og_pred
    aug_pred_coords, aug_pred_clouds, aug_pred_buildings, aug_pred_landcover = aug_pred

    loss_og_coords = coord_loss(og_pred_coords, labels["coords"], labels["coords_weight"])
    loss_og_clouds = basic_cross_loss(og_pred_clouds, labels["clouds"], labels["cloud_weight"])
    loss_og_buildings = basic_mse_loss(og_pred_buildings, labels["buildings"], labels["buildings_weight"])
    loss_og_landcover = basic_cross_loss(og_pred_landcover, labels["landcover"], labels["landcover_weight"])

    loss_aug_coords = coord_loss(aug_pred_coords, labels["coords"], labels["coords_weight"])
    loss_aug_clouds = basic_cross_loss(aug_pred_clouds, labels["clouds"], labels["cloud_weight"])
    loss_aug_buildings = basic_mse_loss(aug_pred_buildings, labels["buildings"], labels["buildings_weight"])
    loss_aug_landcover = basic_cross_loss(aug_pred_landcover, labels["landcover"], labels["landcover_weight"])

    loss_sim = cossim_loss(og_emb, aug_emb)

    _recon_loss = (loss_og_recon + loss_aug_recon) / 2.0
    _loss_coords = (loss_og_coords + loss_aug_coords) / 2.0
    _loss_clouds = ((loss_og_clouds + loss_aug_clouds) / 2.0) * 100
    _loss_buildings = ((loss_og_buildings + loss_aug_buildings) / 2.0) * 100.0
    _loss_landcover = (loss_og_landcover + loss_aug_landcover) / 2.0
    _loss_sim = loss_sim

    loss = (
            _loss_coords
            + _loss_clouds
            + _loss_buildings
            + _loss_landcover
            + _loss_sim
            + _recon_loss
    )

    log = {
        "loss": loss,
        "rec": _recon_loss,
        "xy": _loss_coords,
        "cl": _loss_clouds,
        "b": _loss_buildings,
        "lc": _loss_landcover,
        "sim": _loss_sim,
    }

    return loss, log
