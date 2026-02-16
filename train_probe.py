'''Read camera tokens from saved files and train probe (MLP) based on it.'''

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import os
import json
from torch.utils.data import TensorDataset, DataLoader, random_split
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from utils.probe_utils import get_custom_dir, make_optimizer, merge_camera_tokens
from model.probe_camera_token import PoseEstimator, get_canonical_idx

log = logging.getLogger(__name__)


def probe_loss(pose_predictor, batch_cam_tokens, batch_labels, cano_idx, rel_indices):
    """
    batch_cam_tokens: [B, V, D]
    gt_relative_poses: [B, V, 9]
    """
    v_all = batch_cam_tokens.shape[1]
    x_input = rearrange(batch_cam_tokens, 'b v d -> (b v) d')
    pred_para = pose_predictor(x_input, v_all, cano_idx, rel_indices)
    assert pred_para.size(-1) == batch_labels.size(
        -1), "Output dimension of the probe must match the label dimension!"
    pred_para = rearrange(pred_para, '(b v) d -> b v d', v=v_all)
    pred_rel = pred_para[:, rel_indices]  # [B, V-1, 9]
    loss = F.mse_loss(pred_rel.reshape(-1, pred_para.size(-1)),
                      batch_labels[:, rel_indices].reshape(-1, pred_para.size(-1)), reduction='mean')
    return loss


def c2w_to_9d(c2w: torch.Tensor, cano_idx: int, batch_size=5000):
    '''
    Convert absolute camera pose to relative pose against canonical view,
    then extracting the 9d parameters for regression, returning a [N, v-1, 9] tensor.
    Formula: T_rel = T_cano^{-1} * T_target
    c2w: [N, 24, 4, 4]
    '''
    N, V, _, _ = c2w.shape
    all_labels = []

    for i in range(0, N, batch_size):
        c2w_batch = c2w[i: i + batch_size]
        B = c2w_batch.shape[0]

        t_cano = c2w_batch[:, cano_idx: cano_idx + 1]
        t_cano_inv = torch.inverse(t_cano)
        relative_T = t_cano_inv @ c2w_batch

        # extracting 9D (6 rotation + 3 translation) labels
        # Zhou et al., "On the Continuity of Rotation Representations
        # in Neural Networks", Appendix B
        rot_6d = relative_T[:, :, :3, :2].reshape(B, V, 6)
        trans_3d = relative_T[:, :, :3, 3]
        batch_labels = torch.cat([rot_6d, trans_3d], dim=-1)
        all_labels.append(batch_labels.cpu())

    return torch.cat(all_labels, dim=0)


@hydra.main(version_base=None, config_path="configs/", config_name="rayzer_probe")
def main(config: DictConfig):
    '''Train probe for each layer based on the camera tokens.'''
    logging.getLogger().setLevel(logging.INFO)
    log.info("Loaded config:\n" + OmegaConf.to_yaml(config))

    # set random seed
    seed = int(config.probe.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using Device: {device}")

    # load camera tokens
    camera_tokens = merge_camera_tokens(config.probe.camera_tokens_folder)
    # camera_tokens = torch.load(
    #     config.probe.camera_tokens_path, weights_only=False)

    v_all = config.training.num_views
    cano_idx, rel_indices = get_canonical_idx(v_all, config)
    labels = c2w_to_9d(camera_tokens['c2w'], cano_idx)  # [N, 24, 9]

    test_loss_layer = {}
    for layer, cam_tokens in camera_tokens['camera_token'].items():
        pose_predictor = PoseEstimator(config)
        pose_predictor.to(device)

        dataset = TensorDataset(cam_tokens, labels)
        gen = torch.Generator().manual_seed(42)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=gen
        )
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config.probe.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config.probe.batch_size, shuffle=False)

        log.info(f"Training probe for {layer}...")
        optimizer = make_optimizer(config.probe.optimizer,
                                   pose_predictor.parameters(), lr=config.probe.lr)

        best_test_loss, best_epoch = float('inf'), -1
        for epoch in range(config.probe.max_epoch):
            log.info(f"Epoch {epoch+1}/{config.probe.max_epoch}")
            pose_predictor.train()
            train_loss, test_loss = 0, 0
            for data, label in train_dataloader:
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                loss = probe_loss(pose_predictor, data,
                                  label, cano_idx, rel_indices)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().cpu() * data.size(0)

            pose_predictor.eval()
            with torch.no_grad():
                for data, label in test_dataloader:
                    data = data.to(device)
                    label = label.to(device)
                    loss = probe_loss(
                        pose_predictor, data, label, cano_idx, rel_indices)
                    test_loss += loss.detach().cpu() * data.size(0)

            log.info('Epoch {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(
                epoch+1, train_loss / len(train_dataloader), test_loss / len(test_dataloader)))

            # early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch + 1
            if epoch + 1 > best_epoch + config.probe.early_stopping:
                log.info(
                    f"Early stopping at epoch {epoch+1} with best test loss {best_test_loss:.4f} at epoch {best_epoch}")
                break

        test_loss_layer[layer] = best_test_loss.item() / len(test_dataloader)

    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(test_loss_layer, f, indent=2)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("calc_path", get_custom_dir)
    main()
    # python train_probe.py probe.camera_tokens_path=rayzer_camera_tokens.pt
