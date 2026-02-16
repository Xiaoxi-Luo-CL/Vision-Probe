import datetime
import torch
import os
import re
import glob
from collections import defaultdict


def get_custom_dir(probe_type: str, hidden_dim: list, model_name: str, seed: int):
    model_name = model_name.split('.')[-1]
    # time at present
    cur_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if probe_type != 'linear':
        hidden_size = '-'.join(map(str, hidden_dim))
        return f"outputs/{model_name}/{probe_type}-{hidden_size}/seed-{seed}/{cur_time}"
    else:
        return f"outputs/{model_name}/linear/seed-{seed}/{cur_time}"


def make_optimizer(optimizer_name: str, parameters, lr: float) -> torch.optim.Optimizer:
    """Build and return a torch optimizer for probe training.
    Params:
        optimizer_name: Optimizer identifier. Supports "adamw", "adam", and "sgd".
        lr: Learning rate for the optimizer.
    """
    name = str(optimizer_name).lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. Expected one of: adamw, adam, sgd."
    )


def merge_camera_tokens(folder_path):
    '''Load all files of the form "rank_*.pt" (which save camera tokens) in folder,
      merge them into a single dictionary with concatenated tensors.'''
    file_list = glob.glob(os.path.join(folder_path, "rank_*.pt"))
    file_list.sort(key=lambda f: int(
        re.findall(r'\d+', os.path.basename(f))[0]))
    print(f"Found {len(file_list)} rank files. Merging...")

    merged_data = {'camera_token': defaultdict(list), 'c2w': []}

    for file_path in file_list:
        data = torch.load(file_path, weights_only=False)
        merged_data['c2w'].append(data['c2w'])
        for layer, tokens in data['camera_token'].items():
            merged_data['camera_token'][layer].append(tokens)

    # Concatenate all lists into single tensors
    all_camera_tokens = {
        'c2w': torch.cat(merged_data['c2w'], dim=0),
        'camera_token': {
            layer: torch.cat(token_list, dim=0)
            for layer, token_list in merged_data['camera_token'].items()
        }
    }
    return all_camera_tokens
