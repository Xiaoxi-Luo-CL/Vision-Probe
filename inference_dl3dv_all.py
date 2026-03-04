# Copyright (c) 2025 Hanwen Jiang. Created for the RayZer project.

import importlib
import json
import os
import torch
from torch.utils.data import DataLoader, Subset
from setup import init_config
from model.rayzer import RayZer, RayZerHiddenCollector
from collections import defaultdict
from tqdm import tqdm

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

seed = int(config.training.get("seed", 42))
scene_start = int(config.training.get("scene_start", 0))
scene_end = config.training.get("scene_end", None)
device = torch.device("cuda:0")


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    'tf32': torch.float32
}


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset_dl3dv.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)  # each element [24, 3, 256, 256]

if scene_end is None:
    scene_end = len(dataset)
scene_end = min(int(scene_end), len(dataset))
scene_start = max(0, scene_start)
if scene_start >= scene_end:
    print(f"Empty shard [{scene_start}, {scene_end}) for dataset size {len(dataset)}. Nothing to do.")
    exit(0)

subset = Subset(dataset, range(scene_start, scene_end))

batch_size = int(config.training.batch_size_per_gpu)
num_workers = int(config.training.num_workers)
prefetch_factor = int(config.training.prefetch_factor)
if config.inference.get("if_inference", False):
    # Inference can run with a safer data-loader profile than training.
    batch_size = int(config.inference.get("batch_size_per_gpu", batch_size))
    num_workers = int(config.inference.get("num_workers", num_workers))
    prefetch_factor = int(config.inference.get("prefetch_factor", prefetch_factor))

dataloader_kwargs = {
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": num_workers,
    "persistent_workers": num_workers > 0,
    "pin_memory": False,
    "drop_last": False,  # when extracting hidden layers, we want all
}
if num_workers > 0:
    dataloader_kwargs["prefetch_factor"] = prefetch_factor

dataloader = DataLoader(
    subset,
    **dataloader_kwargs,
)


_, class_name = config.model.class_name.rsplit(".", 1)
assert class_name == 'RayZer', 'Currently only support RayZer for inference'
model = RayZer(config).to(device)

if config.inference.get("model_path", None) is not None and config.inference.get("if_inference", False):
    # use the specified checkpoint if specified
    checkpoint = torch.load(config.inference.model_path,
                            map_location="cpu", weights_only=True)
    # strict=False because the loss computer is different from internal version, BE CAREFUL!
    model.load_state_dict(checkpoint["model"], strict=False)
elif config.inference.get("model_path", None) is None and config.inference.get("if_inference", False):
    # otherwise, use the last training checkpoint
    model.load_ckpt(config.training.checkpoint_dir)
model.eval()
collector = RayZerHiddenCollector(
    model, layer_to_collect=[i for i in range(config.model.transformer.encoder_n_layer)])
collector.register_hooks()

save_root = config.inference.get("save_root", "rayzer_camera_tokens/")
save_dir = os.path.join(save_root, f"seed_{seed}")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"rank_{scene_start:05d}.pt")

scene_path_lookup = {}
for scene_path in dataset.all_scene_paths:
    scene_name = scene_path.strip().split("/")[-2]
    scene_path_lookup[scene_name] = scene_path.strip()
scene_image_names_cache = {}


def get_image_names(scene_name, frame_indices):
    if scene_name not in scene_image_names_cache:
        with open(scene_path_lookup[scene_name], "r") as f:
            scene_json = json.load(f)
        scene_image_names_cache[scene_name] = [
            os.path.basename(frame["file_path"]) for frame in scene_json["frames"]
        ]
    image_names = scene_image_names_cache[scene_name]
    return [image_names[i] for i in frame_indices]


data_to_save = {
    'camera_token': defaultdict(list),
    'c2w': [],
    'scene_name': [],
    'frame_indices': [],
    'image_names': [],
    'seed': seed,
    'scene_start': scene_start,
    'scene_end': scene_end,
}

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in tqdm(dataloader, desc=f"seed={seed} scenes=[{scene_start},{scene_end})"):
        batch = {k: v.to(device) if type(
            v) == torch.Tensor else v for k, v in batch.items()}
        result = model(batch)

        v_all = batch['image'].shape[1]
        camera_tokens = collector.collect_camera_token(
            result['n_cam'], result['n_patch'], v_all)
        for k, v in camera_tokens.items():
            data_to_save['camera_token'][k].append(v.cpu())
        data_to_save['c2w'].append(batch['c2w'].cpu())
        batch_frame_indices = batch['index'][:, :, 0].long().cpu()
        batch_scene_names = list(batch['scene_name'])
        data_to_save['frame_indices'].append(batch_frame_indices)
        data_to_save['scene_name'].extend(batch_scene_names)
        for i, scene_name in enumerate(batch_scene_names):
            frame_indices = batch_frame_indices[i].tolist()
            data_to_save['image_names'].append(get_image_names(scene_name, frame_indices))

    collector.unregister()
    if len(data_to_save['c2w']) == 0:
        raise RuntimeError(f"No samples processed for shard [{scene_start}, {scene_end}).")
    data_to_save['c2w'] = torch.cat(data_to_save['c2w'], dim=0)
    data_to_save['frame_indices'] = torch.cat(data_to_save['frame_indices'], dim=0)
    for k, v in data_to_save['camera_token'].items():
        data_to_save['camera_token'][k] = torch.cat(v, dim=0)

    torch.save(data_to_save, save_path)
    torch.cuda.empty_cache()
