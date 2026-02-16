# Copyright (c) 2025 Hanwen Jiang. Created for the RayZer project.

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from model.rayzer import RayZer, RayZerHiddenCollector
from collections import defaultdict
from tqdm import tqdm

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


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
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)  # each element [24, 3, 256, 256]

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=False,  # when extracting hidden layers, we want all
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()


_, class_name = config.model.class_name.rsplit(".", 1)
assert class_name == 'RayZer', 'Currently only support RayZer for inference'
model = RayZer(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
print(
    f"Rank: {dist.get_rank()} | PID: {os.getpid()} | Parent PID: {os.getppid()}")

if config.inference.get("model_path", None) is not None and config.inference.get("if_inference", False):
    # use the specified checkpoint if specified
    checkpoint = torch.load(config.inference.model_path,
                            map_location="cpu", weights_only=True)
    # strict=False because the loss computer is different from internal version, BE CAREFUL!
    model.module.load_state_dict(checkpoint["model"], strict=False)
elif config.inference.get("model_path", None) is None and config.inference.get("if_inference", False):
    # otherwise, use the last training checkpoint
    model.module.load_ckpt(config.training.checkpoint_dir)

dist.barrier()


datasampler.set_epoch(0)
model.eval()
collector = RayZerHiddenCollector(
    model.module if hasattr(model, 'module') else model, layer_to_collect=[i for i in range(config.model.transformer.encoder_n_layer)])
collector.register_hooks()

save_path = 'rayzer_camera_tokens/'
os.makedirs(save_path, exist_ok=True)
data_to_save = {'camera_token': defaultdict(list), 'c2w': []}

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in tqdm(dataloader):
        batch = {k: v.to(ddp_info.device) if type(
            v) == torch.Tensor else v for k, v in batch.items()}
        result = model(batch)

        v_all = batch['image'].shape[1]
        camera_tokens = collector.collect_camera_token(
            result['n_cam'], result['n_patch'], v_all)
        for k, v in camera_tokens.items():
            data_to_save['camera_token'][k].append(v)
        data_to_save['c2w'].append(batch['c2w'].cpu())

    collector.unregister()
    # stack along the first dimension
    data_to_save['c2w'] = torch.cat(data_to_save['c2w'], dim=0)
    for k, v in data_to_save['camera_token'].items():
        data_to_save['camera_token'][k] = torch.cat(v, dim=0)

    torch.save(data_to_save, os.path.join(
        save_path, f"rank_{dist.get_rank()}.pt"))
    torch.cuda.empty_cache()


dist.barrier()
dist.destroy_process_group()
exit(0)
