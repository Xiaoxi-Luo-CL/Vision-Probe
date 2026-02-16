# RayZer-Based Probing

First, run `inference.py` to cache each image's camera tokens. This process could be slow, and you could set `--nproc_per_node` to the number of GPUs to accelerate. For example, if I have 2 gpus:

```bash
torchrun --nproc_per_node 2 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29507 \
    inference.py --config "configs/rayzer_dl3dv.yaml" \
    training.dataset_path = "./data/dl3dv10k_benchmark.txt" \
    training.batch_size_per_gpu = 4 \
    training.target_has_input =  false \
    inference.if_inference = true \
    inference.view_idx_file_path = "./data/rayzer_evaluation_index_dl3dv_even.json" \
    inference.model_path = ./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \
    inference_out_root = ./experiments/evaluation/test
```

Results will be saved to `rayzer_camera_tokens/rank_{i}.pt`, where `rank_{i}` indicates the i-th process.

Then run `train_probe.py` to train probe based on cached hidden states.
```
python train_probe.py probe.camera_tokens_folder=rayzer_camera_tokens/
```