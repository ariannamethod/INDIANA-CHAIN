Launch:
```bash
sbatch --nodes=1 slurm/train.slurm --model SmolLM2-1.7B-Instruct --task sft --config agent --accelerator zero3
```
Refers to the config  recipes/SmolLM2-1.7B-Instruct/sft/config_agent.yaml
zero3 is one of the accelerate configs in recipes/accelerate_configs



Launch VLM training:
```bash
sbatch --nodes=1 slurm/train.slurm --model Qwen2.5-VL-3B-Instruct --task sft --config agent --accelerator zero3
```

Simple mode
```bash
sbatch --nodes=1 slurm/train.slurm --model Qwen2.5-VL-3B-Instruct --task sft --config agent --accelerator ddp
```