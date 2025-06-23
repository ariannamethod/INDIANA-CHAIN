Launch:
```bash
sbatch --nodes=2 slurm/train.slurm Qwen2.5-Math-7B sft config_agent zero3 
```
Refers to the config  recipes/SmolLM2-1.7B-Instruct/sft/config_agent.yaml
zero3 is one of the accelerate configs in recipes/accelerate_configs