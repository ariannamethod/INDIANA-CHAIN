Launch:
```bash
sbatch --nodes=1 slurm/train.slurm SmolLM2-1.7B-Instruct sft agent zero3
```
Refers to the config  recipes/SmolLM2-1.7B-Instruct/sft/agent.yaml
zero3 is one of the accelerate configs in recipes/accelerate_configs