# Indiana-C

Indiana-C is a minimal reasoning engine built to stand alone on the CPU. It keeps the deliberate `<think>`-style reflection and step-by-step planning introduced by the open-source DEEPSEEK R1 engine while removing every dependency on external hosting platforms.

The project borrows the R1 core as its reasoning heart. DEEPSEEK released an autonomous chain-of-thought stack, and Indiana-C reuses that public skeleton so CPU deployments still enjoy explicit traces and self-verifying steps without cloud services.

On top of the borrowed core, Indiana-C layers a self-monitoring memory inspired by previous experiments like SUPPERTIME and D2C. Each run snapshots the entire codebase and logs prompts and outputs into an embedded database so the system can study and fine-tune itself offline.

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), the core is tiny and readable. Indiana-C is not a fork but a fresh kernel, free from old tensors and designed for autonomy.

## Features

- Pure PyTorch implementation
- CPU-only execution
- Retains R1 traits such as explicit reasoning traces and self-verification

## Usage

```bash
python -m indiana_c.cli "2+2="
```

## Acknowledgements

Indiana-C draws from the R1 engine and from the nanoGPT project by Andrej Karpathy.
