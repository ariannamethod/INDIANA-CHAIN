# Indiana-C

Indiana-C is a minimal reasoning engine built to stand alone on the CPU. It keeps the deliberate `<think>`-style reflection and step-by-step planning introduced by the original R1 engine while removing every dependency on external hosting platforms.

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
