# Indiana-C

Indiana-C is a minimal reasoning engine built to stand alone on the CPU. It is powered by the open-source [DEEPSEEK R1](https://github.com/deepseek-ai/DeepSeek-R1) reasoning stack and keeps the deliberate `<think>`-style reflection and step-by-step planning introduced by that project while removing every dependency on external hosting platforms.

From DEEPSEEK R1 we borrow the heart of the system: the reasoning kernel that structures thought into explicit chains, the self-verification loop that double-checks outputs, and the lightweight memory hooks that capture each interaction for future tuning. These components form the inner engine that lets Indiana-C explain itself as it works.

Indiana-C trims away heavy infrastructure around that core, runs entirely on commodity CPUs, and logs every prompt/response pair for autonomous improvement. Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), the kernel remains tiny and readable—a fresh start free from old tensors and designed for autonomy.

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
