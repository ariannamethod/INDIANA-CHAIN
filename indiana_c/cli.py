import argparse

from .generation import generate_text
from .model import IndianaCConfig
from .memory import log_interaction


def main() -> None:
    parser = argparse.ArgumentParser(description="Indiana-C text generation")
    parser.add_argument("prompt", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--quantize", action="store_true", help="use 2-bit weights")
    args = parser.parse_args()

    config = IndianaCConfig(vocab_size=256)
    text = generate_text(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        config=config,
        quantize=args.quantize,
    )
    print(text)
    log_interaction(args.prompt, text)


if __name__ == "__main__":
    main()
