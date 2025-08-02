import argparse

from .generation import generate_text
from .model import IndianaCConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Indiana-C text generation")
    parser.add_argument("prompt", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    config = IndianaCConfig(vocab_size=256)
    text = generate_text(args.prompt, max_new_tokens=args.max_new_tokens, config=config)
    print(text)


if __name__ == "__main__":
    main()
