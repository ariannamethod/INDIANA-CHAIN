import argparse

from .generation import generate_with_think
from .model import IndianaCConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Indiana-C text generation")
    parser.add_argument("prompt", nargs="?", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--show-thoughts", action="store_true", help="display thoughts before final answer")
    args = parser.parse_args()

    config = IndianaCConfig(vocab_size=256)
    thoughts, final = generate_with_think(
        args.prompt, max_new_tokens=args.max_new_tokens, config=config
    )
    if args.show_thoughts:
        if thoughts:
            print(thoughts)
        print(final)
    else:
        print(final)


if __name__ == "__main__":
    main()
