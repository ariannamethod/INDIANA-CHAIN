import argparse

from .generation import generate_consistent_text, generate_text
from .model import IndianaCConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Indiana-C text generation")
    parser.add_argument("prompt", nargs="?", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="show reasoning log")
    parser.add_argument(
        "--precision",
        type=int,
        choices=[2, 4, 8],
        default=2,
        help="weight quantization precision in bits",
    )
    parser.add_argument(
        "--consistency",
        type=int,
        default=1,
        help="number of attempts to ensure answer consistency",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="enable self-verification through reflection",
    )
    args = parser.parse_args()

    config = IndianaCConfig(vocab_size=256, quantization_bits=args.precision)
    if args.consistency > 1:
        result = generate_consistent_text(
            args.prompt,
            n=args.consistency,
            max_new_tokens=args.max_new_tokens,
            config=config,
            self_reflect=args.reflect,
        )
        print(result)
    else:
        result = generate_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            config=config,
            log_reasoning=args.verbose,
            self_reflect=args.reflect,
        )
        if args.verbose:
            text, meta = result
            print(text)
            print(
                f"LOG@{meta['timestamp']} | Complexity: {meta['complexity']} | Entropy: {meta['entropy']:.2f}"
            )
        else:
            print(result)


if __name__ == "__main__":
    main()
