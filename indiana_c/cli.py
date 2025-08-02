import argparse

from .generation import generate_consistent_text, generate_text
from .model import IndianaCConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Indiana-C text generation")
    parser.add_argument("prompt", nargs="?", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="show reasoning log")
    parser.add_argument(
        "--consistency",
        type=int,
        default=1,
        help="number of attempts to ensure answer consistency",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="top-k filtering")
    parser.add_argument("--top-p", type=float, default=None, help="nucleus top-p sampling")
    args = parser.parse_args()

    config = IndianaCConfig(vocab_size=256)
    if args.consistency > 1:
        result = generate_consistent_text(
            args.prompt,
            n=args.consistency,
            max_new_tokens=args.max_new_tokens,
            config=config,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(result)
    else:
        result = generate_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            config=config,
            log_reasoning=args.verbose,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
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
