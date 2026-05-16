import argparse
import json
import os
import random
import string

from tqdm import tqdm


def generate_random_vocab(size=10000):
    """Generates a list of random 'words' to sample from for speed."""
    print(f"Generating random vocabulary of {size:,} words...")
    vocab = []
    for _ in range(size):
        # Random words between 3 and 8 characters long
        word_len = random.randint(3, 8)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        vocab.append(word)
    return vocab


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment a JSONL pool with synthetic random-vocabulary documents."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the base JSONL file to extend (one document per line, 'text' field).",
    )
    parser.add_argument(
        "--output-template",
        required=True,
        help=(
            "Directory template for output. Must contain '{pct}', which is replaced by "
            "int(growth * 100). One directory is created per --growth value."
        ),
    )
    parser.add_argument(
        "--growth",
        type=float,
        nargs="+",
        default=[2.0],
        help="One or more growth multipliers (e.g. 2.0 = add enough new words to grow the pool by 200%%).",
    )
    parser.add_argument(
        "--new-doc-length",
        type=int,
        default=2000,
        help="Number of words per generated document.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Size of the random vocabulary the synthetic documents are sampled from.",
    )
    parser.add_argument(
        "--output-filename",
        default="dclm_pool_1b_1x.chunk.00.jsonl",
        help="Filename used for the merged JSONL inside each output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if "{pct}" not in args.output_template:
        raise ValueError("--output-template must contain '{pct}'")

    base_lines = 0
    total_original_words = 0

    # 1. First pass: count words AND lines in base dataset (no RAM accumulation).
    print(f"Scanning {args.input} to count words and lines...")
    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning original docs", unit=" docs"):
            base_lines += 1
            doc = json.loads(line)
            total_original_words += len(doc.get("text", "").split())

    print(f"\nTotal original words: {total_original_words:,}")
    print(f"Total base lines: {base_lines:,}")

    # Generate vocabulary once to be reused across all targets.
    vocab = generate_random_vocab(size=args.vocab_size)

    for target in args.growth:
        pct_int = int(target * 100)
        print("\n" + "=" * 50)
        print(f"🚀 PROCESSING {pct_int}% INCREASE (INTERLEAVED SHUFFLE)")
        print("=" * 50)

        target_new_words = int(total_original_words * target)
        output_dir = args.output_template.format(pct=pct_int)
        output_file = os.path.join(output_dir, args.output_filename)
        temp_new_docs_file = os.path.join(output_dir, "temp_new_docs.jsonl")

        os.makedirs(output_dir, exist_ok=True)
        print(f"Target new random words to add ({pct_int}%): {target_new_words:,}\n")

        # 2. Generate new random documents straight to a temporary file.
        new_lines = 0
        words_added = 0
        first_doc_printed = False

        print("Generating random documents and writing to temporary file...")
        with open(temp_new_docs_file, "w", encoding="utf-8") as temp_f:
            with tqdm(total=target_new_words, desc=f"Generating {pct_int}% docs", unit=" words") as pbar:
                while words_added < target_new_words:
                    words_to_add = min(args.new_doc_length, target_new_words - words_added)

                    random_text = " ".join(random.choices(vocab, k=words_to_add))

                    new_doc_dict = {
                        "text": random_text,
                        "is_random_injected": True,
                    }

                    new_doc_str = json.dumps(new_doc_dict, ensure_ascii=False, separators=(",", ":")) + "\n"
                    temp_f.write(new_doc_str)
                    new_lines += 1

                    if not first_doc_printed:
                        print("\n\n" + "-" * 50)
                        print("👀 PREVIEW OF FIRST GENERATED DOCUMENT:")
                        print("-" * 50)
                        preview_text = json.dumps(new_doc_dict, indent=2)
                        if len(preview_text) > 1000:
                            preview_text = preview_text[:1000] + "\n  ...\n  [TEXT TRUNCATED FOR DISPLAY]\n}"
                        print(preview_text)
                        print("-" * 50 + "\n")
                        first_doc_printed = True

                    words_added += words_to_add
                    pbar.update(words_to_add)

        print(f"Finished generating {new_lines:,} new documents.")

        # 3. Interleave the base file and the temp file into the final output file.
        print("\nMerging files with simulated on-the-fly random shuffle...")

        with open(args.input, "r", encoding="utf-8") as f_base, \
             open(temp_new_docs_file, "r", encoding="utf-8") as f_new, \
             open(output_file, "w", encoding="utf-8") as f_out:

            remaining_base = base_lines
            remaining_new = new_lines
            total_lines = base_lines + new_lines

            with tqdm(total=total_lines, desc="Interleaving files", unit=" lines") as pbar:
                while remaining_base > 0 or remaining_new > 0:
                    total_remaining = remaining_base + remaining_new

                    if random.random() < (remaining_base / total_remaining):
                        f_out.write(f_base.readline())
                        remaining_base -= 1
                    else:
                        f_out.write(f_new.readline())
                        remaining_new -= 1

                    pbar.update(1)

        # 4. Cleanup.
        print("\nCleaning up temporary files...")
        os.remove(temp_new_docs_file)

        print(f"✅ Interleaved shuffle complete! Final file ready at: {output_file}")
        print(f"Done with {pct_int}% increase! 🎉\n")


if __name__ == "__main__":
    main()
