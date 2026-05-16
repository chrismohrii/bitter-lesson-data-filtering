import argparse
import json
import os
import random

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Augment a JSONL pool with documents whose words have been shuffled. "
            "New documents are drawn from --source (offset by --gb-offset bytes), shuffled, "
            "and then interleaved with --base into --output-template."
        )
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Path to the base JSONL file to extend.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the JSONL file new documents are pulled from (read past --gb-offset).",
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
        default=[8.0],
        help="One or more growth multipliers (e.g. 8.0 = add enough new words to grow the pool by 800%%).",
    )
    parser.add_argument(
        "--gb-offset",
        type=float,
        default=20.0,
        help="Number of GB to skip into --source before reading new documents.",
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

    gb_offset_bytes = int(args.gb_offset * 1024 * 1024 * 1024)

    base_lines = 0
    total_original_words = 0

    # 1. First pass: count words AND lines in base dataset.
    print(f"Counting words and lines in base dataset: {args.base}...")
    with open(args.base, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning base file", unit=" docs"):
            base_lines += 1
            doc = json.loads(line)
            total_original_words += len(doc.get("text", "").split())

    print(f"\nTotal original words: {total_original_words:,}")
    print(f"Total base lines: {base_lines:,}")

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
        print(f"Target new words to add ({pct_int}%): {target_new_words:,}\n")

        words_added = 0
        new_lines = 0
        first_doc_printed = False

        # 2. Extract, shuffle, and stream the new source documents to a TEMP file.
        print(f"Pulling from {args.source} and writing to temporary file...")
        with open(temp_new_docs_file, "w", encoding="utf-8") as temp_f, \
             open(args.source, "rb") as source_f:

            source_f.seek(gb_offset_bytes)
            source_f.readline()  # Align to next newline.

            with tqdm(total=target_new_words, desc="Generating new docs", unit=" words") as pbar:
                while words_added < target_new_words:
                    raw_line = source_f.readline()
                    if not raw_line:
                        raise EOFError("Reached the end of the source file!")

                    line_str = raw_line.decode("utf-8", errors="ignore")

                    try:
                        source_doc = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    words = source_doc.get("text", "").split()
                    if not words:
                        continue

                    random.shuffle(words)

                    words_needed = target_new_words - words_added
                    if len(words) > words_needed:
                        words = words[:words_needed]

                    new_doc_dict = {
                        "text": " ".join(words),
                        "is_shuffled_words_injected": True,
                        "original_source_length": len(words),
                    }

                    new_doc_str = json.dumps(new_doc_dict, ensure_ascii=False, separators=(",", ":")) + "\n"
                    temp_f.write(new_doc_str)
                    new_lines += 1

                    if not first_doc_printed:
                        print("\n\n" + "-" * 50)
                        print("👀 PREVIEW OF FIRST SHUFFLED DOCUMENT:")
                        print("-" * 50)
                        preview_text = json.dumps(new_doc_dict, indent=2)
                        if len(preview_text) > 1000:
                            preview_text = preview_text[:1000] + "\n  ...\n  [TEXT TRUNCATED FOR DISPLAY]\n}"
                        print(preview_text)
                        print("-" * 50 + "\n")
                        first_doc_printed = True

                    words_added += len(words)
                    pbar.update(len(words))

        print(f"Finished generating {new_lines:,} new documents.")

        # 3. Interleave the base file and the temp file into the final output file.
        print("\nMerging files with simulated on-the-fly random shuffle...")

        with open(args.base, "r", encoding="utf-8") as f_base, \
             open(temp_new_docs_file, "r", encoding="utf-8") as f_new, \
             open(output_file, "w", encoding="utf-8") as f_out:

            remaining_base = base_lines
            remaining_new = new_lines
            total_lines = base_lines + new_lines

            with tqdm(total=total_lines, desc="Interleaving files", unit=" lines") as pbar:
                while remaining_base > 0 or remaining_new > 0:
                    total_remaining = remaining_base + remaining_new

                    # Probabilistic selection models a perfect mathematical shuffle.
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
