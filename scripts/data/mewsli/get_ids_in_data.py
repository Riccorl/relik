import argparse
import json
from pathlib import Path

from tqdm import tqdm


def parse_data(input_dir: str, output_file: str, language: str):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    if language == "all":
        file_paths = input_dir.glob("*.jsonl")
    else:
        file_paths = [
            input_dir / f"{language}.train.jsonl",
            input_dir / f"{language}.val.jsonl",
            input_dir / f"{language}.test.jsonl",
        ]

    labels = set()
    for file_path in file_paths:
        with open(file_path) as fi:
            for i, line in enumerate(tqdm(fi)):
                sample = json.loads(line)
                for a in sample["doc_span_annotations"]:
                    labels.add(a[-1])

    with open(output_file, "w") as fo:
        for l in labels:
            fo.write(l + "\n")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_dir", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("--language", type=str, required=True)
    args = arg_parser.parse_args()

    parse_data(**vars(args))


if __name__ == "__main__":
    main()
