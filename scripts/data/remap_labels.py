import argparse
import json
from pathlib import Path


def map_data(input_folder: str, output_file: str, mapping: str):

    input_folder = Path(input_folder)
    output_file = Path(output_file)
    mapping = Path(mapping)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(mapping, "r") as f:
        mapping = json.load(f)

    with open(input_folder, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            data["doc_span_annotations"] = [
                # [start, end, mapping.get(label, label)]
                [start, end, mapping[label]]
                for start, end, label in data["doc_span_annotations"]
            ]
            f_out.write(json.dumps(data) + "\n")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_folder", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("mapping", type=str)
    args = arg_parser.parse_args()

    map_data(**vars(args))


if __name__ == "__main__":
    main()
