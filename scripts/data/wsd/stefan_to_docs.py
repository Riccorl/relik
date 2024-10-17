import argparse
from pathlib import Path
import json

# from relik.retriever.indexers.document import Document


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    args = arg_parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        data = [line.strip() for line in f]

    with open(output_file, "w") as f:
        for i, stefan_doc in enumerate(data):
            my_doc = dict(
                id=i,
                text=stefan_doc,
                metadata={},
            )

            f.write(json.dumps(my_doc) + "\n")


if __name__ == "__main__":
    main()
