import argparse
from pathlib import Path
import json


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("documents", type=str)
    args = arg_parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.documents, "r") as f:
        documents = [json.loads(line) for line in f]

    documents = {doc["text"]: doc for doc in documents}

    with open(input_file, "r") as f:
        data = json.load(f)

    with open(output_file, "w") as f:
        for i, stefan_sentence in enumerate(data):
            dpr_sentence = {
                "id": i,
                "question": stefan_sentence["question"],
                "negative_ctxs": "",
                "hard_negative_ctxs": "",
            }
            positives = []
            for positive in stefan_sentence["positive_ctxs"]:
                doc_positives = documents[positive["text"]]
                positives.append(doc_positives)

            if len(positives) == 0:
                continue

            dpr_sentence["positive_ctxs"] = positives

            f.write(json.dumps(dpr_sentence) + "\n")


if __name__ == "__main__":
    main()
