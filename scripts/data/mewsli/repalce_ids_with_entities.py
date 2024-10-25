import argparse
import json
from pathlib import Path
import random
import sys

from tqdm import tqdm


def repalce_ids(
    input_file: str,
    output_file: str,
    documents: str,
):

    documents_dict = {}
    with open(documents) as fi:
        for i, line in enumerate(tqdm(fi, desc="Loading documents")):
            doc = json.loads(line)
            documents_dict[doc["metadata"]["wikidata_id"]] = doc

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    nmes = 0
    with open(input_file) as fi, open(output_file, "w") as fo:
        for line in tqdm(fi, desc="Processing samples"):
            sample = json.loads(line)
            annotations = sample["doc_span_annotations"]
            converted_annotations = []
            for ann in annotations:
                total += 1
                if ann[-1] in documents_dict:
                    entity = documents_dict[ann[-1]]["text"]
                else:
                    nmes += 1
                    entity = "--NME--"
                converted_annotations.append(ann[:-1] + [entity])

            sample["doc_span_annotations"] = converted_annotations
            fo.write(json.dumps(sample) + "\n")
    
    print("Total annotations:", total)
    print("Number of missing entities:", nmes)
    print(f"Percetage of missing entities: {nmes/total*100:.2f}%")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("documents", type=str)
    args = arg_parser.parse_args()

    repalce_ids(**vars(args))


if __name__ == "__main__":
    main()
