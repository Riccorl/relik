import argparse
import json
from pathlib import Path

from relik.retriever.indexers.document import Document
from nltk.corpus import wordnet as wn


def get_index(input_folder: str, output_file: str):

    input_folder = Path(input_folder)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(input_folder / "train.jsonl", "r") as f:
        data = json.load(f)
    with open(input_folder / "validation.jsonl", "r") as f:
        data += json.load(f)
    with open(input_folder / "test.jsonl", "r") as f:
        data += json.load(f)
    
    docs = {}
    for sample in data:
        for passage in sample["positive_ctxs"]:
            if passage["title"] not in docs:
                docs[passage["title"]] = passage
    
    print("Synsets:", len(docs))

    # documents = []
    # for i, sample in data:
    #     doc = Document(
    #         id=i,
    #         text=sample["text"],
    #         metadata={
    #             "sense_key": sample["title"],
    #             "synset": wn.synset_from_pos_and_offset(
    #                 sample["text"][-1], int(sample["text"][:-1])
    #             ).name(),
    #         },
    #     )
    #     documents.append(doc)

    # with open(output_file, "w") as f:
    #     for doc in documents:
    #         f.write(json.dumps(doc.to_dict()) + "\n")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_folder", type=str)
    arg_parser.add_argument("output_file", type=str)
    args = arg_parser.parse_args()

    get_index(**vars(args))


if __name__ == "__main__":
    main()
