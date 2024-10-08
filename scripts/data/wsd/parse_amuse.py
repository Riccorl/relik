import argparse
import json
from pathlib import Path
from typing import NamedTuple, Optional, List, Callable, Tuple, Iterable
import xml.etree.cElementTree as ET
from xml.dom import minidom

pos_map = {
    # U-POS
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
    "PROPN": "n",
    # PEN
    "AFX": "a",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "MD": "v",
    "NN": "n",
    "NNP": "n",
    "NNPS": "n",
    "NNS": "n",
    "RB": "r",
    "RP": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


def amuse_to_relik(input_file: str, output_file: str):

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        data = json.load(f)

    # {
    #     "doc_id": int,  # Unique identifier for the document
    #     "doc_text": txt,  # Text of the document
    #     "doc_span_annotations": [  # Char level annotations
    #         [start, end, label],
    #         [start, end, label],
    #         ...,
    #     ],
    # }

    with open(output_file, "w") as f:
        for i, (sentence_id, amuse_sentence) in enumerate(data.items()):
            relik_sentence = {
                "doc_id": i,
                "doc_original_id": sentence_id,
                "doc_words": amuse_sentence["words"],
                "doc_lemmas": amuse_sentence["lemmas"],
                "doc_pos": amuse_sentence["pos_tags"],
            }

            doc_span_annotations = []
            for token_id, labels in amuse_sentence["senses"].items():
                for label in labels:
                    doc_span_annotations.append(
                        [int(token_id), int(token_id) + 1, label]
                    )

            relik_sentence["doc_span_annotations"] = doc_span_annotations
            f.write(json.dumps(relik_sentence) + "\n")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    args = arg_parser.parse_args()

    amuse_to_relik(**vars(args))


if __name__ == "__main__":
    main()
