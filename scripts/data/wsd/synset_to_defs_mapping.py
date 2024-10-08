import argparse
import json
from pathlib import Path

from relik.retriever.indexers.document import Document
from nltk.corpus import wordnet as wn


def mapping(output_file: str):

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for i, synset in enumerate(wn.all_synsets()):
        lemmas = [lemma.name() for lemma in synset.lemmas()]
        mapping[synset.name()] = f"{', '.join(lemmas)}: {synset.definition()}"

    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("output_file", type=str)
    args = arg_parser.parse_args()

    mapping(**vars(args))


if __name__ == "__main__":
    main()
