import argparse
import json
from pathlib import Path

from relik.retriever.indexers.document import Document
from nltk.corpus import wordnet as wn


def get_index(output_file: str):

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    documents = []
    for i, synset in enumerate(wn.all_synsets()):
        lemmas = [lemma.name() for lemma in synset.lemmas()]
        doc = Document(
            id=i,
            text=f"{', '.join(lemmas)}: {synset.definition()}",
            metadata={
                "synset": synset.name(),
                "definition": synset.definition(),
                "lemmas": [lemma.name() for lemma in synset.lemmas()],
                "pos": synset.pos(),
                "offset": synset.offset(),
            },
        )
        documents.append(doc)

    with open(output_file, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict()) + "\n")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("output_file", type=str)
    args = arg_parser.parse_args()

    get_index(**vars(args))


if __name__ == "__main__":
    main()
