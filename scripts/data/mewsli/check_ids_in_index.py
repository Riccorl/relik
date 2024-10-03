import argparse
import json
from pathlib import Path

from tqdm import tqdm


def check_ids(document_path: str, ids_path: str):
    ids = set()
    with open(ids_path) as fi:
        for line in fi:
            ids.add(line.strip())

    doc_ids = {}
    with open(document_path) as fi:
        for i, line in enumerate(tqdm(fi)):
            doc = json.loads(line)
            doc_ids[doc["id"]] = doc

    missing = set()
    totally_missing = set()

    for w_id in ids:
        if w_id not in doc_ids:
            missing.add(w_id)
            continue
        if len(doc_ids[w_id]["labels"]) == 0:
            totally_missing.add(w_id)

    print(f"Missing {len(missing)} out of {len(ids)} ids.")
    print(f"Totally missing: {len(totally_missing)}")

    with open("missing.txt", "w") as fo:
        for w_id in missing:
            fo.write(w_id + "\n")
    with open("totally_missing.txt", "w") as fo:
        for w_id in totally_missing:
            fo.write(w_id + "\n")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("document_path", type=str)
    arg_parser.add_argument("ids_path", type=str)
    args = arg_parser.parse_args()

    check_ids(**vars(args))


if __name__ == "__main__":
    main()
