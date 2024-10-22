import argparse
import json
from pathlib import Path
import random
from tqdm import tqdm


def parse_files(input_folder, output_file):
    input_folder = Path(input_folder)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = []

    for input_file in tqdm(input_folder.iterdir()):
        if input_file.name == "all.dpr.wikidata.jsonl":
            print("Skipping all.dpr.wikidata.jsonl")
            continue
        with open(input_file) as fi:
            data += [json.loads(line) for line in fi]

    with open(output_file, "w") as fo:
        for i, sample in enumerate(tqdm(data)):
            original_id = sample.get("id", "None")
            sample["id"] = str(i)
            sample["original_id"] = str(original_id)
            for positive in sample["positive_ctxs"]:
                if "wikidata" in positive["metadata"]:
                    positive["metadata"].pop("wikidata")
                if "lang" not in positive["metadata"]:
                    positive["metadata"]["lang"] = "xx"
                if "id" in positive["metadata"]:
                    print("id in positive")
                    positive["metadata"].pop("id")
                if "id" in positive:
                    positive.pop("id")
            fo.write(json.dumps(sample) + "\n")



def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_folder", type=str)
    arg_parser.add_argument("output_file", type=str)
    args = arg_parser.parse_args()

    parse_files(**vars(args))


if __name__ == "__main__":
    main()
