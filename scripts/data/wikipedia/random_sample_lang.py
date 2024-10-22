import argparse
import json
from pathlib import Path
import random
from tqdm import tqdm


def parse_files(input_folder, output_folder, sample_per_file):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    counter = 0

    for input_file in tqdm(input_folder.iterdir()):
        with open(input_file) as fi:
            data = [json.loads(line) for line in fi]

        samples = random.sample(data, min(sample_per_file, len(data)))
        counter += len(samples)

        output_file = output_folder / input_file.name
        with open(output_file, "w") as fo:
            for sample in samples:
                if "id" not in sample:
                    print("id not in sample", input_file.name)
                    continue
                for positive in sample["positive_ctxs"]:
                    if "wikidata" in positive["metadata"]:
                        positive["metadata"].pop("wikidata")
                    if "lang" not in positive["metadata"]:
                        positive["metadata"]["lang"] = "xx"
                    if "id" in positive["metadata"]:
                        print("id in positive")
                        positive["metadata"].pop("id")
                json.dump(sample, fo)
                fo.write("\n")

    print(f"Total samples: {counter}")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_folder", type=str)
    arg_parser.add_argument("output_folder", type=str)
    arg_parser.add_argument("--sample-per-file", type=int, default=10_000)
    args = arg_parser.parse_args()

    parse_files(**vars(args))


if __name__ == "__main__":
    main()
