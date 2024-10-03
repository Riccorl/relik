import argparse
import json
from pathlib import Path
import random
import sys

from tqdm import tqdm


def parse_documents(
    input_file: str,
    output_file: str,
    number: int,
    force_ids_path: str = None,
):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ids = set()
    if force_ids_path:
        with open(force_ids_path) as fi:
            for line in fi:
                ids.add(line.strip())

    data = {}
    with open(input_file) as fi:
        for i, line in enumerate(tqdm(fi)):
            doc = json.loads(line)
            data[doc["metadata"]["wikidata_id"]] = doc

    # pick random samples from the data to reach the desired number
    # random_samples = random.sample(list(data.keys()), number)
    random.seed(69)
    print("Sampling random ids")
    keys = list(data.keys())
    random_samples = set(random.choices(keys, k=number))
    print("Number of random samples:", len(random_samples))

    # now check if all ids are in the random samples
    print("Checking if all ids are in the random samples")
    for w_id in ids:
        if w_id not in random_samples:
            random_samples.add(w_id)

    print("Number of random samples after adding ids:", len(random_samples))
    missing = 0
    print("Writing random samples to file")
    with open(output_file, "w") as fo:
        for random_id in random_samples:
            if random_id in data:
                fo.write(json.dumps(data[random_id]) + "\n")
            else:
                missing += 1

    print("Number of missing ids:", missing)


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("number", type=int)
    arg_parser.add_argument("--force-ids-path", type=str, default=None)
    args = arg_parser.parse_args()

    parse_documents(**vars(args))


if __name__ == "__main__":
    main()
