import argparse
import json
from pathlib import Path
import random
import sys

from tqdm import tqdm


def parse_documents(
    input_file: str,
    output_file: str,
    force_ids_path: str = None,
    prefer_pages: bool = False,
):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ids = set()
    if force_ids_path:
        with open(force_ids_path) as fi:
            for line in fi:
                ids.add(line.strip())

    total = 0
    skipped = 0
    random_language_count = 0
    random_description_count = 0
    random_title_count = 0
    totally_missing = 0
    empty_titles = 0

    with open(input_file) as fi, open(output_file, "w") as fo:
        for i, line in enumerate(tqdm(fi)):
            total += 1
            doc = json.loads(line)
            metadata = {
                "wikidata_id": doc["id"],
            }
            titles = doc["labels"]
            if len(titles) == 0:
                if doc["id"] in ids:
                    totally_missing += 1
                    continue
                empty_titles += 1
                continue

            chosen_title_is_en = False
            if "en" in titles:
                chosen_title_is_en = True
                chosen_title = titles["en"]
            else:
                chosen_title = random.choice(list(titles.values()))
                random_title_count += 1
            metadata["title"] = chosen_title

            descriptions = doc["descriptions"]
            if "en" in descriptions:
                chosen_description = descriptions["en"]
            else:
                if len(descriptions) != 0:
                    chosen_description = random.choice(list(descriptions.values()))
                else:
                    chosen_description = ""
            metadata["description"] = chosen_description

            aliases = doc["aliases"]
            if "en" in aliases:
                chosen_aliases = aliases["en"]
            else:
                chosen_aliases = []
            metadata["aliases"] = chosen_aliases

            pages = doc["pages"]
            if "en" in pages:
                chosen_page = pages["en"]
            else:
                # if chosen_title_is_en:
                #     chosen_page = chosen_title
                if len(pages) != 0:
                    chosen_page = random.choice(list(pages.values()))
                else:
                    chosen_page = ""
            metadata["page"] = chosen_page

            if prefer_pages and chosen_page:
                chosen_title = chosen_page

            parsed_doc = {"id": i, "text": chosen_title, "metadata": metadata}

            fo.write(json.dumps(parsed_doc) + "\n")

    print(f"Skipped {skipped} out of {total} documents.")
    print(f"Random language count: {random_language_count}")
    print(f"Random description count: {random_description_count}")
    print(f"Random title count: {random_title_count}")
    print(f"Totally missing: {totally_missing}")


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("--force-ids-path", type=str, default=None)
    arg_parser.add_argument("--prefer-pages", action="store_true")
    args = arg_parser.parse_args()

    parse_documents(**vars(args))


if __name__ == "__main__":
    main()
