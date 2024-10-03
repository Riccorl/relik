import argparse
import json
from pathlib import Path
import random
import sys

from tqdm import tqdm


def parse_documents(
    input_file: str, output_file: str, language: str, force_ids_path: str = None
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


    is_all_lang = bool(language == "all")
    if language == "all":
        language = "en"

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
                    # print(f"Document {i} has no titles but is in the ids list.")
                    # print(doc)
                    # sys.exit(1)
                    totally_missing += 1
                    continue
                # print(f"Document {i} has no titles.")
                # print(doc)
                # sys.exit(1)
                skipped += 1
                continue
            override_language = language

            if language in titles:
                chosen_title = titles[language]
            elif doc["id"] in ids:
                if "en" in titles:
                    chosen_title = titles["en"]
                    override_language = "en"
                else:
                    # pick a random language from the titles
                    random_language = random.choice(list(titles.keys()))
                    chosen_title = titles[random_language]
                    random_language_count += 1
                    override_language = random_language
            else:
                if is_all_lang:
                    chosen_title =  random.choice(list(titles.values()))
                    random_title_count += 1
                else:
                    skipped += 1
                    continue
            
            
            # descriptions = doc["descriptions"]
            # if override_language in descriptions:
            #     chosen_description = descriptions[override_language]
            # else:
            #     chosen_description = ""
            # metadata["description"] = chosen_description

            descriptions = doc["descriptions"]
            if language in descriptions:
                chosen_description = descriptions[language]
            elif doc["id"] in ids:
                if len(descriptions) == 0:
                    chosen_description = ""
                elif "en" in descriptions:
                    chosen_description = descriptions["en"]
                    # override_language = "en"
                else:
                    # pick a random language from the descriptions
                    random_language = random.choice(list(descriptions.keys()))
                    chosen_description = descriptions[random_language]
                    random_description_count += 1
                    # override_language = random_language
            else:
                if is_all_lang:
                    if len(descriptions) != 0:
                        chosen_description =  random.choice(list(descriptions.values()))
                    else:
                        chosen_description = ""
                else:
                    chosen_description = ""
            metadata["description"] = chosen_description


            aliases = doc["aliases"]
            if override_language in aliases:
                chosen_aliases = aliases[override_language]
            else:
                chosen_aliases = []
            metadata["aliases"] = chosen_aliases

            pages = doc["pages"]
            if override_language in pages:
                chosen_page = pages[override_language]
            else:
                chosen_page = ""
            metadata["page"] = chosen_page

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
    arg_parser.add_argument("--language", type=str)
    arg_parser.add_argument("--force-ids-path", type=str, default=None)
    args = arg_parser.parse_args()

    parse_documents(**vars(args))


if __name__ == "__main__":
    main()
