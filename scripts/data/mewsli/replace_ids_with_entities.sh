#!/bin/bash

LANGUAGES="ar de en es fa ja sr ta tr"

# source /root/miniconda3/envs/relik-thesis/bin/python

for lang in $LANGUAGES; do
    echo "Parsing data for language: $lang"
    # python -m mewsli.parse_documents --language $lang
    echo "Train"
    python scripts/data/mewsli/repalce_ids_with_entities.py \
        /media/data/relik/data/ml-el/data/mewsli-9-relik/$lang.train.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages/$lang.train.jsonl \
        /media/data/relik/data/ml-el/documents/wikidata-mewsli-bela.all.pages.3M.jsonl

    echo "Val"
    python scripts/data/mewsli/repalce_ids_with_entities.py \
        /media/data/relik/data/ml-el/data/mewsli-9-relik/$lang.val.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages/$lang.val.jsonl \
        /media/data/relik/data/ml-el/documents/wikidata-mewsli-bela.all.pages.3M.jsonl
    
    echo "Test"
    python scripts/data/mewsli/repalce_ids_with_entities.py \
        /media/data/relik/data/ml-el/data/mewsli-9-relik/$lang.test.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages/$lang.test.jsonl \
        /media/data/relik/data/ml-el/documents/wikidata-mewsli-bela.all.pages.3M.jsonl
done
