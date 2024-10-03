#!/bin/bash

LANGUAGES="ar de en es fa ja sr ta tr"

source /leonardo_work/IscrC_MEL/python-envs/relik-thesis-venv/bin/activate

for lang in $LANGUAGES; do
    echo "Parsing documents for language: $lang"
    # python -m mewsli.parse_documents --language $lang
    python scripts/data/mewsli/parse_documents.py \
        /leonardo_work/IscrC_MEL/relik-ml/data/thesis/ml-el/documents/wikidata-mewsli-bela.raw.complete.jsonl \
        /leonardo_work/IscrC_MEL/relik-ml/data/thesis/ml-el/documents/wikidata-mewsli-bela.$lang.jsonl \
        --language $lang \
        --force-ids-path /leonardo_work/IscrC_MEL/relik-ml/data/thesis/ml-el/documents/mewsli-labels-id/$lang.txt
done