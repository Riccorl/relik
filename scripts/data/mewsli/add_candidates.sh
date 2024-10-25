#!/bin/bash

LANGUAGES="ar de en es fa ja sr ta tr"

# setup conda
CONDA_BASE=$(conda info --base)
# check if conda is installed
if [ -z "$CONDA_BASE" ]; then
  echo "Conda is not installed. Please install conda first."
  exit 1
fi
source "$CONDA_BASE"/etc/profile.d/conda.sh

conda activate relik-thesis

ENCODER=/media/data/relik/data/ml-el/models/riccorl/me5-base-mewsli-all-from-wikipedia
INDEX=/media/data/relik/data/ml-el/index/me5-base-mewsli-all-from-wikipedia-3M-index

for lang in $LANGUAGES; do
    echo "Parsing data for language: $lang"
    # python -m mewsli.parse_documents --language $lang
    echo "Train"
    relik retriever add-candidates $ENCODER $INDEX \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.train.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows-candidates/$lang.train.jsonl \
        --index-device cuda \
        --precision 16 \
        --log-recall True

    echo "Val"
    relik retriever add-candidates $ENCODER $INDEX \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.val.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows-candidates/$lang.val.jsonl \
        --index-device cuda \
        --precision 16 \
        --log-recall True

    echo "Test"
    relik retriever add-candidates $ENCODER $INDEX \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.test.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows-candidates/$lang.test.jsonl \
        --index-device cuda \
        --precision 16 \
        --log-recall True

done
