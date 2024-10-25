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

for lang in $LANGUAGES; do
    echo "Parsing data for language: $lang"
    # python -m mewsli.parse_documents --language $lang
    echo "Train"
    relik data convert-to-dpr \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.train.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows-dpr/$lang.train.jsonl \
        /media/data/relik/data/ml-el/documents/wikidata-mewsli-bela.all.pages.3M.jsonl \
        --label-type span

    echo "Val"
    relik data convert-to-dpr \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.val.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows-dpr/$lang.val.jsonl \
        /media/data/relik/data/ml-el/documents/wikidata-mewsli-bela.all.pages.3M.jsonl \
        --label-type span

    echo "Test"
    relik data convert-to-dpr \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.test.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows-dpr/$lang.test.jsonl \
        /media/data/relik/data/ml-el/documents/wikidata-mewsli-bela.all.pages.3M.jsonl \
        --label-type span

done
