#!/bin/bash

LANGUAGES="ar de en es fa ja sr ta tr"

# source /root/miniconda3/envs/relik-thesis

for lang in $LANGUAGES; do
    # Set tokenizer language based on the lang code, otherwise use 'xx'
    if [[ $lang == "ca" || $lang == "da" || $lang == "de" || $lang == "el" || $lang == "en" || $lang == "es" || $lang == "fr" || $lang == "it" || $lang == "ja" || $lang == "lt" || $lang == "mk" || $lang == "nb" || $lang == "nl" || $lang == "pl" || $lang == "pt" || $lang == "ro" || $lang == "ru" || $lang == "xx" || $lang == "zh" ]]
    then
        tokenizer_lang="$lang"
    else
        tokenizer_lang="xx"
    fi

    echo "Parsing data for language: $lang"
    # python -m mewsli.parse_documents --language $lang
    echo "Train"
    relik data create-windows \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages/$lang.train.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.train.jsonl \
        --language "$tokenizer_lang" \
        --window-size 32 \
        --window-stride 16 \
        --tokenizer-device cpu

    echo "Val"
    relik data create-windows \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages/$lang.val.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.val.jsonl \
        --language "$tokenizer_lang" \
        --window-size 32 \
        --window-stride 16 \
        --tokenizer-device cpu

    echo "Test"
    relik data create-windows \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages/$lang.test.jsonl \
        /media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows/$lang.test.jsonl \
        --language "$tokenizer_lang" \
        --window-size 32 \
        --window-stride 16 \
        --tokenizer-device cpu

done
