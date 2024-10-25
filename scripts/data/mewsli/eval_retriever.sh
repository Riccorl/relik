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

ENCODER=$1

if [ -z "$ENCODER" ]; then
    echo "Please provide the encoder name or path"
    exit 1
fi

INDEX=$2

if [ -z "$INDEX" ]; then
    echo "Please provide the index name or path"
    exit 1
fi

# SOURCE_DIR=/media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-windows
# SOURCE_DIR=/media/data/relik/data/ml-el/data/mewsli-9-relik-entities-windows
SOURCE_DIR=/media/data/relik/data/ml-el/data/mewsli-9-relik-entities-pages-specialized-tokenizer-windows

echo "Eval Test"
TEST_PATHS=(
   $SOURCE_DIR/ar.test.jsonl
   $SOURCE_DIR/de.test.jsonl
   $SOURCE_DIR/en.test.jsonl
   $SOURCE_DIR/es.test.jsonl
   $SOURCE_DIR/fa.test.jsonl
   $SOURCE_DIR/ja.test.jsonl
   $SOURCE_DIR/sr.test.jsonl
   $SOURCE_DIR/ta.test.jsonl
   $SOURCE_DIR/tr.test.jsonl
)
python scripts/data/retriever/compute_metrics.py \
    --question-encoder-name-or-path $ENCODER \
    --document-name-or-path $INDEX \
    --input-paths "${TEST_PATHS[@]}" \
    --index-device cuda \
    --precision 16

echo "Eval Val"
VAL_PATHS=(
   $SOURCE_DIR/ar.val.jsonl
   $SOURCE_DIR/de.val.jsonl
   $SOURCE_DIR/en.val.jsonl
   $SOURCE_DIR/es.val.jsonl
   $SOURCE_DIR/fa.val.jsonl
   $SOURCE_DIR/ja.val.jsonl
   $SOURCE_DIR/sr.val.jsonl
   $SOURCE_DIR/ta.val.jsonl
   $SOURCE_DIR/tr.val.jsonl
)
python scripts/data/retriever/compute_metrics.py \
    --question-encoder-name-or-path $ENCODER \
    --document-name-or-path $INDEX \
    --input-paths "${VAL_PATHS[@]}" \
    --index-device cuda \
    --precision 16
