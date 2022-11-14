#!/bin/bash
CNN_DATASET='false'

while getopts "c" arg; do
  case $arg in
    c)
      CNN_DATASET='true'
      ;;
  esac
done

shift $((OPTIND-1))

OUT_DIR=$1
DATA_DIR=$2

for SPLIT in train test valid
do
  cat "$OUT_DIR/$SPLIT.bpe.target" "$OUT_DIR/$SPLIT.bpe.source" > "$OUT_DIR/$SPLIT.bpe.full"
done

fairseq-preprocess \
    --source-lang "full" \
    --trainpref "${OUT_DIR}/train.bpe" \
    --destdir "${OUT_DIR}" \
    --only-source \
    --dict-only \
    --workers 60

if [ "$CNN_DATASET" = true ]
then
  BPE_DIR=$DATA_DIR/bpe/cnn-dm
else
  BPE_DIR=$DATA_DIR/bpe/xsum
fi
mv ${OUT_DIR}/dict.full.txt  $BPE_DIR/dict.txt
