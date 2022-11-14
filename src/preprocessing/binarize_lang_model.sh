#!/bin/sh

BART_MODEL='false'
CNN_DATASET='false'

while getopts "bc" arg; do
  case $arg in
    b)
      BART_MODEL='true'
      ;;
    c)
      CNN_DATASET='true'
      ;;
  esac
done

shift $((OPTIND-1))

OUT_DIR=$1
DATA_DIR=$2
LANG_DIR=$3

if [ "$BART_MODEL" = true ]
then
  DICT="$DATA_DIR/bpe/bart/dict.txt"
elif [ "$CNN_DATASET" = true ]
then
  DICT="$DATA_DIR/bpe/cnn-dm/dict.txt"
else
  DICT="$DATA_DIR/bpe/xsum/dict.txt"
fi

for SPLIT in train test valid
do
  cat "$OUT_DIR/$SPLIT.bpe.target" "$OUT_DIR/$SPLIT.bpe.source" > "$OUT_DIR/$SPLIT.bpe.full"
done

fairseq-preprocess \
    --only-source \
    --trainpref "$OUT_DIR/train.bpe.target" \
    --validpref "$OUT_DIR/valid.bpe.target" \
    --testpref "$OUT_DIR/test.bpe.target" \
    --destdir "$LANG_DIR" \
    --workers 60 \
    --srcdict $DICT

fairseq-preprocess \
    --only-source \
    --trainpref "$OUT_DIR/train.bpe.full" \
    --validpref "$OUT_DIR/valid.bpe.full" \
    --testpref "$OUT_DIR/test.bpe.full" \
    --destdir "$LANG_DIR-full" \
    --workers 60 \
    --srcdict $DICT

cp $DICT $OUT_DIR/dict.txt
