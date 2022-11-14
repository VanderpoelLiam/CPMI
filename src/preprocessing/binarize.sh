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

if [ "$BART_MODEL" = true ]
then
  DICT="$DATA_DIR/bpe/bart/dict.txt"
elif [ "$CNN_DATASET" = true ]
then
  DICT="$DATA_DIR/bpe/cnn-dm/dict.txt"
else
  DICT="$DATA_DIR/bpe/xsum/dict.txt"
fi

fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "${OUT_DIR}/train.bpe" \
    --validpref "${OUT_DIR}/valid.bpe" \
    --testpref "${OUT_DIR}/test.bpe" \
    --destdir "${OUT_DIR}" \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 60

cp $DICT "$OUT_DIR/dict.txt"
