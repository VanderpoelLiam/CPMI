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
  BPE_DIR=$DATA_DIR/bpe/bart
  mkdir $BPE_DIR -p
  cd $BPE_DIR
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
  cd -

  for SPLIT in train valid test
  do
    for LANG in source target
    do
      python -m examples.roberta.multiprocessing_bpe_encoder \
      --encoder-json "$BPE_DIR/encoder.json" \
      --vocab-bpe "$BPE_DIR/vocab.bpe" \
      --inputs "$OUT_DIR/$SPLIT.$LANG" \
      --outputs "$OUT_DIR/$SPLIT.bpe.$LANG" \
      --workers 60 \
      --keep-empty;
    done
  done
else
  if [ "$CNN_DATASET" = true ]
  then
    BPE_DIR=$DATA_DIR/bpe/cnn-dm
  else
    BPE_DIR=$DATA_DIR/bpe/xsum
  fi
  FAST=~/fastBPE/fast
  for SPLIT in train valid test
  do
    for LANG in source target
    do
      $FAST applybpe ${OUT_DIR}/${SPLIT}.bpe.${LANG} ${OUT_DIR}/${SPLIT}.${LANG} $BPE_DIR/codes
    done
  done
fi
