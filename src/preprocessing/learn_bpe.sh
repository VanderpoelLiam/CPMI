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
FAST=~/fastBPE/fast

if [ "$CNN_DATASET" = true ]
then
  BPE_DIR=$DATA_DIR/bpe/cnn-dm
else
  BPE_DIR=$DATA_DIR/bpe/xsum
fi

# Learn codes
$FAST learnbpe 50000 ${OUT_DIR}/train.source ${OUT_DIR}/train.target > $BPE_DIR/codes

# Apply codes to train
SPLIT=train
for LANG in source target
  do
    $FAST applybpe ${OUT_DIR}/${SPLIT}.bpe.${LANG} ${OUT_DIR}/${SPLIT}.${LANG} $BPE_DIR/codes
  done

# Get train vocabulary
$FAST getvocab ${OUT_DIR}/train.source.bpe ${OUT_DIR}/train.target.bpe  > $BPE_DIR/vocab.bpe

# Apply codes to valid and test
for SPLIT in valid test
do
  for LANG in source target
  do
    $FAST applybpe ${OUT_DIR}/${SPLIT}.bpe.${LANG} ${OUT_DIR}/${SPLIT}.${LANG} $BPE_DIR/codes
  done
done
