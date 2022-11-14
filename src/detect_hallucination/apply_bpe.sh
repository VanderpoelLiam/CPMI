#!/bin/bash

BART_MODEL='false'

while getopts ':b' 'OPTKEY'; do
    case ${OPTKEY} in
        'b')
            BART_MODEL='true'
            ;;
    esac
done

shift $((OPTIND-1))

DATA_DIR=data
OUT_DIR=$2

if [ "$BART_MODEL" = true ]
then
  {
    bpe=$(python src/detect_hallucination/bart_bpe_encoder.py $1)
  } &> /dev/null
else
  {
    FAST=../fastBPE/fast
    bpe=$(echo $1 | $FAST applybpe_stream ${DATA_DIR}/codes)
  } &> /dev/null
fi

echo $bpe
