#!/bin/bash
IS_BART='false'

while getopts ':b' 'OPTKEY'; do
    case ${OPTKEY} in
        'b')
            IS_BART='true'
            ;;
    esac
done

shift $((OPTIND-1))

LOG_DIR=logs/experiments
DATA_DIR=data/xsum-detect-hall

SCRIPTS=~/mosesdecoder/scripts

FILENAME=$LOG_DIR/$1
SUFFIX=$2

PROCESSED_FILENAME=$DATA_DIR/hypo_processed_$SUFFIX
OUT_FILENAME=$DATA_DIR/hypo_$SUFFIX

if [ "$IS_BART" = true ]
then
  TMP=$DATA_DIR/tmp
  CLEAN=$SCRIPTS/training/clean-corpus-n.perl
  grep ^H- $FILENAME | cut -f3- > $PROCESSED_FILENAME
  python src/bart_file_bpe_decoder.py $PROCESSED_FILENAME $TMP.en
  perl $CLEAN -ratio 1000000 $TMP en en $OUT_FILENAME 1 1000000
  rm $TMP.en
  mv $OUT_FILENAME.en $OUT_FILENAME
else
  DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
  grep ^H- $FILENAME | cut -f3- > $PROCESSED_FILENAME
  sed 's/@@ //g' $PROCESSED_FILENAME | perl $DETOKENIZER -threads 8 -q -l en > $OUT_FILENAME
fi
