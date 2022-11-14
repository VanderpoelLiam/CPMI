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

WORK_DIR=$1
FILENAME=$2
SAVE_FILE=$3

grep ^T- $FILENAME | cut -f2- > $WORK_DIR/target.raw
grep ^H- $FILENAME | cut -f3- > $WORK_DIR/hypothesis.raw

if [ "$IS_BART" = true ]
then
  for l in target hypothesis; do
    {
      python src/bart_file_bpe_decoder.py $WORK_DIR/${l}.raw $WORK_DIR/${l}.detok
    } &> /dev/null
    rm $WORK_DIR/${l}.raw
  done
else
  SCRIPTS=~/mosesdecoder/scripts
  DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl

  for l in target hypothesis; do
    sed 's/@@ //g' $WORK_DIR/${l}.raw > $WORK_DIR/${l}.debpe
    cat $WORK_DIR/${l}.debpe |
    perl $DETOKENIZER -threads 8 -q -l en > $WORK_DIR/${l}.detok
    rm $WORK_DIR/${l}.raw
    rm $WORK_DIR/${l}.debpe
  done
fi

echo "Running files2rouge ..."
files2rouge $WORK_DIR/target.detok $WORK_DIR/hypothesis.detok > $WORK_DIR/$SAVE_FILE

echo "ROUGE scores: "
python src/extract_score.py $WORK_DIR/$SAVE_FILE
