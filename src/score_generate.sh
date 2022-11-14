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

NAME=$1

EXPERIMENT_DIR=logs/experiments
ROUGE_DIR=logs/rouge
WORK_DIR=$ROUGE_DIR/$NAME

FILENAME=$EXPERIMENT_DIR/${NAME}

mkdir $WORK_DIR -p

if [ "$IS_BART" = true ]
then
  src/compute_rouge.sh -b $WORK_DIR $FILENAME score
else
  src/compute_rouge.sh $WORK_DIR $FILENAME score
fi
