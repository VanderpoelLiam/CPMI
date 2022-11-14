#!/bin/sh
SCORE_REFERENCE='false'

while getopts ':s' 'OPTKEY'; do
    case ${OPTKEY} in
        's')
            SCORE_REFERENCE='true'
            ;;
    esac
done

shift $((OPTIND-1))

DATASET=$1
DATA_DIR=$2
LAMBDA=$3
THRESHOLD=$4

if [ "$SCORE_REFERENCE" = true ]
then
  fairseq-generate \
  data/$DATA_DIR \
  --path checkpoints/summarization_model/$DATASET/checkpoint_best.pt \
  --batch-size 16 \
  --beam 5 \
  --truncate-source \
  --score-reference \
  --lm-path checkpoints/language_model/$DATASET/checkpoint_best.pt \
  --lm-weight -$LAMBDA \
  --ent-threshold $THRESHOLD
else
  fairseq-generate \
  data/$DATA_DIR \
  --path checkpoints/summarization_model/$DATASET/checkpoint_best.pt \
  --batch-size 4 \
  --beam 5 \
  --truncate-source \
  --lm-path checkpoints/language_model/$DATASET/checkpoint_best.pt \
  --lm-weight -$LAMBDA \
  --ent-threshold $THRESHOLD
fi
