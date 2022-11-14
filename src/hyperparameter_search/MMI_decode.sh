#!/bin/sh
SUM_MODEL=$1
LANG_MODEL=$2
LAMBDA=$3
GEN_SUBSET=$4

fairseq-generate \
data/xsum-summarizer \
--gen-subset $GEN_SUBSET \
--path checkpoints/summarization_model/$SUM_MODEL/checkpoint_best.pt \
--batch-size 16 --beam 5 --truncate-source \
--skip-invalid-size-inputs-valid-test \
--lm-path checkpoints/$LANG_MODEL/checkpoint_best.pt \
--lm-weight -$LAMBDA
