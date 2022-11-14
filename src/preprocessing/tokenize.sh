#!/bin/bash

SCRIPTS=~/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

in_dir=$1
src=$2
tgt=$3

echo "pre-processing train data..."

for l in $src $tgt; do
  cat $in_dir/$l.txt | \
  perl $NORM_PUNC en | \
  perl $TOKENIZER -threads 8 -l en > $in_dir/tok.$l.en
  echo ""
done

echo "removing raw text files..."
for l in $src $tgt; do
  rm $in_dir/$l.txt
  echo ""
done


for l in $src $tgt; do
  perl $CLEAN -ratio 1000000 $in_dir/tok.$l en en $in_dir/clean.$l 1 1000000
  mv $in_dir/clean.${l}.en $in_dir/$l
  echo ""
done

echo "removing tokenized text files..."
for l in $src $tgt; do
  rm $in_dir/tok.$l.en
  echo ""
done
