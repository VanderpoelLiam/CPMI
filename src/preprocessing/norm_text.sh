#!/bin/bash

SCRIPTS=~/mosesdecoder/scripts
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

in_dir=$1
src=$2

echo "normalizing text data..."

cat $in_dir/${src}.txt | \
perl $NORM_PUNC en > $in_dir/norm.$src.en
perl $CLEAN -ratio 1000000 $in_dir/norm.$src en en $in_dir/clean.$src 1 1000000
echo ""


echo "removing raw text files..."
rm $in_dir/${src}.txt
rm $in_dir/norm.${src}.en
mv $in_dir/clean.${src}.en $in_dir/source
