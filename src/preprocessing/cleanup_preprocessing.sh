#!/bin/bash

OUT_DIR=$1

# Remove all the unnecessay preprocessing files in $OUT_DIR
rm $OUT_DIR/*.bpe.*
rm $OUT_DIR/*.source
rm $OUT_DIR/*.target
