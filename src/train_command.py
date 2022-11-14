import argparse
import pyperclip as pc
import re

parser = argparse.ArgumentParser(description='Generate fairseq-train command')

parser.add_argument('exp_num', type=int,
                    help='Experiment number')

parser.add_argument('--arch', type=str, default="transformer_iwslt_de_en",
                    help='Model architecture to use')

parser.add_argument('--update_freq', type=int, default="64",
                    help='Frequency of parameter updates')

parser.add_argument('--dropout', type=float,
                    help='Frequency of parameter updates')

parser.add_argument('--restore_num', type=int,
                    help='Optional checkpoint to restore model from')

args = parser.parse_args()

batch_cmd = """\
bsub -J train_%i \
-o logs/experiments/train_%i.log \
-W 1200 -n 4 \
-R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
""" % (args.exp_num, args.exp_num)

fairseq_cmd = """\
fairseq-train \
data/xsum-summarizer \
--save-dir checkpoints/summarization_model/%i \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4096 \
--update-freq %i \
--no-epoch-checkpoints --no-last-checkpoints \
--truncate-source \
--skip-invalid-size-inputs-valid-test \
--patience 5 \
""" % (args.exp_num, args.update_freq)

arch_cmd = """\
--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--dropout 0.3 --weight-decay 0.0001 \
"""

if args.arch == "transformer_wmt_en_de":
    arch_cmd = """\
--arch transformer_wmt_en_de --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--lr 0.00007 \
--dropout 0.1 --weight-decay 0.0 \
    """

fairseq_cmd += arch_cmd

if not (args.restore_num is None):
    restore_file = """\
--restore-file checkpoints/summarization_model/%i/checkpoint_best.pt \
    """ % (args.restore_num)
    fairseq_cmd += restore_file

if not (args.dropout is None):
    fairseq_cmd = re.sub('--dropout \d.\d*', '--dropout %.4f', fairseq_cmd) % args.dropout

final_cmd = batch_cmd + fairseq_cmd

print("Done.")
pc.copy(final_cmd)
