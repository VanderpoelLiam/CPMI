import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Generate fairseq-generate command')

parser.add_argument('exp_num', type=int,
                    help='Experiment number')

parser.add_argument('--wait_train', action='store_true',
                    help='To wait on fairseq-train to end')

# parser.add_argument('--gen_subset', type=str, default="test",
#                     choices=["train", "valid", "test"],
#                     help='Data subset to generate (train, valid, test)')

args = parser.parse_args()

batch_cmd = """\
bsub -J generate_%i \
-o logs/experiments/generate_%i.log \
-W 60 -n 4 -R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
""" % (args.exp_num, args.exp_num)

if args.wait_train:
    batch_cmd += """-w "ended(train_%i)" """ % args.exp_num

fairseq_cmd = """\
fairseq-generate \
data/xsum-summarizer \
--path checkpoints/summarization_model/%i/checkpoint_best.pt \
--batch-size 64 --beam 5 --truncate-source \
--skip-invalid-size-inputs-valid-test \
""" % args.exp_num

final_cmd = batch_cmd + fairseq_cmd

print("Done.")
pc.copy(final_cmd)
