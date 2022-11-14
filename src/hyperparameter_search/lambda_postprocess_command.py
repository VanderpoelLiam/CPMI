import argparse
import pyperclip as pc

parser = argparse.ArgumentParser(description='Postprocess log from gridsearch over lambda for MMI decoding command')

parser.add_argument('summarization_model', type=int,
                    help='Summarization model number')

parser.add_argument('--lang_full', action='store_true',
                    help='To use the full language model')

args = parser.parse_args()

src_dir = "src/hyperparameter_search/"

if args.lang_full:
    language_model = "lang_full"
else:
    language_model = "lang"

log_dir = "logs/hyperparameters/" + language_model + "/%i/" % args.summarization_model


if args.lang_full:
    batch_cmd = "bsub -J postprocess_lambda_%i_full" % args.summarization_model
    batch_cmd += """ -w "ended(lambda_search_%i_full)"\
""" % args.summarization_model
else:
    batch_cmd = "bsub -J postprocess_lambda_%i" % args.summarization_model
    batch_cmd += """ -w "ended(lambda_search_%i)"\
""" % args.summarization_model

batch_cmd += " -o " + log_dir + "batch.log"
batch_cmd += """\
 -W 60 -n 4 -R "rusage[mem=2048]" \
"""

final_cmd = batch_cmd
final_cmd += "python " + src_dir + "postprocess_lambda_gridsearch.py " + log_dir

print("Done.")
pc.copy(final_cmd)
