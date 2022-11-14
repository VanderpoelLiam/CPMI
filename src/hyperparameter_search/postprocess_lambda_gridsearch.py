import sys
import subprocess
import generate_lambdas

log_dir = sys.argv[1]
lambdas = generate_lambdas.main()
src_dir = "src/"

split_files = "csplit -f " + log_dir + "xx " + log_dir + "lambda_search.log '/fairseq.tasks.text_to_speech/' '{*}'"
create_batch_log = "mv " + log_dir + "xx00 " + log_dir + "batch.log"
delete_lambda_search = "rm " + log_dir + "lambda_search.log"

subprocess.call(split_files, shell=True)
subprocess.call(create_batch_log, shell=True)

delete_target = "rm " + log_dir + "target.detok"
delete_hypothesis = "rm " + log_dir + "hypothesis.detok"

for i, lamb in enumerate(lambdas):
    filename = log_dir + "%0.4E.log" % lamb
    create_lamb_log = "mv " + log_dir + "xx%02d " % (i+1) + filename

    subprocess.call(create_lamb_log, shell=True)
    print("Lambda = %0.4E" % lamb)
    save_file = "score_%0.4E" % lamb
    subprocess.run([src_dir + "compute_rouge.sh", log_dir, filename, save_file])
    subprocess.call(delete_target, shell=True)
    subprocess.call(delete_hypothesis, shell=True)

subprocess.call(delete_lambda_search, shell=True)
