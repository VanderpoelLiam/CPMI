import subprocess
import sys
import generate_lambdas

sum_model = sys.argv[1]
lang_model = sys.argv[2]
log_dir = sys.argv[3]
lambdas = generate_lambdas.main()
src_dir = "src/hyperparameter_search/"

subprocess.call("mkdir -p " + log_dir, shell=True)

for lamb in lambdas:
    subprocess.run([src_dir + "MMI_decode.sh", sum_model, lang_model, str(lamb), "valid"])
