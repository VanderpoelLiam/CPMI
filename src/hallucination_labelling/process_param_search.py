import subprocess
import generate_lambdas
import generate_thresholds
import argparse
import align_data
import entropy_stats
import numpy as np
import os
import os.path
from entropy_stats import get_hall, get_first_hall

def process_server(log_dir, lambdas, thresholds):
    split_files = "csplit -f " + log_dir + "xx " + log_dir + "param_search '/fairseq.tasks.text_to_speech/' '{*}'"
    create_batch_log = "mv " + log_dir + "xx00 " + log_dir + "batch.log"
    delete_param_search = "rm " + log_dir + "param_search"

    subprocess.call(split_files, shell=True)
    subprocess.call(create_batch_log, shell=True)

    i = 1
    for l in lambdas:
        for t in thresholds:
            log_filename = log_dir
            log_filename += "%0.4E" % l
            log_filename += "_%0.4E" % t
            create_param_log = "mv " + log_dir + "xx%02d " % i + log_filename
            subprocess.call(create_param_log, shell=True)
            i += 1
    subprocess.call(delete_param_search, shell=True)

def convert(raw_data, datatype):
    data = []
    for line in raw_data:
        l = list(map(datatype, line))
        data.append(l)
    return data

def process_local_ref(data_dir, log_dir, lambdas, thresholds):
    label_filename = data_dir + "test.label"
    labels = entropy_stats.read_lines(label_filename, int)

    for l in lambdas:
        for t in thresholds:
            suffix = "%0.4E_%0.4E" % (l, t)
            log_filename = log_dir + suffix
            try:
                delete_log = "rm " + log_filename
                save_file = "avg_probs_" + suffix
                probs = align_data.read_lines(log_filename, "P-")
                probs = convert(probs, float)
                hall, non_hall_probs = get_hall(labels, probs)
                first_hall = get_first_hall(labels, probs)
                probs = [item for sublist in probs for item in sublist]

                with open(log_dir + save_file, 'w') as f:
                    print(np.mean(probs), np.mean(hall), np.mean(first_hall), file=f)
                subprocess.call(delete_log, shell=True)
            except Exception as e:
                raise Error("Issue in process_local_ref")

def process_local_gen(log_dir, lambdas, thresholds, dataset):
    delete_target = "rm " + log_dir + "target.detok"
    delete_hypothesis = "rm " + log_dir + "hypothesis.detok"

    for l in lambdas:
        for t in thresholds:
            suffix = "%0.4E_%0.4E" % (l, t)
            log_filename = log_dir + suffix
            delete_log = "rm " + log_filename
            save_file = "score_" + suffix
            if dataset == "bart":
                subprocess.run(["src/compute_rouge.sh", "-b", log_dir, log_filename, save_file])
            else:
                subprocess.run(["src/compute_rouge.sh", log_dir, log_filename, save_file])
            subprocess.call(delete_target, shell=True)
            subprocess.call(delete_hypothesis, shell=True)
            subprocess.call(delete_log, shell=True)

def get_params(filename, split_on):
    res = filename.split(split_on)[1].split("_")
    return res[0], res[1]

def process_local(save_filename):
    delete_ref = "rm -r " + save_filename + "_ref/"
    delete_gen = "rm -r " + save_filename + "_gen/"
    with open(save_filename, 'w') as f:
        print(','.join(["lambda", "ent_threshold", "mean_probs", "mean_hall_probs", "mean_first_hall_probs", "R1", "R2", "RL"]), file=f)

    ref_log_dir = os.fsencode(save_filename + "_ref/")
    gen_log_dir = os.fsencode(save_filename + "_gen/")
    for file in os.listdir(ref_log_dir):
        try:
         filename = os.fsdecode(file)
         start_str = "avg_probs_"
         if filename.startswith(start_str):
            suffix = filename.split(start_str)[1]
            l, t = get_params(filename, start_str)
            with open(os.path.join(ref_log_dir, file)) as f:
                for line in f:
                    p1, p2, p3 = line.rstrip().split()
                    break

            with open(os.path.join(gen_log_dir, os.fsencode("score_" + suffix))) as f:
                scores = [float(line[22:28]) for line in f if "Average_F" in line]
                r1 = str(scores[0] * 100)
                r2 = str(scores[1] * 100)
                rl = str(scores[2] * 100)

            with open(save_filename, 'a') as f:
                print(','.join([l, t, p1, p2, p3, r1, r2, rl]), file=f)
        except Exception as e:
            raise Error("Issue in process_local")

    subprocess.call(delete_ref, shell=True)
    subprocess.call(delete_gen, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process results for parameter search given hallucination labels')

    parser.add_argument('--dataset', type=str, default="standard",
                        choices=["standard", "bart"],
                        help='Dataset model was trained on')

    parser.add_argument('--local', action='store_true',
                        help='Running locally. By default assume script is run on the server')

    args = parser.parse_args()

    base_log_dir = "logs/hallucination_labelling/"
    lambdas = generate_lambdas.main()
    thresholds = generate_thresholds.main()

    data_dir = "data/xsum-hallucination"
    if args.dataset == "bart":
        data_dir += "-bart"
    data_dir += "/"

    if args.local:
        filename = base_log_dir + args.dataset
        ref_log_dir = filename + "_ref/"
        gen_log_dir = filename + "_gen/"
        # process_local_ref(data_dir, ref_log_dir, lambdas, thresholds)
        process_local_gen(gen_log_dir, lambdas, thresholds, args.dataset)
        process_local(filename)
    else:
        for sent_type in ["ref", "gen"]:
            log_dir = base_log_dir + args.dataset + "_" + sent_type + "/"
            process_server(log_dir, lambdas, thresholds)
