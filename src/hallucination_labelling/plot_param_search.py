import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_z(x, y, data, name):
    df = data.loc[(data["lambda"] == x) & (data["ent_threshold"] == y)]
    return df.iloc[0][name]

def get_Z(X, Y, data, name):
    N, M = X.shape
    Z = np.empty_like(X)
    for i in range(N):
        for j in range(M):
            Z[i, j] = get_z(X[i, j], Y[i, j], data, name)
    return Z

def plot(data, cmap, title, name, dataset, show_fig = True, save_fig = False):
    x = np.unique(data["lambda"])
    y = np.unique(data["ent_threshold"])
    X, Y = np.meshgrid(x, y)
    Z = get_Z(X, Y, data, name)
    fig = plt.figure()
    ax = plt.axes()
    plt.contourf(X, Y, Z, 20, cmap=cmap)
    plt.colorbar()
    ax.set_xlabel(r'$\mathregular{\lambda}$', fontsize=15, labelpad=10)
    ax.set_ylabel(r'$\mathregular{\tau}$', rotation=0, fontsize=15, labelpad=20)
    plt.title(title)
    if show_fig:
        plt.show()

    if save_fig:
        filename = "_".join(["param_search", dataset, name])
        path = "/home/liam/Dropbox/ETH/Courses/Research/Thesis/figs/tmp/"
        plt.savefig(path + filename + ".png", dpi=150, bbox_inches = "tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot parameter search results')

    parser.add_argument('--dataset', type=str, default="standard",
                        choices=["standard", "bart"],
                        help='Dataset model was trained on')

    args = parser.parse_args()

    data_dir = "logs/hallucination_labelling/" + args.dataset
    data = pd.read_csv(data_dir)

    w = 1
    if (args.dataset == "bart"):
        w = 0.6

    data['combination'] = w * data['RL'] - data['mean_first_hall_probs']

    title = r'ROUGE-L $\mathregular{F_1}$ score'
    name =  "RL"
    plot(data, 'Reds', title, name, args.dataset, False, True)

    title = r'Average initial hallucination token log probabilities'
    name =  "mean_first_hall_probs"
    plot(data, 'Greens', title, name, args.dataset, False, True)

    title = r'Weighted Combination Rouge Score & Average log probabilities'
    name =  "combination"
    plot(data, 'Reds', title, name, args.dataset, False, True)

    data = data.sort_values(by=['combination'], ascending=False)
    print("Optimal parameters sorted by combination:")
    print(data[0:5])
