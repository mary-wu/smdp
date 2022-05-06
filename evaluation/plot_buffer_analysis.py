"""
Generate the grid-search plot for the different buffer compositions and algorithms.
"""

import os
import yaml
import pickle
from matplotlib import pyplot as plt
import statistics
import seaborn as sns

# Update folder paths if needed
MAX_SEEDS = 9
MODEL_PATH = "../models_paper/"
SUFFIX = "final"
GEN_PLOT = True
VERBOSE = False
OUTPUT_FILE = "../buffer_analysis_plots/buffer_grid_search.pdf"

plt.style.use("../utils/publication.mplstyle")


def get_models(buffer_size, rand_p, second_p, gen_plot=True, verbose=True, ax=None):

    if verbose:
        print(f"Looking for models with: buffer size {buffer_size}, rand prob {rand_p}, and second action prob {second_p}...")

    buffer_prefix = f"buf{buffer_size}_rand{rand_p}_second{second_p}_"
    model_folders = [x for x in os.listdir(MODEL_PATH) if buffer_prefix in x]

    if verbose:
        print(f"Found: {len(model_folders)} models with prefix: {buffer_prefix}")
    
    ddqn_models, dqn_models, bcq_models = [], [], []
    ddqn_seeds, dqn_seeds, bcq_seeds = set(), set(), set()
    for model in model_folders:
        cfg_file = os.path.join(MODEL_PATH, model, 'config.yaml')
        params = yaml.safe_load(open(cfg_file, 'r'))
        if params['network_size'] == 'dqn_fc':
            if params['ddqn'] or (params['ddqn'] == 1):
                if params['random_seed'] not in ddqn_seeds and len(ddqn_seeds) < MAX_SEEDS:
                    ddqn_models.append(model)
                    ddqn_seeds.add(params['random_seed'])
            else:
                if params['random_seed'] not in dqn_seeds and len(dqn_seeds) < MAX_SEEDS:
                    dqn_models.append(model)
                    dqn_seeds.add(params['random_seed'])
        else:
            if params['random_seed'] not in bcq_seeds and len(bcq_seeds) < MAX_SEEDS:
                bcq_models.append(model)
                bcq_seeds.add(params['random_seed'])

    # assert dqn_seeds == ddqn_seeds == bcq_seeds, f"Found multiple random seeds across models... DQN: {dqn_seeds}, DDQN: {ddqn_seeds}, BCQ: {bcq_seeds}"
    assert len(dqn_seeds) == MAX_SEEDS, f"Found fewer than {MAX_SEEDS} random seeds... DQN: {dqn_seeds}, DDQN: {ddqn_seeds}, BCQ: {bcq_seeds}"

    score_list, error_list = [], []
    model_paths = [ddqn_models, dqn_models, bcq_models]
    for model_list in model_paths:
        model_paths = [MODEL_PATH + x + f"/test_scores_{SUFFIX}" for x in model_list]
        model_score = []
        for path in model_paths:
            with open(path, "rb") as handle:
                test_scores = pickle.load(handle)
                model_score.append(sum(test_scores) / len(test_scores))

        score_list.append(sum(model_score) / len(model_paths))
        std_model_score = statistics.stdev(model_score)
        error_list.append(std_model_score)

    width = 0.15
    x = 1

    if gen_plot:
        if ax is None:
            fig = plt.figure()
            my_cmap = sns.light_palette("Paired", as_cmap=True)
            plt.set_cmap(my_cmap)
            ax = fig.gca()
        ax.bar(x - 0.2, score_list[1], width, yerr=error_list[1], capsize=2, label="SDQN", color="#a6cee3")
        ax.bar(x, score_list[0], width, yerr=error_list[0], capsize=2, label="SDDQN", color="#1f78b4")
        ax.bar(x + 0.2, score_list[2], width, yerr=error_list[2], capsize=2, label="SBCQ", color="#33a02c")
        ax.axhline(y=0, color='k', linestyle='-')
        ax.set_ylim([-3, 5])
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        if ax is None:
            fig.savefig(OUTPUT_FILE)
            plt.show()

    return score_list, error_list


if __name__ == '__main__':

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, sharey=True)

    i, j = 0, 0
    for buffer_size in [100, 1000, 10000]:
        scores, errors = [], []
        for second_p in [0.25]:
            for rand_p in [0.1, 0.25, 0.5]:
                try:
                    score_list, error_list = get_models(buffer_size=buffer_size, rand_p=rand_p, second_p=second_p,
                                                        gen_plot=GEN_PLOT, verbose=VERBOSE, ax=axes[i][j])
                except Exception as e:
                    print(f"Failed for model: {buffer_size}, {rand_p}, {second_p} with error: {e}")

                if i == 0:
                    axes[i][j].set_title(f" {second_p:,.0%} second-best \n {rand_p:,.0%} random ")
                if j == ncols - 1:
                    axes[i][j].yaxis.set_label_position("right")
                    axes[i][j].set_ylabel(f"Buffer:\n{buffer_size:,.0f}\nTransitions", rotation=0, labelpad=30)

                j += 1
                j = j % ncols
        i += 1

    plt.set_cmap("Paired")
    plt.subplots_adjust(bottom=0.03, wspace=0.2, right=0.8)
    plt.legend(bbox_to_anchor=(-0.6, -0.1), loc = 'upper center', ncol=3)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Average Episodic Return")

    fig.savefig(OUTPUT_FILE)
    plt.show()
