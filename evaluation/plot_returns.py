import os
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from environment import *

FOLDER_LOCATION = "/../models/"
MODELS = ["online"]
RUNS = [range(6,12), range(12,18), range(6)]
MAVG = True
SCORE_SUFFIX = ""

OUTPUT_FILE = "learning_curves"

plt.style.use('../utils/publication.mplstyle')

prev_model = None


def run():

    fig, axes = plt.subplots(1, 3)

    for model_type in MODELS:

        for j in range(len(RUNS)):

            df = pd.DataFrame({})

            for i in RUNS[j]:

                model_name = model_type + str(i)
                folder_path = (os.getcwd() + FOLDER_LOCATION + model_name).replace("/", "\\")
                if not os.path.exists(folder_path):
                    print(f"Could not find folder path: {folder_path}")
                    continue

                cfg_file = os.path.join(folder_path, 'config.yaml')
                params = yaml.safe_load(open(cfg_file, 'r'))

                try:
                    score = pd.read_csv(folder_path + f"/scores{SCORE_SUFFIX}.csv")['scores'].values
                except FileNotFoundError:
                    print(f"Could not find sores/steps file. Skipping folder: {folder_path}")
                    continue
                except KeyError:
                    print(f"Empty data. Skipping folder: {folder_path}")
                    continue

                x = np.arange(0, len(score)) * params['episodes_per_epoch']
                model_name = "BCQ" if params['network_size'] == 'bcq_fc' else ("DDQN" if params['ddqn'] else 'DQN')
                label_name = "SMDP Variation" if params['sq_learning'] else "MDP Variation"
                temp_df = pd.DataFrame({"Model": label_name, "Epoch": x, "Returns": score, "Random Seed": params['random_seed']})
                temp_df["Returns_Mavg"] = temp_df["Returns"].rolling(5).mean().fillna(temp_df["Returns"])
                df = pd.concat([df, temp_df])

            xlabel = "Training Episodes"
            df = df.sort_values(by=["Model", "Epoch"]).reset_index()
            print(f"\nFor {model_name}:")
            # print(f"\tRandom seeds: {df['Random Seed'].unique()}")
            # print(f"\tLast rewards: {df['Returns'].unique()}")
            print(df.groupby('Model')['Random Seed'].unique())
            print(df[df['Epoch'] == df['Epoch'].max()].round(2).groupby('Model')['Returns'].apply(list))

            if MAVG:
                sns.lineplot(data=df, x="Epoch", y="Returns_Mavg", hue="Model", ax=axes[j])
            else:
                sns.lineplot(data=df, x="Epoch", y="Returns", hue="Model", ax=axes[j])

            if j == 0:
                pparam = dict(xlabel=xlabel,
                              ylabel='Episodic Return',
                              title="(S)" + model_name
                              )
            else:
                pparam = dict(xlabel=xlabel, ylabel="", title="(S)" + model_name
                              )

            axes[j].autoscale(tight=True)
            axes[j].set(**pparam)

            axes[j].legend(title="")
            handles, labels = axes[j].get_legend_handles_labels()
            axes[j].legend(handles=handles[1:], labels=labels[1:])
            axes[j].set_ylim(-0.5, 2.5)

        if MAVG:
            fig.savefig(f'.//{OUTPUT_FILE}_mavg.pdf')
        else:
            fig.savefig(f'./{OUTPUT_FILE}.pdf')

        plt.show()


if __name__ == '__main__':
    run()
