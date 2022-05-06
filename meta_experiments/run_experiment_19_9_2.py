from datetime import datetime

from config import env_vars
from experiments.exp6_5_check_nero import run_experiment as run_experiment_6_5
from experiments.exp2_5_check_dire import run_experiment as run_experiment_2_5

import pandas as pd

functions = [run_experiment_6_5, run_experiment_2_5]
seeds = [52, 53, 54, 55, 56]
more_seeds = [62, 63, 64, 65, 66]
more_more_seeds = [72, 73, 74, 75, 76]
more_more_more_seeds = [82, 83, 84, 85, 86]

dfs = []
now = datetime.now()
date_time_now = dt_string = now.strftime("%d_%m-%H_%M")
save_location = f'meta_results/{__file__.split("/")[-1][:-3]}_{date_time_now}.xlsx'
for seed in more_seeds:
    for learning_rate in [0.005]:  # 0.01, 0.1 lr is not good for GAT + linear / linear - too noisy
        for linear_embedding_size in [30, 10]:
            for function in reversed(functions):
                returned_dictionary = function(seed=seed, embedding_size=linear_embedding_size,
                                               learning_rate=learning_rate)
                df = pd.DataFrame({k: [str(v)] for k, v in returned_dictionary.items()})
                dfs.append(df)
                final_df = pd.concat(dfs, axis=0)
                final_df.to_excel(save_location)
