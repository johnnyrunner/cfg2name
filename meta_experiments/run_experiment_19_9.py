from datetime import datetime

from config import env_vars
from experiments.exp6_8_check_nero_gat import run_experiment as run_experiment_6_8
from experiments.exp6_3_check_nero_triple_linear import run_experiment as run_experiment_6_3
from experiments.exp3_7_check_dire_graph_on_top import run_experiment as run_experiment_3_7
from experiments.exp2_7_check_dire_triple_linear import run_experiment as run_experiment_2_7

import pandas as pd

functions = [run_experiment_6_8, run_experiment_6_3, run_experiment_3_7, run_experiment_2_7]
seeds = [52, 53, 54, 55, 56]
more_seeds = [62, 63, 64, 65, 66]
more_more_seeds = [72, 73, 74, 75, 76]

dfs = []
now = datetime.now()
date_time_now = dt_string = now.strftime("%d_%m-%H_%M")
save_location = f'meta_results/{__file__.split("/")[-1][:-3]}_{date_time_now}.xlsx'
# env_vars.change_use_gpu(False)
for seed in seeds:
    for learning_rate in [0.01, 0.005]: # 0.1 lr is not good for GAT + linear / linear - too noisy
        for linear_embedding_size in [10, 50]:
            for function in functions:
                returned_dictionary = function(seed=seed, embedding_size=linear_embedding_size, learning_rate=learning_rate)
                df = pd.DataFrame({k: [str(v)] for k, v in returned_dictionary.items()})
                dfs.append(df)
                final_df = pd.concat(dfs, axis=0)
                final_df.to_excel(save_location)
