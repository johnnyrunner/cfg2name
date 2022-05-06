from datetime import datetime
from experiments.exp6_5_check_nero import run_experiment as run_experiment_6_5
from experiments.exp2_5_check_dire import run_experiment as run_experiment_2_5
from experiments.exp6_6_check_nero_gat_linear import run_experiment as run_experiment_6_6
from experiments.exp3_7_check_dire_graph_on_top import run_experiment as run_experiment_3_7
from experiments.exp6_3_check_nero_triple_linear import run_experiment as run_experiment_6_3
from experiments.exp2_7_check_dire_triple_linear import  run_experiment as run_experiment_2_7
from experiments.exp3_8_check_dire_double_linear_graph_on_top import run_experiment as run_experiment_3_8
from experiments.exp6_8_check_nero_gat import run_experiment as run_experiment_6_8

import pandas as pd

nero_functions = [run_experiment_6_8, run_experiment_6_3]
dire_functions = [run_experiment_2_7, run_experiment_3_8]

seeds = [52, 53, 54, 55, 56]
more_seeds = [62, 63, 64, 65, 66]
more_more_seeds = [72, 73, 74, 75, 76]

dfs = []
now = datetime.now()
date_time_now = dt_string = now.strftime("%d_%m-%H_%M")
save_location = f'meta_results/{__file__.split("/")[-1][:-3]}_{date_time_now}.xlsx'
for seed in more_more_seeds:
    linear_embedding_size = 30
    learning_rate = 0.0005
    for function in nero_functions:
        returned_dictionary = function(seed=seed, embedding_size=linear_embedding_size, learning_rate=learning_rate)
        df = pd.DataFrame({k: [str(v)] for k, v in returned_dictionary.items()})
        dfs.append(df)
        final_df = pd.concat(dfs, axis=0)
        final_df.to_excel(save_location)

    learning_rate = 0.005
    for function in dire_functions:
        returned_dictionary = function(seed=seed, embedding_size=linear_embedding_size, learning_rate=learning_rate)
        df = pd.DataFrame({k: [str(v)] for k, v in returned_dictionary.items()})
        dfs.append(df)
        final_df = pd.concat(dfs, axis=0)
        final_df.to_excel(save_location)