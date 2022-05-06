from datetime import datetime
from experiments.exp6_8_check_nero_gat import run_experiment as run_experiment_6_8
from experiments.exp6_9_check_nero_gat_rnn import run_experiment as run_experiment_6_9
import pandas as pd

functions = [run_experiment_6_8, run_experiment_6_9]
seeds = [52, 53, 54, 55, 56]
dfs = []
now = datetime.now()
date_time_now = dt_string = now.strftime("%d_%m-%H_%M")
save_location = f'meta_results/{__file__.split("/")[-1][:-3]}_{date_time_now}.xlsx'

for seed in seeds:
    for function in functions:
        returned_dictionary = function(seed=seed, embedding_size=10)
        df = pd.DataFrame({k: [str(v)] for k, v in returned_dictionary.items()})
        dfs.append(df)
        final_df = pd.concat(dfs, axis=0)
        final_df.to_excel(save_location)

