from experiments.exp2_5_check_dire import run_experiment as run_experiment_2_5
from experiments.exp3_check_dire_graph_on_top import run_experiment as run_experiment_3
from experiments.exp5_check_dire_graph_on_top_rnn import run_experiment as run_experiment_5
from experiments.exp6_check_dire_Id_on_top_rnn import run_experiment as run_experiment_6
from experiments.exp2_7_check_dire_triple_linear import run_experiment as run_experiment_2_7
from experiments.exp3_5_check_dire_graph_on_top import run_experiment as run_experiment_3_5

for seed in [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]:
    run_experiment_2_5(seed=seed)
    run_experiment_3(seed=seed)
    run_experiment_2_5(seed=seed, learning_rate=0.005)
    run_experiment_3(seed=seed, learning_rate=0.005)
    run_experiment_2_7(seed=seed)
    run_experiment_3_5(seed=seed)
    run_experiment_2_7(seed=seed, learning_rate=0.005)
    run_experiment_3_5(seed=seed, learning_rate=0.005)
    try:
        run_experiment_5(seed=seed)
    except:
        print('failed')
    try:
        run_experiment_6(seed=seed)
    except:
        print('failed')
