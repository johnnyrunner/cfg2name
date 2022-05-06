from experiments.exp2_5_check_dire import run_experiment as run_experiment_2_5
from experiments.exp3_check_dire_graph_on_top import run_experiment as run_experiment_3

test_dicts = []
for seed in [50, 51, 52, 53, 54, 55, 56]:
    test_dicts.append(run_experiment_3(seed=seed, learning_rate=0.01, embedding_size=10))
    test_dicts.append(run_experiment_2_5(seed=seed, learning_rate=0.01, embedding_size=10))
    test_dicts.append(run_experiment_3(seed=seed, learning_rate=0.01, embedding_size=100))
    test_dicts.append(run_experiment_2_5(seed=seed, learning_rate=0.01, embedding_size=100))
    lr = 0.01
    for em_size in [200, 100, 5]:
        print('-----------------------')
        print('stripped binaries')
        print(em_size)
        print(lr)
        for test_dict in test_dicts:
            if test_dict[-1]['embedding_size'] == em_size and test_dict[-1]['learning_rate'] == lr:
                print(test_dict)
