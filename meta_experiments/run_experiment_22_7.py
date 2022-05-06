from experiments.exp2_5_check_dire import run_experiment as run_experiment_2_5
from experiments.exp3_check_dire_graph_on_top import run_experiment as run_experiment_3

test_dicts = []
for seed in [45, 46, 47, 48, 49, 50, 51, 52, 53]:
    test_dicts.append(run_experiment_2_5(seed=seed, learning_rate=0.01, embedding_size=50))
    test_dicts.append(run_experiment_3(seed=seed, learning_rate=0.01, embedding_size=50))
    test_dicts.append(run_experiment_2_5(seed=seed, learning_rate=0.01, embedding_size=100))
    test_dicts.append(run_experiment_3(seed=seed, learning_rate=0.01, embedding_size=100))
    test_dicts.append(run_experiment_2_5(seed=seed, learning_rate=0.005, embedding_size=50))
    test_dicts.append(run_experiment_3(seed=seed, learning_rate=0.005, embedding_size=50))
    test_dicts.append(run_experiment_2_5(seed=seed, learning_rate=0.005, embedding_size=100))
    test_dicts.append(run_experiment_3(seed=seed, learning_rate=0.005, embedding_size=100))
    test_dicts.append(run_experiment_2_5(seed=seed, learning_rate=0.01, embedding_size=5))
    test_dicts.append(run_experiment_3(seed=seed, learning_rate=0.01, embedding_size=5))
    for em_size in [50, 100, 5]:
        for lr in [0.005, 0.01]:
            print('-----------------------')
            print(em_size)
            print(lr)
            for test_dict in test_dicts:
                if test_dict[-1]['embedding_size'] == em_size and test_dict[-1]['learning_rate'] == lr:
                    print(test_dict)
