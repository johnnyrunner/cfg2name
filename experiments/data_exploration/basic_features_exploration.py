import pickle

import seaborn as sn
import copy
import os
from collections import Counter
from typing import List

import numpy as np
import networkx
import sklearn
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from tqdm import tqdm
import matplotlib.pyplot as plt

from Datasets.graph_features_builder import GraphFeaturesBuilder
from config import DEMANGLED_NAME, DATA_EXPLORATION_RESULTS, DATA_EXPLORATION_CACHE
from experiments.data_exploration.exploration_config import PROGRAM_ID, SUBTOKEN, IN_DEGREE, OUT_DEGREE, \
    NUM_IN_PARAMS, NUM_OUT_PARAMS
from experiments.data_exploration.vocabulary_stats import get_function_vocab
import pandas as pd

from utils.files_utils import get_program_id_from_gexf_name
from utils.function_utils import flatten_python_list, check_if_function_is_not_interesting
from vocabulary_builder.functions_vocab import FunctionsVocab


def extract_features_for_graphs(
        # gexf_files_dir: str = 'D:\\routinio_data\\data_new\\decompiled_binaries\\function_call_graphs',
        gexf_files_dir='/media/jonathan/New Volume/routinio_data/data_new_dire/decompiled_binaries_test/function_call_graphs',
        features_list: List[str] = list(GraphFeaturesBuilder.features_dict.keys()),
        top_number: int = 100):
    '''

    :param gexf_files_dir:
    :param features_list: must cotain 'demangled_name'
    :param top_number:
    :return:
    '''
    print('calculating...')
    assert 'demangled_name' in features_list
    graphs_features = []
    for gexf_file_name in tqdm(os.listdir(gexf_files_dir)[:top_number], unit='file'):
        if gexf_file_name.split('.')[-1] == 'gexf' and '.stripped' not in gexf_file_name:
            try:
                program_id = get_program_id_from_gexf_name(gexf_file_name, gexf_files_dir)
                original_graph = networkx.read_gexf(os.path.join(gexf_files_dir, gexf_file_name))
                graph_features_builder = GraphFeaturesBuilder(original_graph, features_list, program_name=program_id)
                graph_features_df = graph_features_builder.generate_nodes_features()
                graphs_features.append(graph_features_df)
                print('succeeded getting graph_df')
            except:
                print('failed getting graph_df')
    return graphs_features


def row_to_multiple_subtoken_rows(row: pd.Series):
    rows_list = []
    for subtoken in row['encoded_demangled_name'].split(','):
        row_tag = copy.deepcopy(row)
        row_tag[SUBTOKEN] = subtoken
        rows_list.append(list(row_tag))
    return rows_list


def get_subtoken_rows_features(graphs_features_dfs_list, functions_vocab):
    concatenated_df = pd.concat(graphs_features_dfs_list)
    concatenated_df['encoded_demangled_name'] = concatenated_df[DEMANGLED_NAME].apply(
        functions_vocab.encode_as_subtokens)
    subtoken_rows = concatenated_df.apply(row_to_multiple_subtoken_rows, axis=1)
    all_subtokens_concatenated_df = pd.DataFrame(flatten_python_list(list(subtoken_rows)),
                                                 columns=list(concatenated_df.columns) + [SUBTOKEN])
    all_subtokens_concatenated_df = all_subtokens_concatenated_df[
        all_subtokens_concatenated_df['subtoken'].map(lambda x: not check_if_function_is_not_interesting(x))]
    return all_subtokens_concatenated_df


def aggregate_appropriate_columns(all_subtokens_concatenated_df):
    aggregated_subtokens = all_subtokens_concatenated_df.astype({OUT_DEGREE: 'float', IN_DEGREE: 'float',
                                                                 NUM_IN_PARAMS: 'float', NUM_OUT_PARAMS: 'float'}). \
        groupby(SUBTOKEN).agg({OUT_DEGREE: [np.mean, np.std], IN_DEGREE: [np.mean, np.std], SUBTOKEN: 'count',
                               NUM_IN_PARAMS: [np.mean, np.std], NUM_OUT_PARAMS: [np.mean, np.std],
                               PROGRAM_ID: pd.Series.nunique})
    sorted_in_degree = aggregated_subtokens.sort_values(by=(IN_DEGREE, 'mean'), ascending=False)
    sorted_out_degree = aggregated_subtokens.sort_values(by=(OUT_DEGREE, 'mean'), ascending=False)
    return sorted_in_degree, sorted_out_degree


def split_df_to_X_and_y(df, functions_vocab, x_params=[IN_DEGREE, OUT_DEGREE, NUM_IN_PARAMS, NUM_OUT_PARAMS]):
    y = df[SUBTOKEN].apply(lambda x: functions_vocab._vocab_entry.word2id[x] if x not in ['', '*&'] else -999)
    X = df[x_params]
    return X, y, df[SUBTOKEN]


# evaluate a model
def evaluate_model(X, y, X_test, y_test, model, labels, n_jobs=-1, conf_matrix=False):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    # evaluate model
    test_score = -1
    if not conf_matrix:
        scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
        if X_test is not None and y_test is not None:
            model.fit(X, y)
            y_test_pred = model.predict(X_test)
            test_score = sklearn.metrics.balanced_accuracy_score(y_test, y_test_pred)
            print(len(np.unique(y)), len(np.unique(y_test)))
            print(test_score, np.mean(scores))
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs)
        scores = confusion_matrix(labels, y_pred)
    return scores, test_score


def multiclassify_by_params(params_df, functions_vocab: FunctionsVocab, min_number_of_examples_per_class=10,
                            x_params=[IN_DEGREE, OUT_DEGREE, NUM_IN_PARAMS, NUM_OUT_PARAMS], test_df=None, **kwargs):
    X, y, labels = load_data(functions_vocab, min_number_of_examples_per_class, params_df, x_params=x_params)
    if test_df is not None:
        X_test, y_test, labels_test = load_data(functions_vocab, min_number_of_examples_per_class, test_df,
                                                x_params=x_params)
    else:
        X_test, y_test, labels_test = None, None, None
    number_of_classes = len(np.unique(y))
    number_of_features = np.array(X).shape[1]
    print(
        f'number of different classes: {number_of_classes}, number of examples {len(y)}, number of features {number_of_features}')

    # define the reference model
    model = RandomForestClassifier(n_estimators=1000, class_weight='balanced')

    # evaluate the model
    print('starting to train & validate')
    scores, test_score = evaluate_model(X, y, X_test, y_test, model, labels, **kwargs)
    # summarize performance
    try:
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    except:
        print('probably a confusion matrix')
    return scores, number_of_classes, test_score


def load_data(functions_vocab, k_neighbours, params_df, plot=False, balance_classes=False,
              x_params=[IN_DEGREE, OUT_DEGREE, NUM_IN_PARAMS, NUM_OUT_PARAMS],
              remove_small_classes: bool = False):
    if remove_small_classes:
        params_df_counts = params_df.groupby(SUBTOKEN).agg({SUBTOKEN: 'count'})[SUBTOKEN]
        groups_dict = dict(params_df_counts)
        params_df = params_df[params_df[SUBTOKEN].apply(lambda x: groups_dict[x] > k_neighbours)]
    X, y, labels = split_df_to_X_and_y(params_df, functions_vocab, x_params=x_params)
    # y = LabelEncoder().fit_transform(y)
    # if balance_classes:
    #     oversample = SMOTE(k_neighbors=k_neighbours)
    #     X, y = oversample.fit_resample(X, y)
    #     if plot:
    #         counter = Counter(y)
    #         for k, v in counter.items():
    #             per = v / len(y) * 100
    #             print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    #         # plot the distribution
    #         pyplot.bar(counter.keys(), counter.values())
    #         pyplot.show()
    return X, y, labels


def to_excel_after_aggrgation_and_sort(all_subtokens_concatenated_df, top_number):
    sorted_in_degree, sorted_out_degree = aggregate_appropriate_columns(all_subtokens_concatenated_df)
    sorted_in_degree.to_excel(os.path.join(DATA_EXPLORATION_RESULTS, f'sorted_in_degree_{top_number}.xlsx'))
    sorted_out_degree.to_excel(os.path.join(DATA_EXPLORATION_RESULTS, f'sorted_out_degree_{top_number}.xlsx'))


def basic_features_difference_between_classes(top_number=100, number_of_classes=30):
    features_list = ['demangled_name', 'mangled_name', 'out_degree', 'in_degree', 'num_in_params', 'num_out_params']
    _, functions_vocab = get_function_vocab(use_cache=True)
    all_subtokens_concatenated_df = get_all_subtokens_df(features_list, top_number,
                                                         'classes_difference', functions_vocab=functions_vocab)

    chosen_subtokens = np.random.choice(all_subtokens_concatenated_df[SUBTOKEN].unique(), number_of_classes)
    relevant_classic_subtokens_concatenated_df = all_subtokens_concatenated_df[
        all_subtokens_concatenated_df['subtoken'].isin(chosen_subtokens)]

    conf_matrix, _, _ = multiclassify_by_params(relevant_classic_subtokens_concatenated_df, functions_vocab,
                                                        conf_matrix=True)
    print(f'number of classes is {number_of_classes}')
    print(conf_matrix)
    df_cm = pd.DataFrame(conf_matrix)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title('confusion_matrix')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()


def basic_features_vs_random(top_number=1000, train_ration=0.8):
    real_scores_mean_balanced_accuracy = []
    random_scores_mean_balanced_accuracy = []
    real_std = []
    random_std = []
    numbers_of_classes = list(range(100, 1001, 100))
    number_of_classes_real = []
    test_scores = []
    for number_of_classes in numbers_of_classes:
        features_list = ['demangled_name', 'mangled_name', 'out_degree', 'in_degree', 'num_in_params', 'num_out_params']
        _, functions_vocab = get_function_vocab(use_cache=True)
        all_subtokens_concatenated_df = get_all_subtokens_df(features_list, top_number, 'basic_features',
                                                             functions_vocab)
        chosen_subtokens = np.random.choice(all_subtokens_concatenated_df['subtoken'].unique(), number_of_classes)
        relevant_classic_subtokens_concatenated_df = all_subtokens_concatenated_df[
            all_subtokens_concatenated_df['subtoken'].isin(chosen_subtokens)]
        # to_excel_after_aggrgation_and_sort(all_subtokens_concatenated_df, top_number)

        token_to_test_programs = {}
        token_to_train_programs = {}
        for token in chosen_subtokens:
            relevant_classic_subtoken_concatenated_df = all_subtokens_concatenated_df[
                all_subtokens_concatenated_df['subtoken'] == token]
            subtoken_programs = np.unique(relevant_classic_subtoken_concatenated_df[PROGRAM_ID])
            slice_int = int(len(subtoken_programs) * train_ration)
            train_subtoken_programs = subtoken_programs[:slice_int]
            token_to_train_programs[token] = train_subtoken_programs
            test_subtoken_programs = subtoken_programs[slice_int:]
            token_to_test_programs[token] = test_subtoken_programs

        relevant_classic_subtokens_concatenated_df_train = relevant_classic_subtokens_concatenated_df[
            relevant_classic_subtokens_concatenated_df.apply(
                lambda x: x[PROGRAM_ID] in token_to_train_programs[x[SUBTOKEN]], axis=1)
        ]
        relevant_classic_subtokens_concatenated_df_test = relevant_classic_subtokens_concatenated_df[
            relevant_classic_subtokens_concatenated_df.apply(
                lambda x: x[PROGRAM_ID] in token_to_test_programs[x[SUBTOKEN]], axis=1)
        ]

        real_scores, number_of_classes_1, test_score = multiclassify_by_params(
            relevant_classic_subtokens_concatenated_df_train,
            functions_vocab,
            test_df=relevant_classic_subtokens_concatenated_df_test)
        features_list = ['random_feature', 'demangled_name']
        random_all_subtokens_concatenated_df = get_all_subtokens_df(features_list, top_number,
                                                                    'random_feature', functions_vocab)
        relevant_random_subtokens_concatenated_df = random_all_subtokens_concatenated_df[
            random_all_subtokens_concatenated_df['subtoken'].isin(chosen_subtokens)]

        random_scores, number_of_classes_2, _ = multiclassify_by_params(relevant_random_subtokens_concatenated_df,
                                                                           functions_vocab,
                                                                           x_params=['random_feature'])

        # assert number_of_classes_1 == number_of_classes_2
        number_of_classes_real.append(number_of_classes_1)
        real_scores_mean_balanced_accuracy.append(np.mean(real_scores))
        real_std.append(np.std(real_scores))
        random_scores_mean_balanced_accuracy.append(np.mean(random_scores))
        random_std.append(np.std(random_scores))
        test_scores.append(test_score)

    plt.errorbar(number_of_classes_real, real_scores_mean_balanced_accuracy, fmt='', ls='none', yerr=real_std)
    plt.errorbar(number_of_classes_real, random_scores_mean_balanced_accuracy, fmt='', ls='none', yerr=random_std)
    plt.plot(number_of_classes_real, real_scores_mean_balanced_accuracy, '.')
    plt.plot(number_of_classes_real, random_scores_mean_balanced_accuracy, '.')
    plt.plot(number_of_classes_real, [1 / num for num in number_of_classes_real], '.')
    plt.plot(number_of_classes_real, test_scores, '.')
    plt.xlabel('number of classes (correlative to number of examples)')
    plt.ylabel('balanced_accuracy')
    plt.legend(['simple features - CV train score', 'random feature - CV all score', '1/number_of_classes',
                'test score (different projects)'])
    plt.title('real & random balanced accuracy as a function of number of classes')
    plt.show()
    # plt.savefig('balanced accuracy as a function of number of classes')


def get_all_subtokens_df(features_list, top_number, cache_name, functions_vocab=None):
    cache_base_name = f'all_subtokens_df_{cache_name}_top_{top_number}.pkl'
    cache_file_path = os.path.join(DATA_EXPLORATION_CACHE, cache_base_name)
    if not os.path.exists(cache_file_path):
        graphs_features_dfs_list = extract_features_for_graphs(features_list=features_list, top_number=top_number)
        all_subtokens_concatenated_df = get_subtoken_rows_features(graphs_features_dfs_list, functions_vocab)
        with open(cache_file_path, 'wb+') as f:
            pickle.dump(all_subtokens_concatenated_df, f)
            print('dumped')
            print(f'to: {cache_file_path}')
    else:
        with open(cache_file_path, 'rb') as f:
            all_subtokens_concatenated_df = pickle.load(f)
            print('loaded all_subtokens_concatenated_df')
            print(f'from: {cache_file_path}')

    return all_subtokens_concatenated_df


if __name__ == '__main__':
    # basic_features_vs_random(top_number=200)
    basic_features_difference_between_classes(top_number=200)
