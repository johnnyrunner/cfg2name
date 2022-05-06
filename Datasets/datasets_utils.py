from Datasets.programs_dataset import ProgramsDataset, SMALL_PROGRAM_GRAPH_DATASET_ROOT_DIR
from config import PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE
from utils.general_utils import set_seed

def get_masked_names_from_data_graph(data_graph):
    should_validate_mask = data_graph[PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE]
    should_validate_mask = (should_validate_mask == 1).view(-1, 1)
    masked_names = [name for name, true_by_mask in zip(data_graph['demangled_function_names'], should_validate_mask) if true_by_mask]
    return masked_names

def print_summerization_of_dataset(programs_dataset: ProgramsDataset):
    all_masked_names = {}
    for program in programs_dataset:
        data_ast, data_graph = program
        masked_names = get_masked_names_from_data_graph(data_graph)
        print(masked_names)

        for name in masked_names:
            if name in all_masked_names:
                all_masked_names[name] += 1
            else:
                all_masked_names[name] = 1
    print(all_masked_names)


if __name__ =='__main__':
    SEED = 44
    set_seed(SEED)
    print("Building ProgramsDataModule...")

    small_programs_dataset = ProgramsDataset(SMALL_PROGRAM_GRAPH_DATASET_ROOT_DIR, load_dire_vocab_from_file=True)
    print_summerization_of_dataset(small_programs_dataset)