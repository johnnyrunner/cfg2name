import torch
import torch_geometric


def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()


def recursively_print_all_tensor_sizes(obj):
    print(type(obj))
    if isinstance(obj, torch.Tensor):
        print(tensor_size(obj))
    if isinstance(obj, list):
        print('list_start')
        for elem in obj:
            print('list_elem')
            recursively_print_all_tensor_sizes(elem)

        print('list_end')

    if isinstance(obj, torch_geometric.data.Batch):
        print('batch_start')
        for key, elem in obj:
            print(f'batch {key}')
            recursively_print_all_tensor_sizes(elem)
        print('batch_end')

    if isinstance(obj, dict):
        print('dict_start')
        for key, elem in obj.items():
            print(f'dict object {key}')
            recursively_print_all_tensor_sizes(elem)
        print('dict_end')

    if isinstance(obj, int):
        print(f'int {obj}')

    if isinstance(obj, str):
        print(f'str {obj}')