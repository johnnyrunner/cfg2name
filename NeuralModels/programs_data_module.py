import math

import pytorch_lightning as pl
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from config import NUM_WORKERS

SIZE_LIMIT = 10


def limit_example_size(batch):
    for example in batch:
        data_ast, data_graph = example
        if len(data_graph['label']) > SIZE_LIMIT:
            return None
        else:
            return batch


class ProgramsDataModule(pl.LightningDataModule):
    def __init__(self, programs_dataset, train_percentage, val_percentage, batch_size, validation_dataset=None,
                 test_dataset=None, dont_use_test=True):
        super().__init__()
        self.batch_size = batch_size
        assert train_percentage + val_percentage <= 100, 'percentages must add up to 100'
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage

        self.all_programs_dataset = programs_dataset
        number_of_examples = len(self.all_programs_dataset)
        train_number_of_examples = math.floor(number_of_examples * (self.train_percentage / 100.0))
        val_number_of_examples = math.floor(number_of_examples * (self.val_percentage / 100.0))
        test_number_of_examples = number_of_examples - (train_number_of_examples + val_number_of_examples)
        assert all([train_number_of_examples, val_number_of_examples, test_number_of_examples]), \
            'Not all stages have program examples - \n\ttrain_number_of_examples {}\n\tval_number_of_examples {}\n\t ' \
            'test_number_of_examples {}'.format(train_number_of_examples, val_number_of_examples,
                                                test_number_of_examples)

        # if (validation_dataset is not None) or (test_dataset is not None):
        #     if test_dataset is not None or dont_use_test:
        #         print('here')
        #         self.train_dataset, self.val_dataset, self.test_dataset = programs_dataset, validation_dataset, test_dataset
        #     else:
        #         number_of_examples = len(validation_dataset)
        #         val_number_of_examples = math.floor(number_of_examples * (
        #                     self.val_percentage / (1 - self.val_percentage - self.train_percentage) / 100.0))
        #         test_number_of_examples = number_of_examples - val_number_of_examples
        #         self.train_dataset = programs_dataset
        #         self.val_dataset, self.test_dataset = random_split(
        #             validation_dataset, [val_number_of_examples, test_number_of_examples])
        # else:
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.all_programs_dataset, [train_number_of_examples, val_number_of_examples, test_number_of_examples])
        self.val_dataset = validation_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS, shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS,)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
