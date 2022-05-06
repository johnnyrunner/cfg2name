import torch

from NeuralModels.function_namer import FunctionNamer
from config import PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE, env_vars, DIRE_OUTPUT_SIZE


class BlankFunctionNamer(FunctionNamer):
    def __init__(self, encode_function_as_id=False, *args, **kwargs):
        FunctionNamer.__init__(self, *args, **kwargs)
        self.is_function_id = encode_function_as_id

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        _, graph_data = x
        # todo: change to function encoding instead of variable
        number_of_functions = len(graph_data[PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE])
        functions_encoding = torch.ones((number_of_functions, 64, DIRE_OUTPUT_SIZE))
        if self.is_function_id:
            # decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            zeros_filler = torch.zeros((number_of_functions, 64, DIRE_OUTPUT_SIZE), device=env_vars.torch_device)
            zeros_filler[
                torch.arange(number_of_functions), torch.arange(number_of_functions) % 64, :] = 1
            functions_encoding = zeros_filler
        if env_vars.use_gpu:
            functions_encoding = functions_encoding.cuda()
        embedding = self.top_model.forward(functions_encoding, graph_data)
        # print(embedding)
        guessed_word = self.words_guesser(embedding)
        return guessed_word, guessed_word, None, graph_data
