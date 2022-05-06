from pathlib import Path

import config
from config import DATA_EXPLORATION_RESULTS, DATA_EXPLORATION_CACHE

PROGRAM_ID = config.PROGRAM_ID
SUBTOKEN = 'subtoken'
IN_DEGREE = 'in_degree'
OUT_DEGREE = 'out_degree'
NUM_IN_PARAMS = 'num_in_params'
NUM_OUT_PARAMS = 'num_out_params'


# Path(DATA_EXPLORATION_RESULTS).mkdir(exist_ok=True, parents=True)
#
# Path(DATA_EXPLORATION_CACHE).mkdir(exist_ok=True, parents=True)