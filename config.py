
SERVER_IP = '127.0.0.1'
PORT_MAIN = 6006

MODEL_NAME = 'RepVGG-B3'

WEIGHT_PATH_SERVER = './weights/RepVGG-B3-200epochs-train.pth'
WEIGHT_PATH_CLIENT = './weights/RepVGG-B3-200epochs-train.pth'
NUM_BLOCKS = 27

BLOCK_INDEX_MAP = {
    'stage1.0': 0, 'stage1.1': 1, 'stage1.2': 2, 'stage1.3': 3,
    'stage2.0': 4, 'stage2.1': 5, 'stage2.2': 6, 'stage2.3': 7, 'stage2.4': 8, 'stage2.5': 9,
    'stage3.0': 10, 'stage3.1': 11, 'stage3.2': 12, 'stage3.3': 13, 'stage3.4': 14,
    'stage3.5': 15, 'stage3.6': 16, 'stage3.7': 17, 'stage3.8': 18, 'stage3.9': 19,
    'stage3.10': 20, 'stage3.11': 21, 'stage3.12': 22, 'stage3.13': 23,
    'stage3.14': 24, 'stage3.15': 25,
    'stage4.0': 26
}

INDEX_BLOCK_MAP = {v: k for k, v in BLOCK_INDEX_MAP.items()}

FIXED_SPLIT_POINT = 10
SPLIT_SIGMA = 5.0

LR_GAMMA = 0.01
LR_LAMBDA = 0.01
THROUGHPUT_OPTIM_STEPS = 800

MAX_PRUNING_ROUNDS = 10
CANDIDATE_POOL_SIZE = 10
EPOCHS_PER_TEST = 3
THRESHOLD_PERCENT = 2.0
LR_ALPHA = 0.01
L1_LAMBDA = 0.001
PRUNING_THRESHOLD = 0.01

OMEGA_TIME_SERVER = 10.0
OMEGA_TIME_CLIENT = 10.0
OMEGA_TRANS = 5.0

OMEGA_ROOFLINE_CLIENT = 0.0
OMEGA_ROOFLINE_SERVER = 0.0

OMEGA_INTENSITY = 0.05

ALPHA_ROOFLINE_LOWER = 5.0
BETA_ROOFLINE_UPPER = 1.0
CLIENT_INTENSITY_UPPER_RATIO = 1.2
SERVER_INTENSITY_UPPER_RATIO = 2.0

ETA_L1 = 0.001
MU_THROUGHPUT = 1e-6

I_KNEE_SERVER = 25.15
I_KNEE_CLIENT = 16.1

NUM_DOMAINS = 10
BATCH_SIZE = 32
DATA_PATH = './data/imagenet_val'
PARTITION_INFO_PATH = './data/noniid_partitions/partition_info.json'

TRAINING_RESULTS_FILE = f'task1_trained_weights_{MODEL_NAME}/training_results.json'

IMPORTANCE_RANKINGS_FILE = f'task2_importance_rankings_{MODEL_NAME}/importance_rankings.json'
BRANCH_FLOPS_FILE = f'task2_importance_rankings_{MODEL_NAME}/branch_flops.json'

OUTPUT_DIR = 'joint_optimization_results'
LUT_OUTPUT_FILE = f'{OUTPUT_DIR}/hardware_lut_complete.json'

class CMD:
    LUT_MEASURE = 0x01
    LUT_RESULT = 0x02
    SYNC_MODEL = 0x03
    FORWARD_DATA = 0x04
    BACKWARD_GRAD = 0x05
    EVAL_REQUEST = 0x06
    EVAL_RESULT = 0x07
    TRAIN_EPOCH_START = 0x08
    TRAIN_EPOCH_END = 0x09
    HYPER_WEIGHTS = 0x0A
    DOMAIN_START = 0x0B
    DOMAIN_END = 0x0C
    ACK = 0x0D

    EVALUATE_START = 0x10
    EVALUATE_BATCH = 0x11
    EVALUATE_BATCH_RESULT = 0x12
    EVALUATE_END = 0x13
    
    SHUTDOWN = 0xFF