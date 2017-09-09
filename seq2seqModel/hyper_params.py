import os
import definitions

####
###hyperparameters
####

#dimensions
WORD_EMB_SIZE = 12
LOG_TOKEN_EMB_SIZE = 12
DECODER_HIDDEN_SIZE = 50
LSTM_HIDDEN_SIZE = 30
SENT_EMB_SIZE = 2 * LSTM_HIDDEN_SIZE
HISTORY_LENGTH = 4

#other hyper parameters
LEARNING_RATE = 0.001
BETA = 0.5
EPSILON_FOR_BEAM_SEARCH = 0
MAX_N_EPOCHS = 15

BATCH_SIZE_UNSUPERVISED = 8
BATCH_SIZE_SUPERVISED = 10
USE_BOW_HISTORY = False
IRRELEVANT_TOKENS_IN_GRAD = True
AUTOMATIC_TOKENS_IN_GRAD = False
HISTORY_EMB_SIZE = HISTORY_LENGTH * LOG_TOKEN_EMB_SIZE
USE_CACHED_PROGRAMS = False
N_CACHED_PROGRAMS = 10 if USE_CACHED_PROGRAMS else 0
LOAD_CACHED_PROGRAMS = False
SAVE_CACHED_PROGRAMS = False
SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH = True
AVOID_ALL_TRUE_SENTENCES = True
PRINT_EVERY = 10


#paths

LOGICAL_TOKENS_MAPPING_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping_limitations')
WORD_EMBEDDINGS_PATH = os.path.join(definitions.ROOT_DIR, 'word2vec', 'embeddings_10iters_12dim')
TRAINED_WEIGHTS_SUP_HISTORY_4 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'learnedWeights', 'trained_variables_sup_check_hs4.ckpt')
TRAINED_WEIGHTS_SUP_HISTORY_6 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables_sup_check_hs6.ckpt')
TRAINED_WEIGHTS_UNSUP_HISTORY_4 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeightsUns','trained_variables_unsup_new_train.ckpt-1')
TRAINED_WEIGHTS_TEMP = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','temp.ckpt-1')
LOGICAL_TOKENS_LIST =  os.path.join(definitions.DATA_DIR, 'logical forms', 'logical_tokens_list')
CACHED_PROGRAMS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'output decodings', 'cached_programs')


#beam settings
MAX_DECODING_LENGTH = 22
MAX_STEPS = 14
BEAM_SIZE = 40
SKIP_AUTO_TOKENS = True
INJECT_TO_BEAM = False and USE_CACHED_PROGRAMS

# chosen weights path - change every time
INPUT_WEIGHTS = TRAINED_WEIGHTS_UNSUP_HISTORY_4 #TRAINED_WEIGHTS_SUP_HISTORY_4
OUTPUT_WEIGHTS = None






