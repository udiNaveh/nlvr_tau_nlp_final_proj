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
AVOID_ALL_TRUE_SENTENCES = False
PRINT_EVERY = 5


#paths


WORD_EMBEDDINGS_PATH = os.path.join(definitions.ROOT_DIR, 'word2vec', 'embeddings_10iters_12dim')
PRE_TRAINED_WEIGHTS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'learnedWeights', 'trained_variables_sup_check_hs4.ckpt')
TRAINED_WEIGHTS_BEST = \
    os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeightsUns','weights_cached_auto_inj2017-09-09_10_49.ckpt-15')
LOGICAL_TOKENS_LIST =  os.path.join(definitions.DATA_DIR, 'logical forms', 'logical_tokens_list')
CACHED_PROGRAMS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'output decodings', 'cached_programs')
CACHED_PROGRAMS_PRETRAIN = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'outputs',
                                        'cached_programs_based_on_pretrain')
BEAMS_PATH = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'outputs', 'beams_by_sentences')
NGRAM_PROBS =  os.path.join(definitions.DATA_DIR, 'sentence-processing', 'ngram_logprobs')


#beam settings
MAX_DECODING_LENGTH = 22
MAX_STEPS = 14
BEAM_SIZE = 40
SKIP_AUTO_TOKENS = True
INJECT_TO_BEAM = False and USE_CACHED_PROGRAMS

# chosen weights path - change every time
INPUT_WEIGHTS = PRE_TRAINED_WEIGHTS #TRAINED_WEIGHTS_SUP_HISTORY_4
OUTPUT_WEIGHTS = None






