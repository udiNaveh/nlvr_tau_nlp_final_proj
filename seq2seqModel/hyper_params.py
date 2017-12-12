import os
import definitions

####
###hyperparameters
####


#dimensions
WORD_EMB_SIZE = 12
LOG_TOKEN_EMB_SIZE = 12
DECODER_HIDDEN_SIZE = 50
DECODER_HIDDEN_SIZE_NEW = 500
LSTM_HIDDEN_SIZE = 30
SENT_EMB_SIZE = 2 * LSTM_HIDDEN_SIZE
HISTORY_LENGTH = 4

#other hyper parameters
REWARD_EVERY = 4
LEARNING_RATE = 0.001
BETA = 0.5
EPSILON_FOR_BEAM_SEARCH = 0
MAX_N_EPOCHS = 20

BATCH_SIZE_UNSUPERVISED = 8
BATCH_SIZE_SUPERVISED = 10
LEARN_EMBEDDINGS = True
LEARN_EMBEDDINGS_IN_PRETRAIN = False #TODO check False
USE_BOW_HISTORY = False
    # if true, a binary vector representing the tokens outputted so far in the program is concatenated
    # to the history embedding
SIMILARITY_WITH_P = False
IRRELEVANT_TOKENS_IN_GRAD = True
    # if false, a masking is used so that invalid tokens do not affect the gradient.

AUTOMATIC_TOKENS_IN_GRAD = False #TODO what is that?
    # if false, tokens that are added automatically to a program (when they are the only valid options,
    # are not used when taking the gradient.

HISTORY_EMB_SIZE = HISTORY_LENGTH * LOG_TOKEN_EMB_SIZE

USE_CACHED_PROGRAMS = True
N_CACHED_PROGRAMS = 10 if USE_CACHED_PROGRAMS else 0
LOAD_CACHED_PROGRAMS = False
SAVE_CACHED_PROGRAMS = False


AVOID_ALL_TRUE_SENTENCES = False
    # if true, the data set of the trainning will incluse only sentences that have also images labeles false.

PRINT_EVERY = 10
PRINT_PARAMS = True



#beam settings
MAX_DECODING_LENGTH = 22 # the maximum length of a program from the beam (in number of tokens)
MAX_STEPS = 14 # the default number of decoding steps for a program in the ebam search
BEAM_SIZE = 40
SKIP_AUTO_TOKENS = True
    # if true, tokens that are the only valid option are automatically added to the programs in the bean search,
    # in the same step.

INJECT_TO_BEAM = True and USE_CACHED_PROGRAMS
    # if true, the prefixes of suggested cached programs are injected to the beam at each step, if not in th beam already.

if definitions.version1:
    SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH = True
    # if true, the set of logical tokens that can be used in a parogram is reduced to tokens
    # that can relate to the content of the sentence ()
else:
    SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH = False

#paths

if definitions.MANUAL_REPLACEMENTS:
    WORD_EMBEDDINGS_PATH = os.path.join(definitions.SEQ2SEQ_DIR, 'word2vec', 'embeddings_10iters_12dim')
    PRE_TRAINED_WEIGHTS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'learnedWeightsPreTrain', 'trained_variables_sup_with_embeddings.ckpt')
else:
    WORD_EMBEDDINGS_PATH = os.path.join(definitions.SEQ2SEQ_DIR, 'word2vec', 'new_embeddings')
    PRE_TRAINED_WEIGHTS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'learnedWeights', 'new_trained_variables_sup_with_embeddings.ckpt')

# TRAINED_WEIGHTS_BEST = \
#     os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeightsWeaklySupervised','weights_best_8_replacements')
TRAINED_WEIGHTS_BEST = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeightsWeaklySupervised','weights_best_10_12')
RERANKER_WEIGHTS_BEST = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeightsWeaklySupervised','weights_best_reranker')
LOGICAL_TOKENS_LIST = os.path.join(definitions.DATA_DIR, 'logical forms', 'logical_tokens_list')
CACHED_PROGRAMS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'output decodings', 'cached_programs')
CACHED_PROGRAMS_PRETRAIN = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'outputs',
                                        'cached_programs_based_on_pretrain')
NGRAM_PROBS = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'ngram_logprobs')

