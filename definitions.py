import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
NLVR_DATA = os.path.join(DATA_DIR, 'nlvr-data')
SEQ2SEQ_DIR = os.path.join(ROOT_DIR, 'seq2seqModel')

TRAIN_JSON = os.path.join(NLVR_DATA, 'train', 'train.json')
DEV_JSON = os.path.join(NLVR_DATA, 'dev', 'dev.json')
TEST_JSON = os.path.join(NLVR_DATA, 'test', 'test.json')

TEST2_JSON = '' # '' # ADD THE PATH TO THE HIDDEN TEST JSON HERE
version1 = False
## PLEASE RUN ONCE WITH TRUE AND ONCE WITH FALSE ##

TRAIN_IMAGES = os.path.join(NLVR_DATA, 'train', 'images')
DEV_IMAGES = os.path.join(NLVR_DATA, 'dev', 'images')
TEST_IMAGES = os.path.join(NLVR_DATA, 'test', 'images')

ENG_VOCAB_60K = os.path.join(DATA_DIR, 'sentence-processing', 'en-vocabulary-60k.txt')
TOKEN_COUNTS = os.path.join(DATA_DIR, 'sentence-processing', 'tokens_spellproofed.txt')
BIGRAM_COUNTS = os.path.join(DATA_DIR, 'sentence-processing', 'bigrams_spellproofed.txt')
TOKEN_COUNTS_PROCESSED = os.path.join(DATA_DIR, 'sentence-processing', 'tokens_processed.txt')
BIGRAM_COUNTS_PROCESSED = os.path.join(DATA_DIR, 'sentence-processing', 'bigrams_processed.txt')


if version1:
    MANUAL_REPLACEMENTS = False
else:
    MANUAL_REPLACEMENTS = True

if MANUAL_REPLACEMENTS:
    SUPERVISED_TRAIN_PICKLE = os.path.join(DATA_DIR, 'parsed sentences', 'pairs_train_final')
    SUPERVISED_TRAIN_PICKLE_3 = os.path.join(DATA_DIR, 'parsed sentences', 'pairs_train_3')
    SUPERVISED_VALIDATION_PICKLE = os.path.join(DATA_DIR, 'parsed sentences', 'pairs_validation_final')
    LOGICAL_TOKENS_MAPPING_PATH = os.path.join(DATA_DIR, 'logical forms', 'token mapping.txt')
    SYNONYMS_PATH = os.path.join(DATA_DIR, 'sentence-processing', 'manual_replacements.txt')
else:
    SUPERVISED_TRAIN_PICKLE = os.path.join(DATA_DIR, 'parsed sentences', 'new_proper_pairs_train_final')
    SUPERVISED_VALIDATION_PICKLE = os.path.join(DATA_DIR, 'parsed sentences', 'new_proper_pairs_validation_final')
    LOGICAL_TOKENS_MAPPING_PATH = os.path.join(DATA_DIR, 'logical forms', 'new token mapping.txt')
