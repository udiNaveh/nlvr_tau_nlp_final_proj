import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
NLVR_DATA = os.path.join(DATA_DIR, 'nlvr-data')

TRAIN_JSON = os.path.join(NLVR_DATA, 'train', 'train.json')
DEV_JSON = os.path.join(NLVR_DATA, 'dev', 'dev.json')
TEST_JSON = os.path.join(NLVR_DATA, 'test', 'test.json')

TRAIN_IMAGES = os.path.join(NLVR_DATA, 'train', 'images')
DEV_IMAGES = os.path.join(NLVR_DATA, 'dev', 'images')
TEST_IMAGES = os.path.join(NLVR_DATA, 'test', 'images')

ENG_VOCAB_60K =  os.path.join(DATA_DIR, 'sentence-processing', 'en-vocabulary-60k.txt')
TOKEN_COUNTS =  os.path.join(DATA_DIR, 'sentence-processing', 'tokens_spellproofed.txt')
BIGRAM_COUNTS =  os.path.join(DATA_DIR, 'sentence-processing', 'bigrams_spellproofed.txt')
TOKEN_COUNTS_PROCESSED =  os.path.join(DATA_DIR, 'sentence-processing', 'tokens_processed.txt')
BIGRAM_COUNTS_PROCESSED =  os.path.join(DATA_DIR, 'sentence-processing', 'bigrams_processed.txt')

SYNONYMS =  os.path.join(DATA_DIR, 'sentence-processing', 'manual_replacements.txt')
SUPERVISED_TRAIN_PICKLE = os.path.join(ROOT_DIR, 'pre-training', 'pairs_train_2')
SUPERVISED_VALIDATION_PICKLE = os.path.join(ROOT_DIR, 'pre-training', 'pairs_validation')
SENTENCES_IN_PRETRAIN_PATTERNS = os.path.join(ROOT_DIR, 'pre-training', 'sentences_in_pattern')