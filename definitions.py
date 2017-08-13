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
