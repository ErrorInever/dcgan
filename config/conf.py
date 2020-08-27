from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.NUM_EPOCHS = 250
__C.BATCH_SIZE = 100
__C.LEARNING_RATE = 0.0002
__C.WORKERS = 2
__C.BETA_1 = 0.5
__C.OUT_DIR = '.'