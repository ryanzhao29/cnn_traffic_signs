from global_head_file import *
from train_bottleneck import *
from transfer_learning import *
print('start training bottleneck feature')
# train_network_BN(50, 1)
# print('start training feature extraction')
train_network_FE(50, resume = True, enable_dropout= False, FE_phase1 = feature_extraction_status.feature_extraction)
print('start fine tune network')
# train_network_FE(10, resume = True, enable_dropout= True, FE_phase1 = feature_extraction_status.fine_tune)

