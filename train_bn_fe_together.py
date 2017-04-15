from global_head_file import *
from train_bottleneck import *
from transfer_learning import *
print('start training bottleneck feature')
train_network_BN(num_iterations = 50, resume = 1, learning_rate = 0)
# print('start training feature extraction')
# train_network_FE(100, resume = True, enable_dropout= True, learning_rate = 0, FE_phase1 = feature_extraction_status.feature_extraction)
# print('start fine tune network')
# train_network_FE(10, resume = True, enable_dropout= True,learning_rate = 0, FE_phase1 = feature_extraction_status.fine_tune)

