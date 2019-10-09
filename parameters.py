import dataset_path
import argparse
dataset_dir = dataset_path.dataset_dir


parser = argparse.ArgumentParser()
parser.add_argument('-model', default="")
parser.add_argument('-ducmodel', default="")
args = parser.parse_args()


model_name_dict = {'bg': 'baselinegigaword',
                   'lg100d5g': 'lg100d5gigaword',
                   'lg200d5g': 'lg200d5gigaword',
                   'lg500d5g': 'lg500d5gigaword',
                   'lg1000d5g': 'lg1000d5gigaword',
                   'lgAlld5g': 'lgAlld5gigaword',
                   'neg100g': 'neg100gigaword',
                   'neg200g': 'neg200gigaword',
                   'neg500g': 'neg500gigaword',
                   'neg1000g': 'neg1000gigaword',
                   'negAllg': 'negAllgigaword',
                   ####DUC Models
                   'bd': 'baselineduc',
                   'lg100d5d': 'lg100d5duc',
                   'lg200d5d': 'lg200d5duc',
                   'lg500d5d': 'lg500d5duc',
                   'lg1000d5d': 'lg1000d5duc',
                   'lgAlld5d': 'lgAlld5duc',
                   'neg100d': 'neg100duc',
                   'neg200d': 'neg200duc',
                   'neg500d': 'neg500duc',
                   'neg1000d': 'neg1000duc',
                   'negAlld': 'negAllduc',
                   "": None}



model_id = model_name_dict[args.model]  #model_name_dict['lg100d5g']
#model_id = model_name_dict['bg']


duc_model_id = model_name_dict[args.ducmodel]


#mode_list = [None,  # mode_list[0]
#             'build_training_dataset',  # mode_list[1]
#             'build_validation_dataset',  # mode_list[2]
#             'build_testing_data',  # mode_list[3]
#             'train',  # mode_list[4] train
#             'test',  # mode_list[5] test
#             'build_DUC_testing_data',  # mode_list[6]
#             'test_DUC',  # mode_list[7] test
#             ]
#mode = mode_list[1]

##### Parameters
model_name = model_id
start_epoch_no = 1  # the training numbering starts from start_epoch_no
epochs_num = 12
batch_size = 64
embedding_dim = 300
hidden_dim = 200
layers_num = 2
learning_rate = 1e-3
beam_width = 4
keep_prob = 0.8
using_word2vec_embeddings = True

# training
true_false = [False, True]
sample_training_dataset = true_false[0]  # true_false[1]: True for sample training
train_restored_saved_model = True
print_loss_per_steps = 500

# building training dataset
lines_per_chunk = 300000 #
read_lines = 9111222333 #a very large number for reading all lines