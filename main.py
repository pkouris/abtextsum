import time
import datetime
from train_v2 import Train
from validation import Validation
import parameters as param
from build_dataset import BuildDataset
from testing import Testing
from train_word2vec import TrainWord2Vec

class Main:

    def __init__(self):
        # print('Mode: ', param.mode)
        self.run(mode=param.mode, train_restored_saved_model=param.train_restored_saved_model)

    #@staticmethod
    def run(self, mode, train_restored_saved_model):
        start_time = time.time()
        print('Mode: ', mode)
        print('Model_id: ', param.model_id)
        # building dataset
        if mode == param.mode_list[1] or mode == param.mode_list[2] or mode == param.mode_list[3] or \
                mode == param.mode_list[6]:
            BuildDataset(mode=param.mode)
        # Training
        elif mode == param.mode_list[4]:
            Train()
        # Testing
        elif mode == param.mode_list[5]:
            Testing(testing_mode='simple')
        elif mode == param.mode_list[7]:
            Testing(testing_mode='simple_duc')
        else:
            print('Please specify the mode')
        print('\nTime: {}\t({:.3f}sec)'.format((datetime.timedelta(seconds=time.time() - start_time)), time.time() - start_time))

if __name__ == "__main__":
    Main()
