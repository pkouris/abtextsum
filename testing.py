import tensorflow as tf
import pickle
import time
import numpy as np
import datetime
import parameters as param
from model import Model
from pyrouge import Rouge155
import os
from build_dataset import BuildDataset
import string


class Testing:

    def __init__(self, testing_mode='rouge_of_individual_files'):
        _ = None
        start_time = time.time()


        if testing_mode == 'simple':
            # '''
            system_summaries_file_path = \
                self.generate_system_summaries(param.mode, param.model_name, param.model_dir,
                                               param.test_x_file_path,
                                               param.word2int_file_path,
                                               param.int2word_file_path,
                                               param.maxlen_file_path,
                                               param.test_maxlen_file_path,
                                               param.test_system_summaries_dir,
                                               param.test_system_summaries_file_suffix,
                                               param.batch_size)

            #if param.training_on_collocations:
            #    self.remove_collocation_symbol_from_file(system_summaries_file_path)

            t = time.time()
            time_generate_system_summaries = t - start_time
            # '''
            self.make_rouge_files(system_summaries_file_path=system_summaries_file_path,
                                  model_summaries_file_path=param.test_summary_file_path,
                                  rouge_files_dir=param.test_rouge_files_dir,
                                  rouge_system_summaries_dir=param.test_rouge_system_summaries_dir,
                                  rouge_model_summaries_dir=param.test_rouge_model_summaries_dir)
            # '''
            time_make_rouge_files = time.time() - t

            self.rouge_scores(param.model_name,
                              system_summaries_file_path,
                              param.test_summary_file_path,
                              param.test_rouge_system_summaries_dir,
                              param.test_rouge_model_summaries_dir,
                              param.test_rouge_results_dir,
                              param.test_rouge_results_file_suffix,
                              time_generate_system_summaries, time_make_rouge_files, start_time)

            self.remove_rouge_files(rouge_system_summaries_dir=param.test_rouge_system_summaries_dir,
                                    rouge_model_summaries_dir=param.test_rouge_model_summaries_dir)
        if testing_mode == 'simple_duc':
            # '''
            system_summaries_file_path = \
                self.generate_system_summaries(param.mode, param.model_id, param.model_dir,
                                               param.test_duc_x_file_path,
                                               param.word2int_file_path,
                                               param.int2word_file_path,
                                               param.maxlen_file_path,
                                               param.test_duc_maxlen_file_path,
                                               param.test_duc_system_summaries_dir,
                                               param.test_duc_system_summaries_file_suffix,
                                               param.batch_size)

            # '''

           # if param.training_on_collocations:
           #     self.remove_collocation_symbol_from_file(system_summaries_file_path)

            t = time.time()
            time_generate_system_summaries = t - start_time
            # '''
            self.make_rouge_files_for_duc(system_summaries_file_path=system_summaries_file_path,
                                          model_summaries_file_path_list=[param.test_duc_ref1, param.test_duc_ref2,
                                                                          param.test_duc_ref3, param.test_duc_ref4],
                                          rouge_files_dir=param.test_duc_rouge_files_dir,
                                          rouge_system_summaries_dir=param.test_duc_rouge_system_summaries_dir,
                                          rouge_model_summaries_dir=param.test_duc_rouge_model_summaries_dir)
            # '''
            time_make_rouge_files = time.time() - t

            self.rouge_scores(param.duc_model_id,
                              system_summaries_file_path,
                              param.test_duc_golden_summary_file_path_list,
                              param.test_duc_rouge_system_summaries_dir,
                              param.test_duc_rouge_model_summaries_dir,
                              param.test_duc_rouge_results_dir,
                              param.test_duc_rouge_results_file_suffix,
                              time_generate_system_summaries, time_make_rouge_files, start_time)

            self.remove_rouge_files(rouge_system_summaries_dir=param.test_duc_rouge_system_summaries_dir,
                                    rouge_model_summaries_dir=param.test_duc_rouge_model_summaries_dir)

        elif testing_mode == 'rouge_of_individual_files':
            temp_dir = 'C:/datasets/textsum/gigaword/temp/'
            system_summaries_file_path = 'docs/output_sum'

            model_summaries_file_path = 'docs/golden_sum'
            start_time = time.time()
            self.make_rouge_files(system_summaries_file_path=system_summaries_file_path,
                                  model_summaries_file_path=model_summaries_file_path,
                                  rouge_files_dir=temp_dir,
                                  rouge_system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                                  rouge_model_summaries_dir=temp_dir + 'rouge_model_summaries/')
            t1 = time.time()
            self.rouge_scores(model_id='m',
                              system_summaries_file_path=system_summaries_file_path,
                              model_summaries_file_path=model_summaries_file_path,
                              system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                              model_summaries_dir=temp_dir + 'rouge_model_summaries/',
                              evaluation_results_dir=temp_dir + 'rouge_results/',
                              evaluation_results_filename_suffix='results.txt',
                              time_generate_system_summaries=start_time, time_make_rouge_files=t1,
                              start_time=start_time)

            self.remove_rouge_files(
                rouge_system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                rouge_model_summaries_dir=temp_dir + 'rouge_model_summaries/')

        elif testing_mode == 'duc_rouge_of_individual_files':
            temp_dir = 'C:/datasets/textsum/gigaword/temp/'
            system_summaries_file_path = 'docs/output_sum'  # system_generalized summary
            start_time = time.time()
            self.make_rouge_files_for_duc(system_summaries_file_path=system_summaries_file_path,
                                          model_summaries_file_path_list=[param.test_baselineduc_ref1_file_path,
                                                                          param.test_baselineduc_ref2_file_path,
                                                                          param.test_baselineduc_ref3_file_path,
                                                                          param.test_baselineduc_ref4_file_path],
                                          rouge_files_dir=temp_dir,
                                          rouge_system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                                          rouge_model_summaries_dir=temp_dir + 'rouge_model_summaries/')
            t1 = time.time()
            self.rouge_scores(model_id=param.duc_model_id,
                              system_summaries_file_path=system_summaries_file_path,
                              model_summaries_file_path=[param.test_baselineduc_ref1_file_path,
                                                         param.test_baselineduc_ref2_file_path,
                                                         param.test_baselineduc_ref3_file_path,
                                                         param.test_baselineduc_ref4_file_path],
                              system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                              model_summaries_dir=temp_dir + 'rouge_model_summaries/',
                              evaluation_results_dir=temp_dir + 'rouge_results/',
                              evaluation_results_filename_suffix='results.txt',
                              time_generate_system_summaries=start_time, time_make_rouge_files=t1,
                              start_time=start_time)

            self.remove_rouge_files(
                rouge_system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                rouge_model_summaries_dir=temp_dir + 'rouge_model_summaries/')
        elif testing_mode == 'rouge_of_individual_files_capped_at_75bytes':
            temp_dir = 'C:/datasets/textsum/gigaword/temp/'
            system_summaries_file_path = 'docs/output_sum'  # system_generalized_summaries
            # 'test_subset_system_summaries/' \
            # 'lemmagener_test_subset_system_summaries.txt'
            model_summaries_file_path = 'docs/golden_sum'  # model summaries
            start_time = time.time()
            self.make_rouge_files_capped_at_75b(system_summaries_file_path=system_summaries_file_path,
                                                model_summaries_file_path=model_summaries_file_path,
                                                rouge_files_dir=temp_dir,
                                                rouge_system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                                                rouge_model_summaries_dir=temp_dir + 'rouge_model_summaries/')
            t1 = time.time()
            self.rouge_scores(model_id='m',
                              system_summaries_file_path=system_summaries_file_path,
                              model_summaries_file_path=model_summaries_file_path,
                              system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                              model_summaries_dir=temp_dir + 'rouge_model_summaries/',
                              evaluation_results_dir=temp_dir + 'rouge_results/',
                              evaluation_results_filename_suffix='results.txt',
                              time_generate_system_summaries=start_time, time_make_rouge_files=t1,
                              start_time=start_time)

            self.remove_rouge_files(
                rouge_system_summaries_dir=temp_dir + 'rouge_system_summaries/',
                rouge_model_summaries_dir=temp_dir + 'rouge_model_summaries/')

    @staticmethod
    def remove_collocation_symbol_from_file(file_path):
        output_line_list = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                output_line_list.append(line.replace('_', ' '))
            f.close()
        with open(file_path, 'w', encoding='utf8') as f:
            for line in output_line_list:
                f.write(line)
            f.close()

    @staticmethod
    def remove_rouge_files(rouge_system_summaries_dir, rouge_model_summaries_dir):
        for filename in os.listdir(rouge_system_summaries_dir):
            os.remove(rouge_system_summaries_dir + filename)
        for filename in os.listdir(rouge_model_summaries_dir):
            os.remove(rouge_model_summaries_dir + filename)
        print('Rouge files have been removed')

    @staticmethod
    # using official perl rouge implementation
    def rouge_scores(model_id,
                     system_summaries_file_path,
                     model_summaries_file_path,
                     system_summaries_dir,
                     model_summaries_dir,
                     evaluation_results_dir,
                     evaluation_results_filename_suffix,
                     time_generate_system_summaries, time_make_rouge_files, start_time):
        rouge_scores_start_time = time.time()

        print('rouge_scores()\n\tmodel_id: {}\n\tsystem_summaries_dir: {}\n\tmodel_summaries_dir: {}\n'
              '\tevaluation_results_dir: {}\n'
              '\tevaluation_results_filename_suffix: {}\n'.format(model_id,
                                                                  system_summaries_dir,
                                                                  model_summaries_dir,
                                                                  evaluation_results_dir,
                                                                  evaluation_results_filename_suffix))

        if not os.path.exists(system_summaries_dir):
            os.makedirs(system_summaries_dir)
        if not os.path.exists(model_summaries_dir):
            os.makedirs(model_summaries_dir)
        if not os.path.exists(evaluation_results_dir):
            os.makedirs(evaluation_results_dir)

        rouge = Rouge155()
        rouge.system_dir = system_summaries_dir
        rouge.model_dir = model_summaries_dir

        ss_list = os.listdir(system_summaries_dir)
        ms_list = os.listdir(model_summaries_dir)

        rouge.system_filename_pattern = 'system_summary.(\d+).txt'
        rouge.model_filename_pattern = 'model_summary.[A-Z].#ID#.txt'
        results = rouge.convert_and_evaluate()
        # print(results)
        rouge_output_dict = rouge.output_to_dict(results)
        now = datetime.datetime.now()
        # results_filename = file_prefix + '_' + evaluation_results_filename
        results_filename = str(now.strftime("%Y%m%d_%H%M_" + evaluation_results_filename_suffix))
        results_file_path = evaluation_results_dir + results_filename
        results_file = open(results_file_path, 'w+', encoding='utf8')

        print_str = '{}\n\n' \
                    'Testing model: {}\n' \
                    'Testing of {} system and {} model summary files:\n' \
                    '\tSystem summaries file: {}\n' \
                    '\tModel Summaries file: {}\n' \
                    '\tRouge system_summaries_dir: {}\n' \
                    '\tRouge model_summaries_dir: {}\n' \
                    ''.format(results_filename, model_id,
                              len(ss_list), len(ms_list),
                              system_summaries_file_path,
                              model_summaries_file_path,
                              system_summaries_dir, model_summaries_dir)

        results_file.write(print_str)
        print(print_str)

        print_str = '\n\nTesting results:\n {}'.format(results)
        print(print_str)
        results_file.write(print_str)
        results_file.write('\n\nTesting results in dictionary form:\n---------------------------------------------\n')
        for k, v in zip(rouge_output_dict.keys(), rouge_output_dict.values()):
            results_file.write(str(k) + ': ' + str(v) + '\n')
            # print(str(k) + ': ' + str(v))
        current_time = time.time()
        print_str = 'Rouge scores have been written to file: {}\n' \
                    '\nTime:\n' \
                    '\tgenerate system summaries: {}\n' \
                    '\tmake rouge files: {}\n' \
                    '\tcompute rouge scores: {}\n' \
                    '\ttotal: {}\n\n' \
                    'Process finished.'.format(results_file_path,
                                               datetime.timedelta(seconds=round(time_generate_system_summaries, 0)),
                                               datetime.timedelta(seconds=round(time_make_rouge_files, 0)),
                                               datetime.timedelta(
                                                   seconds=round(current_time - rouge_scores_start_time, 0)),
                                               datetime.timedelta(seconds=round(current_time - start_time, 0)))
        print(print_str)
        results_file.write(print_str)
        results_file.close()
        return rouge_output_dict

    @staticmethod
    def make_rouge_files_capped_at_75b(system_summaries_file_path,
                                       model_summaries_file_path,
                                       rouge_files_dir,
                                       rouge_system_summaries_dir,
                                       rouge_model_summaries_dir):

        print('make_rouge_files()\n\tsystem_summaries_file_path: {}'.format(system_summaries_file_path))
        print('\tmodel_summaries_file_path: {}'.format(model_summaries_file_path))
        print('\trouge_files_dir: {}'.format(rouge_files_dir))
        print('\trouge_system_summaries_dir: {}'.format(rouge_system_summaries_dir))
        print('\trouge_model_summaries_dir: {}\n'.format(rouge_model_summaries_dir))

        if not os.path.exists(rouge_files_dir):
            os.makedirs(rouge_files_dir)
        if not os.path.exists(rouge_system_summaries_dir):
            os.makedirs(rouge_system_summaries_dir)
        if not os.path.exists(rouge_model_summaries_dir):
            os.makedirs(rouge_model_summaries_dir)

        system_summaries_file = open(system_summaries_file_path, 'r', encoding='utf8')
        model_summaries_file = open(model_summaries_file_path, 'r', encoding='utf8')

        line_index = 0  # line of both files
        for system_summary_line, model_summary_line in zip(system_summaries_file, model_summaries_file):
            line_index += 1
            file_no_str = ''
            if line_index < 10:
                file_no_str = '00' + str(line_index)
            elif line_index < 100:
                file_no_str = '0' + str(line_index)
            else:
                file_no_str = str(line_index)
            system_summary_file = open(rouge_system_summaries_dir + 'system_summary.' + file_no_str + '.txt', 'w+',
                                       encoding='utf8')
            model_summary_file = open(rouge_model_summaries_dir + 'model_summary.A.' + file_no_str + '.txt', 'w+',
                                      encoding='utf8')
            # system_filename_pattern = 'some_name.(\d+).txt'
            # model_filename_pattern = 'some_name.[A-Z].#ID#.txt'
            system_summary_file.write(system_summary_line[:75])
            model_summary_file.write(model_summary_line[:75])
            system_summary_file.close()
            model_summary_file.close()
        print(str(line_index) + " files have been created for system and model summaries")

    @staticmethod
    def make_rouge_files_for_duc(system_summaries_file_path,
                                 model_summaries_file_path_list,
                                 rouge_files_dir,
                                 rouge_system_summaries_dir,
                                 rouge_model_summaries_dir):

        print('make_rouge_files()\n\tsystem_summaries_file_path: {}'.format(system_summaries_file_path))
        print('\tmodel_summaries_file_paths:')
        for f in model_summaries_file_path_list:
            print('\t{}'.format(f))
        print('\trouge_files_dir: {}'.format(rouge_files_dir))
        print('\trouge_system_summaries_dir: {}'.format(rouge_system_summaries_dir))
        print('\trouge_model_summaries_dir: {}\n'.format(rouge_model_summaries_dir))

        if not os.path.exists(rouge_files_dir):
            os.makedirs(rouge_files_dir)
        if not os.path.exists(rouge_system_summaries_dir):
            os.makedirs(rouge_system_summaries_dir)
        if not os.path.exists(rouge_model_summaries_dir):
            os.makedirs(rouge_model_summaries_dir)

        system_summaries_file = open(system_summaries_file_path, 'r', encoding='utf8')
        # model_summaries_file = open(model_summaries_file_path, 'r', encoding='utf8')

        line_index = 0
        for system_summary_line in system_summaries_file:
            line_index += 1
            file_no_str = ''
            if line_index < 10:
                file_no_str = '00' + str(line_index)
            elif line_index < 100:
                file_no_str = '0' + str(line_index)
            else:
                file_no_str = str(line_index)
            system_summary_file = open(rouge_system_summaries_dir + 'system_summary.' + file_no_str + '.txt', 'w+',
                                       encoding='utf8')
            # system_filename_pattern = 'some_name.(\d+).txt'
            # model_filename_pattern = 'some_name.[A-Z].#ID#.txt'
            system_summary_file.write(system_summary_line)  # [:75]
            system_summary_file.close()
        line_index = 0
        alphabet_list = list(string.ascii_uppercase)
        for model_summaries_file_path, char in zip(model_summaries_file_path_list, alphabet_list):
            model_summaries_file = open(model_summaries_file_path, 'r', encoding='utf8')
            for model_summary_line in model_summaries_file:
                line_index += 1
                file_no_str = ''
                if line_index < 10:
                    file_no_str = '00' + str(line_index)
                elif line_index < 100:
                    file_no_str = '0' + str(line_index)
                else:
                    file_no_str = str(line_index)
                model_summary_file = open(
                    rouge_model_summaries_dir + 'model_summary.{}.'.format(char) + file_no_str + '.txt', 'w+',
                    encoding='utf8')
                # system_filename_pattern = 'some_name.(\d+).txt'
                # model_filename_pattern = 'some_name.[A-Z].#ID#.txt'
                model_summary_file.write(model_summary_line)  # [:75]
                model_summary_file.close()
        print(str(line_index) + " files have been created for system and model summaries")

    @staticmethod
    def make_rouge_files(system_summaries_file_path,
                         model_summaries_file_path,
                         rouge_files_dir,
                         rouge_system_summaries_dir,
                         rouge_model_summaries_dir):

        print('make_rouge_files()\n\tsystem_summaries_file_path: {}'.format(system_summaries_file_path))
        print('\tmodel_summaries_file_path: {}'.format(model_summaries_file_path))
        print('\trouge_files_dir: {}'.format(rouge_files_dir))
        print('\trouge_system_summaries_dir: {}'.format(rouge_system_summaries_dir))
        print('\trouge_model_summaries_dir: {}\n'.format(rouge_model_summaries_dir))

        if not os.path.exists(rouge_files_dir):
            os.makedirs(rouge_files_dir)
        if not os.path.exists(rouge_system_summaries_dir):
            os.makedirs(rouge_system_summaries_dir)
        if not os.path.exists(rouge_model_summaries_dir):
            os.makedirs(rouge_model_summaries_dir)

        system_summaries_file = open(system_summaries_file_path, 'r', encoding='utf8')
        model_summaries_file = open(model_summaries_file_path, 'r', encoding='utf8')

        line_index = 0  # line of both files
        for system_summary_line, model_summary_line in zip(system_summaries_file, model_summaries_file):
            line_index += 1
            file_no_str = ''
            if line_index < 10:
                file_no_str = '00' + str(line_index)
            elif line_index < 100:
                file_no_str = '0' + str(line_index)
            else:
                file_no_str = str(line_index)
            system_summary_file = open(rouge_system_summaries_dir + 'system_summary.' + file_no_str + '.txt', 'w+',
                                       encoding='utf8')
            model_summary_file = open(rouge_model_summaries_dir + 'model_summary.A.' + file_no_str + '.txt', 'w+',
                                      encoding='utf8')
            # system_filename_pattern = 'some_name.(\d+).txt'
            # model_filename_pattern = 'some_name.[A-Z].#ID#.txt'
            system_summary_file.write(system_summary_line)
            model_summary_file.write(model_summary_line)
            system_summary_file.close()
            model_summary_file.close()
        print(str(line_index) + " files have been created for system and model summaries")

    def generate_system_summaries(self, mode, model_name, model_dir, x_test_pickle_file_path,
                                  word2int_dict_file_path, int2word_dict_file_path,
                                  maxlen_pickle_file_path, test_article_max_len_pickle_file_path,
                                  system_summaries_file_dir, system_summaries_filename_suffix,
                                  batch_size):

        start_time = time.time()
        if not os.path.exists(param.logs_dir):
            os.makedirs(param.logs_dir)
        if not os.path.exists(system_summaries_file_dir):
            os.makedirs(system_summaries_file_dir)

        model_checkpoint_filename = model_name + '_checkpoint.txt'
        now = datetime.datetime.now()
        time_id = str(now.strftime('%Y%m%d_%H%M'))
        system_summaries_file_path = '{}{}_{}_{}'.format(
            system_summaries_file_dir, time_id, model_name, system_summaries_filename_suffix)
        # now = datetime.datetime.now()
        logfile_name = str(now.strftime(model_name + '_%Y%m%d_%H%M_test_logs.txt'))
        logfile_path = param.logs_dir + logfile_name
        logfile_writer = open(logfile_path, 'w+', encoding='utf8')

        print_str = 'generate_system_summaries()\n\tmode: {}\n\tmodel_name: {}\n\tmodel_dir: {}\n' \
                    '\tx_test_pickle_file_path: {}\n\tword2int_dict_file_path: {}\n' \
                    '\tint2word_dict_file_path: {}\n\tmaxlen_pickle_file_path: {}\n' \
                    '\ttest_article_max_len_pickle_file_path: {}\n' \
                    '\tsystem_summaries_file_dir: {}\n\n'.format(mode, model_name, model_dir, x_test_pickle_file_path,
                                                                 word2int_dict_file_path, int2word_dict_file_path,
                                                                 maxlen_pickle_file_path,
                                                                 test_article_max_len_pickle_file_path,
                                                                 system_summaries_file_dir)

        logfile_writer.write(print_str)
        print(print_str, end='')

        parameters_str = '\tstart_epoch_no = ' + str(param.start_epoch_no) + '\n' + \
                         '\tepochs_num = ' + str(param.epochs_num) + '\n' + \
                         '\tbatch_size = ' + str(param.batch_size) + '\n' + \
                         '\tembedding_dim = ' + str(param.embedding_dim) + '\n' + \
                         '\thidden_dim = ' + str(param.hidden_dim) + '\n' + \
                         '\tlayers_num = ' + str(param.layers_num) + '\n' + \
                         '\tlearning_rate = ' + str(param.learning_rate) + '\n' + \
                         '\tbeam_width = ' + str(param.beam_width) + '\n' + \
                         '\tkeep_prob = ' + str(param.keep_prob) + '\n' + \
                         '\tforward_only = ' + str(True) + '\n' + \
                         '\tusing_word2vec_embeddings = ' + str(param.using_word2vec_embeddings) + '\n' + \
                         '\tword_embeddings = ' + str(param.word_embendings) + '\n' + \
                         '\ttrain_restore_saved_model = ' + str(param.train_restored_saved_model) + '\n'

        print_str = 'Logs\nLogs File: {}\nMode: {}\nModel: {}\nstarting: {}\nParameters:\n{}\n' \
                    'Loading vocabulary and testing dataset...\n' \
                    ''.format(logfile_name, mode, model_name, now.strftime("%Y-%m-%d %H:%M"), parameters_str)
        print(print_str, end='')
        logfile_writer.write(print_str)

        test_x, word2int_dict, int2word_dict, article_max_len, summary_max_len, vocabulary_size = \
            self.load_testing_data(x_test_pickle_file_path,
                                   word2int_dict_file_path, int2word_dict_file_path,
                                   maxlen_pickle_file_path, test_article_max_len_pickle_file_path,
                                   debug=False)

        print_str = 'x shape:{}\nvocabulary_size: {}\n' \
                    'Article and summary max_len: {}, {}\n'.format(np.array(test_x).shape, vocabulary_size,
                                                                   article_max_len, summary_max_len)
        print(print_str, end='')
        logfile_writer.write(print_str)

        with tf.Session() as sess:
            logfile_writer.write("\nLoading saved model...\n")
            print("Loading saved model...")

            model_test = Model(article_max_len, summary_max_len,
                               param.embedding_dim, param.hidden_dim, param.layers_num,
                               param.learning_rate, param.beam_width, param.keep_prob, vocabulary_size,
                               batch_size, [], forward_only=True,
                               using_word2vec_embeddings=False)

            print('Restoring the saved model.')
            logfile_writer.write('Restoring the saved model.\n')
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_dir,
                                                 latest_filename=model_checkpoint_filename)
            print_str = 'model_dir: {}\ncheckpoint_filename: {}\n' \
                        'checkpoint_path: {}\n'.format(model_dir, model_checkpoint_filename, ckpt.model_checkpoint_path)

            print(print_str)
            logfile_writer.write(print_str)
            saver.restore(sess, ckpt.model_checkpoint_path)

            batches = self.batch_iter(test_x, [0] * len(test_x), batch_size, 1)

            with open(system_summaries_file_path, "w+") as f:
                print("", end="", file=f)  # clear file
            print_str = 'Writing summaries to file: {}\n'.format(system_summaries_file_path)
            logfile_writer.write(print_str)
            print(print_str, end='')

            for batch_x, _ in batches:
                batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))

                valid_feed_dict = {
                    model_test.batch_size: len(batch_x),
                    model_test.X: batch_x,
                    model_test.X_len: batch_x_len,
                }

                prediction = sess.run(model_test.prediction, feed_dict=valid_feed_dict)
                prediction_output = list(map(lambda x: [int2word_dict[y] for y in x], prediction[:, 0, :]))

                with open(system_summaries_file_path, "a") as f:
                    for line in prediction_output:
                        summary = list()
                        prev_word = ''
                        for word in line:
                            if word == "</S>":
                                break
                            # if word not in summary:
                            if word is not prev_word:
                                summary.append(word)
                            prev_word = word
                        print(" ".join(summary), file=f)

            print_str = 'Summaries have been saved\n\nTime: {}\n\n.'.format(
                datetime.timedelta(seconds=time.time() - start_time))
            logfile_writer.write(print_str)
            print(print_str, end='')
            logfile_writer.close()
        sess.close()
        return system_summaries_file_path

    @staticmethod
    def batch_iter(inputs, outputs, batch_size, num_epochs):
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        len_inputs = len(inputs)
        num_batches_per_epoch = (len_inputs - 1) // batch_size + 1
        for epoch in range(num_epochs):
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(inputs))
                yield inputs[start_index:end_index], outputs[start_index:end_index]

    def load_testing_data(self, x_test_pickle_file_path,
                          word2int_dict_file_path, int2word_dict_file_path,
                          maxlen_pickle_file_path, test_article_max_len_pickle_file_path,
                          debug=False):
        x = self.read_pickle_file(x_test_pickle_file_path)
        word2int_dict = self.read_pickle_file(word2int_dict_file_path)
        int2word_dict = self.read_pickle_file(int2word_dict_file_path)
        maxlen = self.read_pickle_file(maxlen_pickle_file_path)
        maxlen_test = self.read_pickle_file(test_article_max_len_pickle_file_path)
        summary_max_len = maxlen['summary_max_len']
        vocabulary_len = maxlen['vocabulary_len']
        test_article_max_len = maxlen_test['test_article_max_len']
        if debug:
            for a in x:
                print(a)
            print(word2int_dict)
            print(int2word_dict)
            print(test_article_max_len)
            print(summary_max_len)
            print(str(vocabulary_len))
        return x, word2int_dict, int2word_dict, test_article_max_len, summary_max_len, vocabulary_len

    @staticmethod
    def read_pickle_file(path_to_pickle_file):
        with open(path_to_pickle_file, "rb") as f:
            b = pickle.load(f)
        return b


if __name__ == '__main__':
    Testing(testing_mode='rouge_of_individual_files')
