import tensorflow as tf
import pickle
import numpy as np
import parameters as param
from pyrouge import Rouge155
import rouge
import traceback


class Validation:
    rouge_dict = dict()

    def __init__(self):
        _ = None

    def get_rouge_dict(self):
        return self.rouge_dict

    def rouge_scores_of_validation_set(self, sess, model_validation, int2word_dict, batch_size, file_prefix,
                                       official_perl_rouge=True):
        system_summaries_file_path = self.predict_system_summaries(sess,
                                                                   model_validation,
                                                                   param.validation_data_dir + 'x_validation.pickle',
                                                                   param.validation_system_summaries_dir,
                                                                   param.validation_system_summaries_filename_id,
                                                                   int2word_dict,
                                                                   batch_size,
                                                                   file_prefix)

        rouge_scores_dict = dict()
        if official_perl_rouge:
            self.make_rouge_files(system_summaries_file_path=system_summaries_file_path,
                                  model_summaries_file_path=param.validation_summary_file_path,
                                  rouge_system_summaries_dir=param.validation_rouge_system_summaries_dir,
                                  rouge_model_summaries_dir=param.validation_rouge_model_summaries_dir)

            rouge_scores_dict = \
                self.rouge_scores(system_summaries_dir=param.validation_rouge_system_summaries_dir,
                                  model_summaries_dir=param.validation_rouge_model_summaries_dir,
                                  evaluation_results_dir=param.validation_rouge_results_dir,
                                  evaluation_results_filename=param.validation_rouge_results_filename_suffix,
                                  file_id_str=file_prefix)
        else:
            rouge_scores_dict = \
                self.python_pyrouge_scores(system_summaries_file_path=system_summaries_file_path,
                                           model_summaries_file_path=param.validation_summary_file_path,
                                           evaluation_results_dir=param.validation_rouge_results_dir,
                                           evaluation_results_filename=param.validation_rouge_results_filename_suffix,
                                           file_id_str=file_prefix)
        return rouge_scores_dict




    @staticmethod
    # using official perl rouge implementation
    def rouge_scores(system_summaries_dir,
                     model_summaries_dir,
                     evaluation_results_dir,
                     evaluation_results_filename, file_id_str):
        rouge = Rouge155()
        rouge.system_dir = system_summaries_dir
        rouge.model_dir = model_summaries_dir
        rouge.system_filename_pattern = 'system_summary.(\d+).txt'
        rouge.model_filename_pattern = 'model_summary.[A-Z].#ID#.txt'
        results = rouge.convert_and_evaluate()
        # print(results)
        rouge_output_dict = rouge.output_to_dict(results)
        results_filename = file_id_str + '_' + evaluation_results_filename
        results_file_path = evaluation_results_dir + results_filename
        results_file = open(results_file_path, 'w+', encoding='utf8')
        results_file.write(results_filename + '\n\nEvaluation results: \n')
        results_file.write(results +
                           '\n\nEvaluation results in dictionary form:\n---------------------------------------------\n')
        for k, v in zip(rouge_output_dict.keys(), rouge_output_dict.values()):
            results_file.write(str(k) + ': ' + str(v) + '\n')
            # print(str(k) + ': ' + str(v))
        results_file.close()
        return rouge_output_dict

    @staticmethod
    # using full python rouge implementation with py-rouge 1.1
    def python_pyrouge_scores(system_summaries_file_path,
                              model_summaries_file_path,
                              evaluation_results_dir,
                              evaluation_results_filename, file_id_str):
        hypothesis_list = []
        references_list = []
        line_index = 0  # line of both files
        system_summaries_file = open(system_summaries_file_path, 'r', encoding='utf8')
        model_summaries_file = open(model_summaries_file_path, 'r', encoding='utf8')
        for system_summary_line, model_summary_line in zip(system_summaries_file, model_summaries_file):
            line_index += 1
            hypothesis_list.append(system_summary_line)
            references_list.append(model_summary_line)
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                max_n=2,
                                limit_length=False,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=True,
                                apply_best=False,
                                alpha=0.5,  # Default F1_score
                                weight_factor=1.2,
                                stemming=True,
                                ensure_compatibility=True)
        scores = evaluator.get_scores(hypothesis_list, references_list)
        rouge_output_dict = dict()
        for metric, results in scores.items():
            ############################################
            # print('\t{}:\tP: {:5f}\tR: {:5f}\tF1: {:5f}'.format(metric, results['p'], results['r'], results['f']))
            if metric == 'rouge-1':
                rouge_output_dict['rouge_1_f_score'] = results['f']
            elif metric == 'rouge-2':
                rouge_output_dict['rouge_2_f_score'] = results['f']
            elif metric == 'rouge-l':
                rouge_output_dict['rouge_l_f_score'] = results['f']
        results_filename = file_id_str + '_' + evaluation_results_filename
        results_file_path = evaluation_results_dir + results_filename
        results_file = open(results_file_path, 'w+', encoding='utf8')
        results_file.write(
            '\n\nEvaluation results in dictionary form:\n---------------------------------------------\n')
        for k, v in zip(rouge_output_dict.keys(), rouge_output_dict.values()):
            results_file.write(str(k) + ': ' + str(v) + '\n')
            # print(str(k) + ': ' + str(v))
        results_file.close()
        return rouge_output_dict

    def predict_system_summaries(self, sess, model_validation, validation_x_data_path,
                                 system_summaries_file_dir, system_summaries_filename,
                                 int2word_dict, batch_size, file_prefix=''):
        validation_x = self.read_pickle_file(validation_x_data_path)
        batches = self.batch_iter(validation_x, [0] * len(validation_x), batch_size, 1)
        system_summaries_file_path = system_summaries_file_dir + str(file_prefix) + '_' + system_summaries_filename
        f = open(system_summaries_file_path, "w+")
        print("", end="", file=f)  # clear file

        for batch_x, _ in batches:
            batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))

            valid_feed_dict = {
                model_validation.batch_size: len(batch_x),
                model_validation.X: batch_x,
                model_validation.X_len: batch_x_len,
            }

            prediction = sess.run(model_validation.prediction, feed_dict=valid_feed_dict)
            prediction_output = \
                list(map(lambda x: [int2word_dict.get(y, int2word_dict[1]) for y in x], prediction[:, 0, :]))
            for line in prediction_output:
                summary = list()
                prev_word = ''
                for word in line:
                    if word == "</S>":
                        break
                    if word is not prev_word:
                        summary.append(word)
                    prev_word = word
                print(" ".join(summary), file=f)
        f.close()
        return system_summaries_file_path

    @staticmethod
    def make_rouge_files(system_summaries_file_path,
                         model_summaries_file_path,
                         rouge_system_summaries_dir,
                         rouge_model_summaries_dir):
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
            system_summary_file.write(system_summary_line)
            model_summary_file.write(model_summary_line)
            system_summary_file.close()
            model_summary_file.close()
        print(str(line_index) + " files have been created for system and model summaries")

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

    def load_validation_data_v2(self, validation_articles_path):
        return self.read_pickle_file(validation_articles_path)

    def load_validation_data(self, validation_articles_path,
                             word2int_dict_path, int2word_dict_path,
                             train_max_len_file_path,
                             validation_max_len_file_path,
                             debug=False):
        x = self.read_pickle_file(validation_articles_path)
        # y = self.read_pickle_file(self.test_data_path + 'y.pickle')
        word2int_dict = self.read_pickle_file(word2int_dict_path)
        int2word_dict = self.read_pickle_file(int2word_dict_path)
        train_maxlen = self.read_pickle_file(train_max_len_file_path)
        validation_articles_maxlen = self.read_pickle_file(validation_max_len_file_path)
        summary_max_len = train_maxlen['summary_max_len']
        vocabulary_len = train_maxlen['vocabulary_len']
        validation_article_max_len = validation_articles_maxlen['validataion_article_max_len']
        if debug:
            for a in x:
                print(a)
            print(word2int_dict)
            print(int2word_dict)
            print(validation_article_max_len)
            print(summary_max_len)
            print(str(vocabulary_len))
        return x, word2int_dict, int2word_dict, validation_article_max_len, summary_max_len, vocabulary_len

    @staticmethod
    def read_pickle_file(path_to_pickle_file):
        with open(path_to_pickle_file, "rb") as f:
            b = pickle.load(f)
        return b


if __name__ == "__main__":
    Validation()