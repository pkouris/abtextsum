import tensorflow as tf
import argparse
import pickle
import os
import re
import sys, traceback
import numpy as np
import time
import datetime
from matplotlib.pyplot import figure, ylabel, tight_layout, plot, savefig, tick_params, xlabel, subplot, title, close
from model import Model
import parameters as param
import os.path
import random
from sklearn.utils import shuffle
from validation import Validation
import signal
import sys
import paths



class Train:
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default="", help="\nmodel values: lg100d5, lg200d5, neg100, neg200...\n"
                                                   "e.g. python train.py -model neg100")

    def get_args(self):
        return self.parser.parse_args()

    def __init__(self):
        args = self.get_args()
        self.train(x_chunks_dir=paths.x_dir,
                   y_chunks_dir=paths.y_dir,
                   # training_size=param.training_size,
                   model_name=param.model_name,
                   logs_dir=paths.logs_dir,
                   mode='train',
                   batch_size=param.batch_size,
                   model_dir=paths.model_dir,
                   official_perl_rouge=paths.validation_official_perl_rouge,
                   start_epoch_no=param.start_epoch_no,
                   #training_on_collocations=param.training_on_collocations
                   )

    parameters_str = '\tstart_epoch_no = ' + str(param.start_epoch_no) + '\n' + \
                     '\tepochs_num = ' + str(param.epochs_num) + '\n' + \
                     '\tbatch_size = ' + str(param.batch_size) + '\n' + \
                     '\tembedding_dim = ' + str(param.embedding_dim) + '\n' + \
                     '\thidden_dim = ' + str(param.hidden_dim) + '\n' + \
                     '\tlayers_num = ' + str(param.layers_num) + '\n' + \
                     '\tlearning_rate = ' + str(param.learning_rate) + '\n' + \
                     '\tbeam_width = ' + str(param.beam_width) + '\n' + \
                     '\tkeep_prob = ' + str(param.keep_prob) + '\n' + \
                     '\tforward_only = ' + str(False) + '\n' + \
                     '\tusing_word2vec_embeddings = ' + str(param.using_word2vec_embeddings) + '\n' + \
                     '\tword_embeddings = ' + str(paths.word_embendings) + '\n' + \
                     '\ttrain_restore_saved_model = ' + str(param.train_restored_saved_model) + '\n'

    def train(self, x_chunks_dir, y_chunks_dir, model_name, logs_dir, mode, batch_size, model_dir,
              official_perl_rouge=True, start_epoch_no=1,  training_on_collocations=False):
        total_start_time = time.time()
        avg_loss_per_batches_list = []  # list of avg loss per a number of batches
        avg_loss_per_epoch = []
        rouge1_per_epoch_list = []
        rouge2_per_epoch_list = []
        rougeL_per_epoch_list = []
        best_loss = {'loss': 999888, 'epoch': 0, 'batch': 0}  # a huge initial value of best_loss
        best_rouge1_f1 = {'rouge': 0.0, 'epoch': 0, 'batch': 0}
        best_rouge2_f1 = {'rouge': 0.0, 'epoch': 0, 'batch': 0}
        best_rougeL_f1 = {'rouge': 0.0, 'epoch': 0, 'batch': 0}

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_checkpoint_filename = model_name + '_checkpoint.txt'
        now = datetime.datetime.now()
        logfile_name = str(now.strftime(model_name + '_%Y%m%d_%H%M_train_logs.txt'))
        logfile_path = logs_dir + logfile_name
        logfile_writer_e = None
        logfile_writer_w = open(logfile_path, 'w', encoding='utf8')
        sess = None
        try:
            print_str = 'Logs\nLogs File: {}\nMode: {}\nModel: {}\nstarting: {}\nParameters:\n{}\n' \
                        'Loading dictionary and training dataset...\n' \
                        ''.format(logfile_name, mode, model_name, now.strftime("%Y-%m-%d %H:%M"), self.parameters_str)
            print(print_str, end='')
            logfile_writer_w.write(print_str)

            word2int_dict, int2word_dict, article_max_len, summary_max_len, vocabulary_size = self.load_training_data()
            summary_max_len += 1  # because of adding </S> or <S>
            x_and_y_file_path_pairs_list = self.file_path_pairs(x_chunks_dir=x_chunks_dir, y_chunks_dir=y_chunks_dir)

            num_of_chunks = len(x_and_y_file_path_pairs_list)
            num_of_batches_per_epoch = self.num_of_batches_per_epoch(x_and_y_file_path_pairs_list, batch_size)

            print_str = 'Batches per epoch: {}\n' \
                        'vocabulary_size: {}\n' \
                        'Article and summary max_len: {}, {}\n'.format(num_of_batches_per_epoch, vocabulary_size,
                                                                       article_max_len, summary_max_len)
            print(print_str, end='')
            logfile_writer_w.write(print_str)

            sess = tf.Session()  # as sess:

            print("Loading word2vec...")
            word2vec_embeddings = Model.get_init_embedding(int2word_dict, param.embedding_dim,
                                                           paths.word2vec_file_path)
            print_str = 'Word embeddings have been loaded.\n'
            print(print_str, end='')
            logfile_writer_w.write(print_str)

            model = Model(article_max_len, summary_max_len,
                          param.embedding_dim, param.hidden_dim, param.layers_num,
                          param.learning_rate, param.beam_width, param.keep_prob, vocabulary_size,
                          batch_size, word2vec_embeddings, forward_only=False,
                          using_word2vec_embeddings=param.using_word2vec_embeddings)

            model_validation = Model(article_max_len, summary_max_len,
                                     param.embedding_dim, param.hidden_dim, param.layers_num,
                                     param.learning_rate, param.beam_width, param.keep_prob, vocabulary_size,
                                     batch_size, word2vec_embeddings, forward_only=True,
                                     using_word2vec_embeddings=False)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

            if param.train_restored_saved_model and os.path.exists(model_dir + model_checkpoint_filename):
                print('Restoring the saved model.')
                logfile_writer_w.write('Restoring the saved model.\n')
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=paths.model_dir,
                                                     latest_filename=model_checkpoint_filename)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Training of new model.')
                logfile_writer_w.write('Training of new model.\n')
            logfile_writer_w.close()
            epoch = start_epoch_no
            epoch_no = start_epoch_no
            end_epoch_no = epoch_no + param.epochs_num - 1
            remaining_epochs = end_epoch_no - start_epoch_no
            for _ in range(param.epochs_num):
                end_epoch_no = epoch + remaining_epochs
                remaining_epochs -= 1
                logfile_writer_e = open(logfile_path, 'a', encoding='utf8')
                epoch_loss_list = []
                epoch_start_time = time.time()
                print_str = '\n{} of {} epoch\n'.format(epoch, end_epoch_no)
                print(print_str, end='')
                logfile_writer_e.write(print_str)

                x_and_y_file_path_pairs_list = shuffle(x_and_y_file_path_pairs_list,
                                                       random_state=round(time.time() // 12345))
                chunk_index = 0
                for (x_file_path, y_file_path) in x_and_y_file_path_pairs_list:
                    #chunk_start_time = time.time()
                    chunk_index += 1
                    print_str = '\t{} of {} chunk (files: {} & {})\n'.format(chunk_index, num_of_chunks,
                                                                             x_file_path[-60:], y_file_path[-60:])
                    print(print_str, end='')
                    logfile_writer_e.write(print_str)

                    train_x, train_y, _ = self.load_and_shuffle_training_set(x_file_path, y_file_path)
                    # if param.training_size is not None:
                    # remaining_training_size = remaining_training_size - current_training_size

                    print_str = '\t\tx & y chunk shape: {}, {}\n'.format(np.array(train_x).shape,
                                                                         np.array(train_y).shape)
                    logfile_writer_e.write(print_str)
                    print(print_str, end='')

                    batches = self.batch_iter_v2(train_x, train_y, batch_size)
                    num_batches_of_chunk = (len(train_x) - 1) // batch_size + 1

                    print("\t\tBatches of chunk: {}".format(num_batches_of_chunk))
                    # print("Iteration starts.")

                    logfile_writer_e.write("\t\tBatches of chunk: {}\n".format(num_batches_of_chunk))
                    # logfile_writer.write("\nIteration starts.\n")

                    batch_index = 0
                    batches_start_time = time.time()
                    batches_loss_list = []
                    for batch_x, batch_y in batches:
                        batch_index += 1
                        batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
                        batch_decoder_input = list(
                            map(lambda x: [word2int_dict["<S>"]] + list(x), batch_y))  # y starts with <s>
                        batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
                        batch_decoder_output = list(
                            map(lambda x: list(x) + [word2int_dict["</S>"]], batch_y))  # y ends with </s>

                        batch_decoder_input = list(
                            map(lambda d: d + (summary_max_len - len(d)) * [word2int_dict["<PAD>"]],
                                batch_decoder_input))
                        batch_decoder_output = list(
                            map(lambda d: d + (summary_max_len - len(d)) * [word2int_dict["<PAD>"]],
                                batch_decoder_output))

                        train_feed_dict = {
                            model.batch_size: len(batch_x),
                            model.X: batch_x,
                            model.X_len: batch_x_len,
                            model.decoder_input: batch_decoder_input,
                            model.decoder_len: batch_decoder_len,
                            model.decoder_target: batch_decoder_output
                        }


                        fetches = [model.update, model.global_step, model.loss]
                        _, batch_no, loss = sess.run(fetches, feed_dict=train_feed_dict)
                        epoch_loss_list.append(loss)
                        batches_loss_list.append(loss)
                        if batch_no % param.print_loss_per_steps == 0:
                            avg_batch_loss = np.average(np.array(batches_loss_list))
                            avg_loss_per_batches_list.append(avg_batch_loss)
                            batches_loss_list = []
                            print_str = "\t\tEpoch: {} Batch: {}, Loss: {:.5f}, " \
                                        "Time (batches, epoch & total): {}, {} & {}\n" \
                                        "".format(epoch, batch_no, avg_batch_loss,
                                                  datetime.timedelta(
                                                      seconds=round(time.time() - batches_start_time, 0)),
                                                  datetime.timedelta(
                                                      seconds=round(time.time() - epoch_start_time, 0)),
                                                  datetime.timedelta(
                                                      seconds=round(time.time() - total_start_time, 0)))
                            print(print_str, end='')
                            logfile_writer_e.write(print_str)
                            batches_start_time = time.time()

                        # end of epoch --> saving the best model and print validation values
                        if batch_no % num_of_batches_per_epoch == 0:  # end of epoch
                            epoch = batch_no // num_of_batches_per_epoch
                            temp = np.array(epoch_loss_list)
                            avg_epoch_loss = np.average(temp)
                            del temp
                            avg_loss_per_epoch.append(avg_epoch_loss)
                            file_prefix = '{}_{}epoch_{}batch'.format(
                                model_name, self.int_to_two_digits_str(epoch), batch_no)
                            validation = Validation()
                            rouge_scores_dict = \
                                validation.rouge_scores_of_validation_set(sess,
                                                                          model_validation,
                                                                          int2word_dict,
                                                                          batch_size,
                                                                          file_prefix=file_prefix,
                                                                          official_perl_rouge=official_perl_rouge,
                                                                          #training_on_collocations=training_on_collocations
                                                                          )
                            rouge1_per_epoch_list.append(rouge_scores_dict['rouge_1_f_score'])
                            rouge2_per_epoch_list.append(rouge_scores_dict['rouge_2_f_score'])
                            rougeL_per_epoch_list.append(rouge_scores_dict['rouge_l_f_score'])
                            saving_flag = False

                            if rouge_scores_dict['rouge_1_f_score'] > best_rouge1_f1['rouge']:
                                best_rouge1_f1['rouge'] = rouge_scores_dict['rouge_1_f_score']
                                best_rouge1_f1['epoch'] = epoch
                                best_rouge1_f1['batch'] = batch_no
                                saving_flag = True
                            if rouge_scores_dict['rouge_2_f_score'] > best_rouge2_f1['rouge']:
                                best_rouge2_f1['rouge'] = rouge_scores_dict['rouge_2_f_score']
                                best_rouge2_f1['epoch'] = epoch
                                best_rouge2_f1['batch'] = batch_no
                                saving_flag = True
                            if rouge_scores_dict['rouge_l_f_score'] > best_rougeL_f1['rouge']:
                                best_rougeL_f1['rouge'] = rouge_scores_dict['rouge_l_f_score']
                                best_rougeL_f1['epoch'] = epoch
                                best_rougeL_f1['batch'] = batch_no
                                saving_flag = True
                            if avg_epoch_loss < best_loss['loss']:
                                best_loss['loss'] = avg_epoch_loss
                                best_loss['epoch'] = epoch
                                best_loss['batch'] = batch_no
                                saving_flag = True
                                # Save all variables of the TensorFlow graph to file.

                            print_str = 'Model: {}\n' \
                                '   Avg_epoch_loss  (current & best): {:.5f} & {:.5f} (epoch {}, batch {})\n' \
                                '   Rouge_1_f-score (current & best): {:.5f} & {:.5f} (epoch {}, batch {})\n' \
                                '   Rouge_2_f-score (current & best): {:.5f} & {:.5f} (epoch {}, batch {})\n' \
                                '   Rouge_L_f-score (current & best): {:.5f} & {:.5f} (epoch {}, batch {})\n' \
                                '   Time (epoch & total): {} & {}.\n'.format(model_name,
                                    avg_epoch_loss, best_loss['loss'], best_loss['epoch'], best_loss['batch'],
                                    rouge_scores_dict['rouge_1_f_score'],
                                    best_rouge1_f1['rouge'], best_rouge1_f1['epoch'], best_rouge1_f1['batch'],
                                    rouge_scores_dict['rouge_2_f_score'],
                                    best_rouge2_f1['rouge'], best_rouge2_f1['epoch'], best_rouge2_f1['batch'],
                                    rouge_scores_dict['rouge_l_f_score'],
                                    best_rougeL_f1['rouge'], best_rougeL_f1['epoch'], best_rougeL_f1['batch'],
                                    datetime.timedelta(seconds=round(time.time() - epoch_start_time, 0)),
                                    datetime.timedelta(seconds=round(time.time() - total_start_time, 0)))

                            if saving_flag:
                                current_model_name = model_name + '_' + self.int_to_two_digits_str(
                                    epoch) + 'epoch.ckpt'
                                # model_checkpoint_filename = model_name + 'checkpoint'
                                checkpoint = saver.save(sess=sess, save_path=model_dir + current_model_name,
                                                        latest_filename=model_checkpoint_filename,
                                                        global_step=batch_no)
                                add_str = '\nEpoch {}: The model is saved as the best one (...{}).\n' \
                                          ''.format(epoch, checkpoint[-25:])
                                print(add_str + print_str, end='')
                                logfile_writer_e.write(add_str + print_str)
                            else:
                                add_str = '\nEpoch {}: Τhe model was not saved as it is not the best one.\n' \
                                          ''.format(epoch)
                                print(add_str + print_str, end='')
                                logfile_writer_e.write(add_str + print_str)
                            epoch = epoch + 1
                logfile_writer_e.close()

        except KeyboardInterrupt:
            if logfile_writer_w is not None:
                logfile_writer_w.close()
            if logfile_writer_e is not None:
                logfile_writer_e.close()
            #print_keyboard_interrupt_str = '\nException: KeyboardInterrupt\n'
            logfile_writer = open(logfile_path, 'a', encoding='utf8')
            print_str = '\n' + "-" * 60 + '\nException: KeyboardInterrupt\n'
            print(print_str)
            logfile_writer.write(print_str)
            traceback.print_exc(file=sys.stdout)
            traceback.print_exc(file=logfile_writer)
            print_str = "-" * 60
            print(print_str)
            logfile_writer.write(print_str)
            logfile_writer.close()
        except Exception:
            if logfile_writer_w is not None:
                logfile_writer_w.close()
            if logfile_writer_e is not None:
                logfile_writer_e.close()
            logfile_writer = open(logfile_path, 'a', encoding='utf8')
            print_str = '\n' + "-" * 60 + '\n' + 'Exception:\n'
            print(print_str)
            logfile_writer.write(print_str)
            traceback.print_exc(file=sys.stdout)
            traceback.print_exc(file=logfile_writer)
            print_str = "-" * 60
            print(print_str)
            logfile_writer.write(print_str)
            logfile_writer.close()
        finally:
            self.finally_of_train_method(logfile_path, avg_loss_per_batches_list, avg_loss_per_epoch, rouge1_per_epoch_list,
                                         rouge2_per_epoch_list, rougeL_per_epoch_list, total_start_time, now)
            sess.close()

    # print text and plot charts for finally section of train method
    @staticmethod
    def finally_of_train_method(logfile_path, avg_loss_per_batches_list, avg_loss_per_epoch, rouge1_per_epoch_list,
                                rouge2_per_epoch_list, rougeL_per_epoch_list, total_start_time, now):
        # if logfile_writer is None:
        logfile_writer = open(logfile_path, 'a', encoding='utf8')

        print_str = '\nAvg Loss per batches: {}\n' \
                    'Avg loss per epoch: {}\n' \
                    'Rouge_1 F-score per epoch: {}\n' \
                    'Rouge_2 F-score per epoch: {}\n' \
                    'Rouge_L F-score per epoch: {}\n\n' \
                    'Total_time: {} hh:mm:ss\n\n' \
                    'Log file: {}\n' \
                    'Process finished\n'.format(avg_loss_per_batches_list, avg_loss_per_epoch, rouge1_per_epoch_list,
                                                rouge2_per_epoch_list, rougeL_per_epoch_list,
                                                datetime.timedelta(seconds=time.time() - total_start_time),
                                                logfile_path)
        print(print_str, end='')
        logfile_writer.write(print_str)
        logfile_writer.close()
        # Plot
        chart_file = paths.chart_dir + param.model_name + str(now.strftime("_%Y%m%d_%H%M_train_charts.pdf"))
        figure(1)
        subplot(311)
        plot(range(1, len(avg_loss_per_batches_list) + 1), avg_loss_per_batches_list)
        ylabel('Loss')
        xlabel('Batch')
        title('Avg loss per batches (batches: {} of size: {})'.format(param.print_loss_per_steps, param.batch_size))
        ax = subplot(312)
        r = range(1, len(rouge1_per_epoch_list) + 1)
        ax.plot(r, rouge1_per_epoch_list, color='blue', label='Rouge_1')
        ax.plot(r, rouge2_per_epoch_list, color='black', label='Rouge_2')
        ax.plot(r, rougeL_per_epoch_list, color='red', label='Rouge_L')
        ax.legend()
        ylabel('Rouge f-score')
        xlabel('Epoch')
        title('Rouge_1 per epoch')
        subplot(313)
        plot(range(1, len(avg_loss_per_epoch) + 1), avg_loss_per_epoch)
        title('Loss per epoch (epochs: ' + str(param.epochs_num) + ')')
        ylabel('Loss')
        xlabel('Epoch')
        tick_params()
        tight_layout()
        savefig(chart_file)

    @staticmethod
    def int_to_two_digits_str(number):
        if number < 10:
            return '0' + str(number)
        elif number < 100:
            return str(number)
        else:
            print("The number has more than three digits")
            return str(number)

    @staticmethod
    def file_path_pairs(x_chunks_dir, y_chunks_dir):
        x_chunked_filenames_list = os.listdir(x_chunks_dir)
        x_chunked_filenames_list = sorted(x_chunked_filenames_list, key=str.lower)
        y_chunked_filenames_list = os.listdir(y_chunks_dir)
        y_chunked_filenames_list = sorted(y_chunked_filenames_list, key=str.lower)
        file_path_pairs_list = []
        for x_filename, y_filename in zip(x_chunked_filenames_list, y_chunked_filenames_list):
            if str(x_filename[1:]) is not str(y_filename[1:]):
                if 'y' + str(x_filename[1:]) in y_chunked_filenames_list:
                    index = y_chunked_filenames_list.index('y' + str(x_filename[1:]))
                    y_filename = y_chunked_filenames_list[index]
                    file_path_pairs_list.append((x_chunks_dir + x_filename, y_chunks_dir + y_filename))
                else:
                    print('\t' + str(x_filename) + 'rejected because it does not corresponds to any y_file')
        return file_path_pairs_list

    def batch_iter_v2(self, inputs, outputs, batch_size):
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        len_inputs = len(inputs)
        num_of_batches = (len_inputs - 1) // batch_size + 1
        # for epoch in range(num_epochs):
        for batch_num in range(num_of_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]

    def load_training_data(self):
        word2int_dict = self.read_pickle_file(paths.train_data_dir + 'word2int_dict.pickle')
        int2word_dict = self.read_pickle_file(paths.train_data_dir + 'int2word_dict.pickle')
        maxlen = self.read_pickle_file(paths.train_data_dir + 'maxlen.pickle')
        # for k, b in zip(maxlen.keys(), maxlen.values):
        article_max_len = maxlen['article_max_len']
        summary_max_len = maxlen['summary_max_len']
        vocabulary_len = maxlen['vocabulary_len']
        return word2int_dict, int2word_dict, article_max_len, summary_max_len, vocabulary_len

    def load_and_shuffle_training_set(self, x_data_path, y_data_path, training_size=None):
        x = self.read_pickle_file(x_data_path)
        y = self.read_pickle_file(y_data_path)
        if training_size:
            x_new = []
            y_new = []
            counter = 1
            for i, j in zip(x, y):
                x_new.append(i)
                y_new.append(j)
                counter += 1
                if counter > training_size:
                    break
            x = x_new
            y = y_new
            del x_new, y_new
            training_size = counter - 1
        x, y = shuffle(x, y, random_state=round(time.time() // 12345))
        return x, y, training_size

    @staticmethod
    def read_pickle_file(path_to_pickle_file):
        with open(path_to_pickle_file, "rb") as f:
            b = pickle.load(f)
        return b

    @staticmethod
    def num_of_lines_of_file(file):
        count = 0
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                count += 1
        # print("Number of lines: ", str(count))
        return count

    def num_of_batches_per_epoch(self, x_and_y_file_path_pairs_list, batch_size):
        batches = 0
        for pair in x_and_y_file_path_pairs_list:
            x_temp = self.read_pickle_file(pair[0])
            size = len(x_temp)
            batches += len(x_temp) // batch_size
            if size % batch_size > 0:
                batches += 1
        # print('Butches: {}'.format(batches))
        return batches

    def num_of_instances_of_binary_files(self, files_list):
        count = 0
        for f in files_list:
            x_temp = self.read_pickle_file(f)
            count += len(x_temp)
        print('instances: {}'.format(count))
        return count
