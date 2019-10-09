import pickle
import parameters as param
import os
import argparse
import paths


class BuildDataset:
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default="", help="\nmode values: train, test, validation, duc_test\n"
                                                  "e.g. python build_dataset.py -mode train -model lg100d5g")
    parser.add_argument('-model', default="", help="\nmodel values: lg100d5, lg200d5, neg100, neg200...\n"
                                                   "e.g. python build_dataset.py -mode test -model neg200g")
    def get_args(self):
        return self.parser.parse_args()

    def __init__(self):
        args = self.get_args()
        #if args.mode == '':  # for running in IDE
        #    args.mode = 'test'
        #    args.model = 'bg'


        if args.mode == 'train':
            self.build_whole_train_dataset_and_dictionary(paths.train_data_dir,
                                                          paths.train_article_file_path,
                                                          paths.train_title_file_path,
                                                          paths.x_chunks_dir,
                                                          paths.y_chunks_dir,
                                                          paths.x_txt_dir,
                                                          paths.y_txt_dir,
                                                          paths.x_chunks_sample_dir,
                                                          paths.y_chunks_sample_dir,
                                                          paths.word2int_file_path, paths.word2int_txt_file_path,
                                                          paths.int2word_file_path, paths.int2word_txt_file_path,
                                                          paths.word_freq_file_path, paths.vocab_txt_file_path,
                                                          paths.maxlen_file_path, paths.maxlen_txt_file_path,
                                                          param.lines_per_chunk, param.read_lines)
        elif args.mode == 'validation':
            self.build_validation_or_testing_dataset(paths.validation_article_file_path,
                                                     paths.validation_data_dir,
                                                     paths.validation_rouge_results_dir,
                                                     paths.validation_system_summaries_dir,
                                                     paths.word2int_file_path,
                                                     paths.int2word_file_path,
                                                     paths.maxlen_file_path,
                                                     purpose='validation')
        elif args.mode == 'test':
            self.build_validation_or_testing_dataset(paths.test_article_file_path,
                                                     paths.test_data_dir,
                                                     paths.test_rouge_results_dir,
                                                     paths.test_system_summaries_dir,
                                                     paths.word2int_file_path,
                                                     paths.int2word_file_path,
                                                     paths.maxlen_file_path,
                                                     purpose='test')
        elif args.mode == 'duc_test':
            self.build_validation_or_testing_dataset(paths.test_duc_article_file_path,
                                                     paths.test_duc_data_dir,
                                                     paths.test_duc_rouge_results_dir,
                                                     paths.test_duc_system_summaries_dir,
                                                     paths.word2int_file_path,
                                                     paths.int2word_file_path,
                                                     paths.maxlen_file_path,
                                                     purpose='test')

        else:
            print("\nmode values: train, test, validation, duc_test\n"
                  "e.g. python build_dataset.py -mode train -model lg100d5")



    @staticmethod
    def num_of_lines_of_file(file):
        count = 0
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                count += 1
        # print("Number of lines: ", str(count))
        return count

    @staticmethod
    def view_some_lines(file=paths.train_article_file_path, from_line=1, to_line=5):
        with open(file, 'r', encoding='utf8') as f:
            line_index = 0
            for line in f:
                line_index += 1
                if from_line <= line_index <= to_line:
                    print(line)

    def build_whole_train_dataset_and_dictionary(self, train_data_dir, train_article_path, train_summary_path,
                                                 x_chunks_dir,
                                                 y_chunks_dir, x_txt_dir, y_txt_dir, x_sample_dir, y_sample_dir,
                                                 word2int_file_path, word2int_txt_file_path,
                                                 int2word_file_path, int2word_txt_file_path,
                                                 word_freq_file_path, vocab_txt_file_path,
                                                 maxlen_file_path, maxlen_txt_file_path,
                                                 lines_per_chunk, read_lines):

        if not os.path.exists(x_chunks_dir):
            os.makedirs(x_chunks_dir)
        else:
            for filename in os.listdir(x_chunks_dir):
                os.remove(x_chunks_dir + filename)
        if not os.path.exists(y_chunks_dir):
            os.makedirs(y_chunks_dir)
        else:
            for filename in os.listdir(y_chunks_dir):
                os.remove(y_chunks_dir + filename)
        if not os.path.exists(x_txt_dir):
            os.makedirs(x_txt_dir)
        else:
            for filename in os.listdir(x_txt_dir):
                os.remove(x_txt_dir + filename)
        if not os.path.exists(y_txt_dir):
            os.makedirs(y_txt_dir)
        else:
            for filename in os.listdir(y_txt_dir):
                os.remove(y_txt_dir + filename)
        if not os.path.exists(x_sample_dir):
            os.makedirs(x_sample_dir)
        else:
            for filename in os.listdir(x_sample_dir):
                os.remove(x_sample_dir + filename)
        if not os.path.exists(y_sample_dir):
            os.makedirs(y_sample_dir)
        else:
            for filename in os.listdir(y_sample_dir):
                os.remove(y_sample_dir + filename)
        print('Directories have been created or old chunks have been removed')
        print('Article file: {}'.format(train_article_path))
        print('Summary file: {}'.format(train_summary_path))
        article_max_len = 0
        summary_max_len = 0
        num_of_lines = self.num_of_lines_of_file(file=train_article_path)
        if num_of_lines > read_lines:
            num_of_lines = read_lines
        # words = list()
        word_freq_dict = dict()
        num_of_chuncks = num_of_lines // lines_per_chunk
        if num_of_lines % lines_per_chunk > 0:
            num_of_chuncks += 1
        print('Lines of file: ' + str(num_of_lines))
        print('Lines per chunk: ' + str(lines_per_chunk))
        print('Chunks: ' + str(num_of_chuncks))
        chunk_index = 0
        line_index = 1

        while chunk_index < num_of_chuncks:
            start_line = line_index
            end_line = start_line + lines_per_chunk - 1
            if end_line > num_of_lines:
                end_line = num_of_lines
            chunk_index += 1
            print(str(chunk_index) + '/' + str(num_of_chuncks) +
                  ' chunk: (start_line, end_line) -> (' + str(start_line) + ', ' + str(end_line) + ')')
            line_index += lines_per_chunk

            train_article_list = self.get_text_list_from_line_range(train_article_path, start_line, end_line)
            # print(train_article_list)
            print('  (1/8) The articles have been assigned to list.')
            train_summary_list = self.get_text_list_from_line_range(train_summary_path, start_line, end_line)
            print('  (2/8) The summaries have been assigned to list.')
            x = []
            y = []
            for sentence in train_article_list:
                # print(sentence)
                # tokenized_sentence = nltk.tokenize.word_tokenize(sentence)
                tokenized_sentence = sentence.split()
                x.append(tokenized_sentence)
                sent_len = len(tokenized_sentence)
                if sent_len > article_max_len:
                    article_max_len = sent_len
                for word in tokenized_sentence:
                    if word not in word_freq_dict:
                        word_freq_dict[word] = 1
                    else:
                        word_freq = word_freq_dict[word] + 1
                        word_freq_dict[word] = word_freq


            print('  (3/8) The words of articles have been added to the words dictionary.')
            print('  (4/8) X set has been created.')
            chunk_index_str = self.int_to_three_digits_str(chunk_index)
            with open(x_chunks_dir + 'x_' + chunk_index_str + '.pickle', 'wb') as f:
                pickle.dump(x, f)
            print('  (5/8) X_' + chunk_index_str + ' chunk has been written to file.')
            del x
            del train_article_list

            for sentence in train_summary_list:
                # print(sentence)
                # tokenized_sentence = nltk.tokenize.word_tokenize(sentence)
                tokenized_sentence = sentence.split()
                y.append(tokenized_sentence)
                sent_len = len(tokenized_sentence)
                if sent_len > summary_max_len:
                    summary_max_len = sent_len
                for word in tokenized_sentence:
                    if word not in word_freq_dict:
                        word_freq_dict[word] = 1
                    else:
                        word_freq = word_freq_dict[word] + 1
                        word_freq_dict[word] = word_freq
                    # words.append(word)

            print('  (6/8) The words of summaries have been appended to the list of words.')
            print('  (7/8) Y set has been created.')

            with open(y_chunks_dir + 'y_' + chunk_index_str + '.pickle', 'wb') as f:
                pickle.dump(y, f)
                print('  (8/8) Y_' + chunk_index_str + ' chunk has been written to file.')
            del y
            del train_summary_list


        print('(1/12) the whole dataset has been obtained')

        with open(word_freq_file_path, "wb") as f:
            pickle.dump(word_freq_dict, f)

        word2int_dict = dict()
        int2word_dict = dict()
        word2int_dict['<PAD>'] = 0
        word2int_dict['<UNK>'] = 1
        word2int_dict['<S>'] = 2
        word2int_dict['</S>'] = 3
        int2word_dict[0] = '<PAD>'
        int2word_dict[1] = '<UNK>'
        int2word_dict[2] = '<S>'
        int2word_dict[3] = '</S>'
        index_word_dict = 4
        for word, freq in word_freq_dict.items():
            word2int_dict[word] = index_word_dict
            int2word_dict[index_word_dict] = word
            index_word_dict += 1
            # word_dict[word] = len(word_dict)
        print('(2/12) word2int dictionary has been created.')
        # int2word_dict = dict(zip(word2int_dict.values(), word2int_dict.keys()))
        print('(3/12) int2word dictionary has been created.')

        with open(word2int_file_path, "wb") as f:
            pickle.dump(word2int_dict, f)
        print('(4/12) word2int has been written to binary file.')
        with open(int2word_file_path, "wb") as f:
            pickle.dump(int2word_dict, f)
        print('(5/12) Int2word has been written to binary file.')

        ######################
        with open(word2int_txt_file_path, "w+", encoding='utf8') as f:
            for k, v in word2int_dict.items():
                f.write('{} {}\n'.format(k, v))
        with open(int2word_txt_file_path, "w+", encoding='utf8') as f:
            for k, v in int2word_dict.items():
                f.write('{} {}\n'.format(k, v))
        print('(6/12) word2int and Int2word has been written to txt file.')
        #############################3

        for filename in os.listdir(x_chunks_dir):
            # print(filename)
            path_to_file = x_chunks_dir + filename
            x = self.read_pickle_file(path_to_file)
            # print(str(len(x[0])) + " " + str(x))
            x = list(map(lambda d: list(map(lambda w: word2int_dict.get(w, word2int_dict["<UNK>"]), d)), x))
            # print(str(len(x[0])) + " " + str(x))
            x = list(map(lambda d: d[:article_max_len], x))
            # print(str(len(x[0])) + " " + str(x))
            x = list(map(lambda d: d + (article_max_len - len(d)) * [word2int_dict["<PAD>"]], x))
            # print(str(len(x[0])) + " " + str(x))
            # ##############3
            # print(x[0])
            # print(x[1])
            with open(path_to_file, 'wb') as f:
                pickle.dump(x, f)
            del x

        print('(7/12) X set has been written to chunked files')

        for filename in os.listdir(y_chunks_dir):
            # print(filename)
            path_to_file = y_chunks_dir + filename
            y = self.read_pickle_file(y_chunks_dir + filename)
            # print(str(len(y[0])) + " " + str(y))
            y = list(map(lambda d: list(map(lambda w: word2int_dict.get(w, word2int_dict["<UNK>"]), d)), y))
            # print(str(len(y[0])) + " " + str(y))
            # y = list(map(lambda d: d[:summary_max_len], y))
            # print(str(len(y[0])) + " " + str(y))
            # ############33
            # print(y[0])
            # print(y[1])
            with open(path_to_file, 'wb') as f:
                pickle.dump(y, f)
            del y

        print('(8/12) Y set has been written to chunked files')

        maxlen_dict = dict()
        maxlen_dict['article_max_len'] = article_max_len
        maxlen_dict['summary_max_len'] = summary_max_len
        maxlen_dict['vocabulary_len'] = index_word_dict
        maxlen_dict['number_of_articles'] = num_of_lines
        maxlen_dict['number_of_chunks'] = num_of_chuncks
        maxlen_dict['articles_per_chunk'] = lines_per_chunk
        with open(maxlen_file_path, 'wb') as f:
            pickle.dump(maxlen_dict, f)

        with open(maxlen_txt_file_path, 'w') as f:
            for k, v in maxlen_dict.items():
                f.write(str(k) + ' ' + str(v) + '\n')
        print('(9/12) The dataset parameters have been written to file.')

        # Vocabulary
        vocab_list = []
        for key, value in word_freq_dict.items():
            temp = (key, value)
            vocab_list.append(temp)
        vocab_list = sorted(vocab_list, key=lambda tup: -tup[1])
        with open(vocab_txt_file_path, 'w+', encoding='utf8') as f:
            for w in vocab_list:
                f.write(str(w[0]) + ' ' + str(w[1]) + '\n')
        print('(10/12) The vocabulary has been written to file.')
        del vocab_list
        # return x, y, word_dict, reversed_dict, article_max_len, summary_max_len, maxlen_dict
        self.write_txt_files_for_binary_files(x_chunks_dir, x_txt_dir,
                                              int2word_file_path)
        # print('(11/9) The txt x chunk files have been written.')
        self.write_txt_files_for_binary_files(y_chunks_dir, y_txt_dir,
                                              int2word_file_path)
        print('(11/12) The txt chunk files have been written.')

        self.create_sample_training_set()
        print('(12/12) The sample dataset has been created')

        print('Process finished.')

    @staticmethod
    def int_to_three_digits_str(number):
        if number < 10:
            return '00' + str(number)
        elif number < 100:
            return '0' + str(number)
        elif number < 1000:
            return str(number)
        else:
            return "Error: the number has more than three digits"

    def get_text_list_from_line_range(self, data_path, start_line, end_line):
        with open(data_path, "r", encoding='utf-8') as f:
            lines_list = []
            start_greater_than = start_line - 1
            end_less_than = end_line + 1
            line_index = 0
            for line in f:
                line_index += 1
                if start_greater_than < line_index < end_less_than:
                    # line = self.clean_str(line.strip())
                    lines_list.append(line)
                    # print(str(line_index) + " " + line)
                if line_index == end_line:
                    break
        return lines_list

    def build_validation_or_testing_dataset(self, articles_data_file,
                                            data_dir, rouge_results_dir, system_summaries_dir,
                                            word2int_dict_pickle_file_path,
                                            int2word_dict_pickle_file_path,
                                            maxlen_file_path,
                                            purpose='validataion', simple_to_full=False):  # or purpose='test'
        # for filename in os.listdir(data_dir):
        #    os.remove(data_dir + filename)
        print('Old files from {}_data have been removed'.format(purpose))
        print('Article file: {}'.format(articles_data_file))
        print('data dir: {}'.format(data_dir))
        print('word2int_dict file: {}'.format(word2int_dict_pickle_file_path))
        print('int2word_dict file: {}'.format(int2word_dict_pickle_file_path))
        print('maxlen  file: {}'.format(maxlen_file_path))
        print('validataion: {}'.format(purpose))

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(rouge_results_dir):
            os.makedirs(rouge_results_dir)
        if not os.path.exists(system_summaries_dir):
            os.makedirs(system_summaries_dir)
        # if not os.path.exists(rouge_files_dir):
        #    os.makedirs(rouge_files_dir)

        word2int_dict = self.read_pickle_file(word2int_dict_pickle_file_path)
        print('(1/8) Word2int dictionary has been assigned to the variable')
        article_max_len = self.read_pickle_file(maxlen_file_path)['article_max_len']
        if simple_to_full:
            article_max_len = self.read_pickle_file(maxlen_file_path)['article_max_len']
        ################
        #print('article_max_len ', article_max_len)
        test_article_list = self.get_text_list(articles_data_file)
        print('(2/8) The {} articles has been assigned to list'.format(purpose))
        # x = list(map(lambda d: nltk.tokenize.word_tokenize(d), test_article_list))
        x = list(map(lambda d: d.split(), test_article_list))
        count_aritcles = 0
        for i in x:
            count_aritcles += 1
        #    if len(i) > article_max_len:
        #        # print(len(i))
        #        article_max_len = len(i)
        print('(3/8) The maximum article length has been obtained')
        # print(article_max_len)
        x = list(map(lambda d: list(map(lambda w: word2int_dict.get(w, word2int_dict["<UNK>"]), d)), x))
        x = list(map(lambda d: d[:article_max_len], x))
        x = list(map(lambda d: d + (article_max_len - len(d)) * [word2int_dict["<PAD>"]], x))
        print('(4/8) X has been created')
        with open(data_dir + "x_" + purpose + ".pickle", "wb") as f:
            pickle.dump(x, f)
        print('(5/8) X has been written to file: {}'.format(data_dir + "x_" + purpose + ".pickle"))
        maxlen_dict = dict()
        maxlen_dict[purpose + '_article_max_len'] = article_max_len
        maxlen_dict['number_of_articles'] = count_aritcles
        with open(data_dir + purpose + "_article_max_len.pickle", "wb") as f:
            pickle.dump(maxlen_dict, f)
        with open(data_dir + purpose + "_article_max_len.txt", 'w') as f:
            for k, v in maxlen_dict.items():
                f.write(str(k) + ' ' + str(v) + '\n')
        print('(6/8) The {} set has been created.'.format(purpose))
        self.write_txt_file_of_binary_file(data_dir + "x_" + purpose + ".pickle",
                                           data_dir + "x_" + purpose + ".txt",
                                           int2word_dict_pickle_file_path)
        print('(7/8) X has been written to txt file: {}.'.format(data_dir + "x_" + purpose + ".txt"))

        if simple_to_full:
            dir = paths.test_dir
            prefix = 'test_subset'
            if purpose == 'validation':
                dir = paths.validation_dir
                prefix = purpose

            read_file = dir + '{}_{}_titles.txt'.format(prefix, param.full_summary_model_id)
            write_file = dir + '{}_{}_titles.txt'.format(prefix, param.simple_to_full_model_id)
            r = open(read_file, 'r', encoding='utf8')
            w = open(write_file, 'w', encoding='utf8')
            for line in r:
                w.write(line)
            r.close()
            w.close()
            print('(8/8) File {} has been copied to {}.'.format(read_file, write_file))

        print('process finished.')

    @staticmethod
    def read_pickle_file(path_to_pickle_file):
        with open(path_to_pickle_file, "rb") as f:
            b = pickle.load(f)
        return b

    @staticmethod
    # for a text file, it returns a list of lines
    def get_text_list(data_path):
        with open(data_path, "r", encoding='utf-8') as f:
            lines_list = []
            for line in f:
                lines_list.append(line)
        return lines_list

    def create_sample_training_set(self, x_dir=paths.x_chunks_dir, y_dir=paths.y_chunks_dir,
                                   x_sample_dir=paths.x_chunks_sample_dir, y_sample_dir=paths.y_chunks_sample_dir,
                                   training_size=100, num_of_files=3):
        x_chunked_filenames_list = os.listdir(x_dir)
        y_chunked_filenames_list = os.listdir(y_dir)
        x_chunked_filenames_list.sort()
        y_chunked_filenames_list.sort()
        for xf, yf in zip(x_chunked_filenames_list[:num_of_files], y_chunked_filenames_list[:num_of_files]):
            print('{}  {}'.format(xf, yf))
            x = self.read_pickle_file(x_dir + xf)
            y = self.read_pickle_file(y_dir + yf)
            x_new = []
            y_new = []
            counter = 1
            for i, j in zip(x, y):
                x_new.append(i)
                y_new.append(j)
                counter += 1
                if counter > training_size:
                    break
            with open(x_sample_dir + xf, "wb") as f:
                pickle.dump(x_new, f)
            print('X set has been written to file: {}'.format(x_sample_dir + xf))

            with open(y_sample_dir + yf, "wb") as f:
                pickle.dump(y_new, f)
            print('Y set has been written to file:{}'.format(y_sample_dir + yf))

    # It write a txt file from a binary file
    def write_txt_file_of_binary_file(self, binary_file_path, txt_file_path, int2word_file_path):
        int2word = self.read_pickle_file(int2word_file_path)
        text_int_list = self.read_pickle_file(binary_file_path)
        with open(txt_file_path, 'w+', encoding='utf8') as f:
            for int_line_list in text_int_list:
                line_str = ''
                for int_word in int_line_list:
                    line_str += int2word[int_word] + " "
                f.write(line_str + '\n')

    # it writes txt files from a list of binary files
    def write_txt_files_for_binary_files(self, binary_file_dir, txt_file_dir, int2word_file_path):
        binary_files_list = os.listdir(binary_file_dir)
        int2word = self.read_pickle_file(int2word_file_path)
        for b in binary_files_list:
            text_int_list = self.read_pickle_file(binary_file_dir + b)
            with open(txt_file_dir + b + '.txt', 'w+', encoding='utf8') as f:
                for int_line_list in text_int_list:
                    line_str = ''
                    for int_word in int_line_list:
                        line_str += int2word[int_word] + " "
                    f.write(line_str + '\n')


if __name__ == "__main__":
    BuildDataset()
