import collections
import pickle
import nltk
import re
from pycontractions import Contractions
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import gensim
import utils
import os
from nltk.corpus import wordnet


class DataPreprocessing:

    def count_words_of_embedding_file(self, word2int_dict_pickle_file_path, word2vec_keyedvector_file_path):
        word2int_dict = self.read_pickle_file(word2int_dict_pickle_file_path)
        w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_keyedvector_file_path)
        count_existing_words = 0
        count_overall_words = 0
        count_absent_words = 0
        absent_words_dict = dict()
        for word, _ in word2int_dict.items():
            count_overall_words += 1
            if count_overall_words % 5000 == 0:
                print('count overall, existing & absent words {}, {} & {}'.format(count_overall_words,
                                                                                  count_existing_words,
                                                                                  count_absent_words))
            try:
                word = word.split(sep='_')[0]
                _ = w2v.word_vec(word)
                count_existing_words += 1
            except KeyError:
                count_absent_words += 1
                if absent_words_dict.get(word, None):
                    freq = absent_words_dict[word]
                    absent_words_dict[word] = freq + 1
                else:
                    absent_words_dict[word] = 1
        absent_words_list = []
        for k, v in absent_words_dict.items():
            absent_words_list.append((k, v))
        absent_words_list = sorted(absent_words_list, key=lambda tup: -tup[1])
        for (word, freq) in absent_words_list:
            print('{} {}'.format(word, freq))

        print('\ncount_existing_words {} {:.3f}%'.format(count_existing_words,
                                                         count_existing_words * 100.0 / count_overall_words))
        print(
            'count_absent_words {} {:.3f}'.format(count_absent_words, count_absent_words * 100.0 / count_overall_words))
        print('count_words {} {:.3f}%'.format(count_overall_words, count_overall_words * 100.0 / count_overall_words))
        return

    @staticmethod
    def read_pickle_file(path_to_pickle_file):
        with open(path_to_pickle_file, "rb") as f:
            b = pickle.load(f)
        return b

    def convert_binary_file_to_txt_file(self, binary_file_path, txt_file_path, read_per_megabytes=8):
        txt_file = open(txt_file_path, 'w', encoding='utf8')
        with open(binary_file_path, "rb") as f:
            read_bytes = read_per_megabytes * 1048576
            count_MB = read_per_megabytes
            bytes = f.read(read_bytes)
            while bytes:
                txt_file.write(" ".join(map(str, bytes)))
                bytes = f.read(read_bytes)
                print('MB: {}'.format(count_MB))
                count_MB += read_per_megabytes
            txt_file.write('\n')
        print('Binary file is converted.')
        return txt_file_path

    # it writes a binary and an txt file with random numbers of a range of given numbers and
    # return the list of random numbers
    @staticmethod
    def generate_a_range_of_random_numbers(test_path_to_pickle_file, test_path_to_txt_file,
                                           val_path_to_pickle_file, val_path_to_txt_file,
                                           test_num_of_samples, val_num_of_samples,
                                           start=1, end=144986):
        numbers = []
        for i in range(start, end + 1):
            numbers.append(i)
        selected_random_numbers = shuffle(numbers, random_state=4453664,
                                          n_samples=test_num_of_samples + val_num_of_samples)
        test_random_numbers = []
        val_random_numbers = []
        for i in range(0, test_num_of_samples):
            test_random_numbers.append(selected_random_numbers[i])
        for i in range(test_num_of_samples, test_num_of_samples + val_num_of_samples):
            val_random_numbers.append(selected_random_numbers[i])
        with open(test_path_to_pickle_file, 'wb') as f:
            pickle.dump(test_random_numbers, f)
        with open(val_path_to_pickle_file, 'wb') as f:
            pickle.dump(val_random_numbers, f)
        with open(test_path_to_txt_file, 'w', encoding='utf8') as f:
            for i in test_random_numbers:
                f.write('{}\n'.format(i))
        with open(val_path_to_txt_file, 'w', encoding='utf8') as f:
            for i in val_random_numbers:
                f.write('{}\n'.format(i))
        print('rows: {}, total samples: {}, test samples: {}, val samples: {}'.format(len(numbers),
                                                                                      len(selected_random_numbers),
                                                                                      len(test_random_numbers),
                                                                                      len(val_random_numbers)))

    def create_testing_and_validation_initial_subset(self, test_initial_article_path,
                                                     test_initial_summary_path,
                                                     test_subset_initial_article_file_path,
                                                     test_subset_initial_summary_file_path,
                                                     validation_initial_article_file_path,
                                                     validation_initial_summary_file_path,
                                                     test_random_lines_file_path,
                                                     validation_random_lines_file_path):
        print('creating_testing_and_validation_initial_subset...')
        article_read_f = open(test_initial_article_path, 'r', encoding='utf8')
        summary_read_f = open(test_initial_summary_path, 'r', encoding='utf8')
        pairs = []
        for article_line, summary_line in zip(article_read_f, summary_read_f):
            pairs.append((article_line, summary_line))
        test_lines_list = []
        val_lines_list = []
        with open(test_random_lines_file_path, "rb") as f:
            test_lines_list = pickle.load(f)
        with open(validation_random_lines_file_path, "rb") as f:
            val_lines_list = pickle.load(f)
        with open(test_subset_initial_article_file_path, 'w', encoding='utf8') as a:
            with open(test_subset_initial_summary_file_path, 'w', encoding='utf8') as s:
                for i in test_lines_list:
                    p = pairs[i]
                    a.write(p[0])
                    s.write(p[1])
        with open(validation_initial_article_file_path, 'w', encoding='utf8') as a:
            with open(validation_initial_summary_file_path, 'w', encoding='utf8') as s:
                for i in val_lines_list:
                    p = pairs[i]
                    a.write(p[0])
                    s.write(p[1])

    def view_line_with_phrase(self, file_path, phrase, without_list, num_of_lines=555444333):
        with open(file_path, 'r', encoding='utf8') as f:
            count_lines = 0
            count_lines_with_phrase = 0
            token_dict = dict()
            for line in f:
                count_lines += 1
                index = line.find(phrase)
                without_flag = True
                for s in without_list:
                    if line.find(s) > -1:
                        without_flag = False
                        break
                if index > -1 and without_flag:
                    count_lines_with_phrase += 1
                    print("{} {}".format(count_lines, line))
                if count_lines > num_of_lines:
                    print("Break at line: {}".format(count_lines))
                    break
        print('Cases: {}'.format(count_lines_with_phrase))
        print('Lines: {}'.format(count_lines))
        print('Lines with phrase ({}): {} ({}%)'.format(phrase,
                                                        count_lines_with_phrase,
                                                        round(count_lines_with_phrase * 100 / count_lines, 2)))

    def view_n_gram_with_symbol_and_phrase(self, file_path, n_gram, symbol, phrase, without_list, freq_greater_than):
        with open(file_path, 'r', encoding='utf8') as f:
            count_lines = 0
            count_lines_with_symbol = 0
            token_dict = dict()
            for line in f:
                count_lines += 1
                index = line.find(phrase)
                if index > -1:
                    without_flag = True
                    for s in without_list:
                        if line.find(s) > -1:
                            without_flag = False
                            break
                    if without_flag:

                        # print(line)
                        line_split_list = line.split()

                        line_length = len(line_split_list)
                        word_index = -1
                        for w in line_split_list:
                            word_index += 1
                            if w.find(symbol) > -1:
                                pre_w = ''
                                post_w = ''
                                post_post_w = ''
                                post_post_post_w = ''
                                if word_index > 0 and n_gram > 1:
                                    pre_w = line_split_list[word_index - 1]
                                if word_index + 1 < line_length and n_gram > 2:
                                    post_w = line_split_list[word_index + 1]
                                if word_index + 2 < line_length and n_gram > 3:
                                    post_post_w = line_split_list[word_index + 2]
                                if word_index + 3 < line_length and n_gram > 4:
                                    post_post_post_w = line_split_list[word_index + 3]
                                n_gram_phrase = pre_w + ' ' + w + ' ' + post_w + ' ' + post_post_w + post_post_post_w
                                if token_dict.get(n_gram_phrase, None):
                                    freq = token_dict[n_gram_phrase]
                                    token_dict[n_gram_phrase] = freq + 1
                                    count_lines_with_symbol += 1
                                else:
                                    token_dict[n_gram_phrase] = 1
                                    count_lines_with_symbol += 1
        token_list = []
        for key, value in token_dict.items():
            temp = (key, value)
            token_list.append(temp)

        symbol_list = sorted(token_list, key=lambda tup: -tup[1])
        print('\n\n')
        for i in symbol_list:
            if i[1] > freq_greater_than:
                print('{} {}'.format(i[0], i[1]))

        print('\nDistinct cases: {}'.format(len(symbol_list)))
        print('Lines: {}'.format(count_lines))
        print('Lines with {}-gram ({}): {} ({}%)'.format(n_gram, symbol, count_lines_with_symbol,
                                                         round(count_lines_with_symbol * 100 / count_lines, 2)))

    def vocab_count_words_and_statistics(self, article_path, model_summary_path, reports_dir,
                                         report_id='report', debug=False):
        start_time = time.time()
        article_tokens_dict = {}
        summary_tokens_dict = {}
        count_article_tokens = 0
        count_summary_tokens = 0
        count_article_distinct_tokens = 0
        count_summary_distinct_tokens = 0
        distinct_tokens_set = set()
        article_length_list = []
        summary_length_list = []
        line_counter = 0
        print("Reading files:\n   {}\n   {}".format(article_path, model_summary_path))
        article_read_f = open(article_path, 'r', encoding='utf8')
        summary_read_f = open(model_summary_path, 'r', encoding='utf8')
        t1 = t2 = t3 = t4 = t5 = t6 = 0
        for article_line, summary_line in zip(article_read_f, summary_read_f):
            line_counter += 1
            # t = time.time()
            article_line_tokens = article_line.split()  # nltk.tokenize.word_tokenize(article_line)  #
            # t1 += (time.time() - t)
            # t = time.time()
            summary_line_tokens = summary_line.split()  # nltk.tokenize.word_tokenize(summary_line)  #
            # t2 += (time.time() - t)
            # t = time.time()
            article_length_list.append(len(article_line_tokens))
            # t3 += (time.time() - t)
            # t = time.time()
            summary_length_list.append(len(summary_line_tokens))
            # t4 += (time.time() - t)
            # t = time.time()
            for toc in article_line_tokens:
                if toc not in article_tokens_dict.keys():
                    article_tokens_dict[toc] = 1
                    count_article_tokens += 1
                    count_article_distinct_tokens += 1
                    distinct_tokens_set.add(toc)
                else:
                    freq = article_tokens_dict[toc]
                    article_tokens_dict[toc] = freq + 1
                    count_article_tokens += 1
                # if article_tokens_dict.get(toc, None):
                #    freq = article_tokens_dict[toc]
                #    article_tokens_dict[toc] = freq + 1
                #    count_article_tokens += 1
                # else:
                #    article_tokens_dict[toc] = 1
                #    count_article_tokens += 1
                #    count_article_distinct_tokens += 1
            for toc in summary_line_tokens:
                if toc not in summary_tokens_dict.keys():
                    summary_tokens_dict[toc] = 1
                    count_summary_tokens += 1
                    count_summary_distinct_tokens += 1
                    distinct_tokens_set.add(toc)
                else:
                    freq = summary_tokens_dict[toc]
                    summary_tokens_dict[toc] = freq + 1
                    count_summary_tokens += 1
                # if summary_tokens_dict.get(toc, None):
                #    freq = summary_tokens_dict[toc]
                #    summary_tokens_dict[toc] = freq + 1
                #    count_summary_tokens += 1
                # else:
                #    summary_tokens_dict[toc] = 1
                #    count_summary_tokens += 1
                #    count_summary_distinct_tokens += 1
            if line_counter % 100000 == 0:
                print('{} lines, Time: {}'.format(line_counter,
                                                  datetime.timedelta(seconds=round(time.time() - start_time, 1))))
            if debug:
                debug_lines = 100000
                if line_counter > debug_lines:
                    print('t1 article_line_tokens {}\n'
                          't2 summary_line_tokens {}\n'
                          't3 article_length_list {}\n'
                          't4 summary_length_list {}\n'
                          't5 article_tokens_dict {}\n'
                          't6 summary_tokens_dict {}\n'.format(datetime.timedelta(seconds=t1),
                                                               datetime.timedelta(seconds=t2),
                                                               datetime.timedelta(seconds=t3),
                                                               datetime.timedelta(seconds=t4),
                                                               datetime.timedelta(seconds=t5),
                                                               datetime.timedelta(seconds=t6)))
                    break

        print("tokens for articles and summaries file have been added to dictionaries")
        article_read_f.close()
        summary_read_f.close()
        article_tokens_list = []
        summary_tokens_list = []
        for k, v in article_tokens_dict.items():
            article_tokens_list.append((k, v))
        for k, v in summary_tokens_dict.items():
            summary_tokens_list.append((k, v))
        article_tokens_list = sorted(article_tokens_list, key=lambda tup: -tup[1])
        summary_tokens_list = sorted(summary_tokens_list, key=lambda tup: -tup[1])
        print('Tokens have been added to sorted lists')
        article_vocabulary_path = '{}{}_vocab_article.txt'.format(reports_dir, report_id)
        summary_vocabulary_path = '{}{}_vocab_summary.txt'.format(reports_dir, report_id)
        report_file_path = '{}{}_report.txt'.format(reports_dir, report_id)
        chart_file_path = '{}{}_chart.pdf'.format(reports_dir, report_id)
        print('Writing files:\n   {}\n   {}\n   {}\n   {}'.format(article_vocabulary_path,
                                                                  summary_vocabulary_path,
                                                                  report_file_path,
                                                                  chart_file_path))
        article_max_len = np.max(np.array(article_length_list))
        article_min_len = np.min(np.array(article_length_list))
        article_avg_len = np.mean(np.array(article_length_list))
        article_var_len = np.var(np.array(article_length_list))
        article_std_len = np.std(np.array(article_length_list))
        summary_max_len = np.max(np.array(summary_length_list))
        summary_min_len = np.min(np.array(summary_length_list))
        summary_avg_len = np.mean(np.array(summary_length_list))
        summary_var_len = np.var(np.array(summary_length_list))
        summary_std_len = np.std(np.array(summary_length_list))
        count_distinct_tokens = len(distinct_tokens_set)
        del distinct_tokens_set
        with open(article_vocabulary_path, 'w+', encoding='utf8') as f:
            for a in article_tokens_list:
                f.write('{} {}\n'.format(a[0], a[1]))
        with open(summary_vocabulary_path, 'w+', encoding='utf8') as f:
            for a in summary_tokens_list:
                f.write('{} {}\n'.format(a[0], a[1]))
        print('Vocabulary files for both articles and summaries have been written')
        with open(report_file_path, 'w+', encoding='utf8') as f:
            comment = 'reports for {} data files'.format(report_id)
            f.write("Reports of dataset files.\nComment: {}\n\n".format(comment))
            f.write("Reading files:\n   {}\n   {}\n\n".format(article_path, model_summary_path))
            f.write('Writing files:\n   {}\n   {}\n   {}\n   {}\n\n'.format(article_vocabulary_path,
                                                                            summary_vocabulary_path,
                                                                            report_file_path,
                                                                            chart_file_path))
            f.write('Vocabulary size: {}\n\n'.format(count_distinct_tokens))
            f.write('Articles:'
                    '\n   number of instances {}'
                    '\n   tokens: {}'
                    '\n   distinct tokens: {}  {:.4f}%'
                    '\n   min, max & avg length: {}, {}, {:.3f}'
                    '\n   var & std of length:'
                    ' {:.3f} & {:.3f}\n\n'.format(line_counter,
                                                  count_article_tokens,
                                                  count_article_distinct_tokens,
                                                  100 * count_article_distinct_tokens / count_article_tokens,
                                                  article_min_len, article_max_len, article_avg_len,
                                                  article_var_len, article_std_len
                                                  ))

            f.write('Summaries:'
                    '\n   number of instances {}'
                    '\n   tokens: {}'
                    '\n   distinct tokens: {}  {:.4f}%'
                    '\n   min, max & avg length: {}, {} & {:.3f}'
                    '\n   var & std of length:'
                    ' {:.3f} & {:.3f}\n\n'.format(line_counter,
                                                  count_summary_tokens,
                                                  count_summary_distinct_tokens,
                                                  100 * count_summary_distinct_tokens / count_summary_tokens,
                                                  summary_min_len, summary_max_len, summary_avg_len,
                                                  summary_var_len, summary_std_len
                                                  ))
            # f.write('article_length_list = {}\n\n'.format(article_length_list))
            # f.write('summary_length_list = {}\n\n'.format(summary_length_list))
        print('Report file has been written.')
        # x = np.random.normal(size=1000)
        plt.figure(1)
        plt.subplot(211)
        plt.hist(article_length_list, histtype='barstacked', density=True, bins=article_max_len)  # normed=True,
        plt.ylabel('Probability')
        plt.ylabel('Article length')
        # plt.savefig(reports_dir + reports_filename + '_chart_artcicle_len_distr.pdf')
        plt.subplot(212)
        plt.hist(summary_length_list, histtype='stepfilled', density=True, bins=article_max_len)
        plt.ylabel('Probability')
        plt.ylabel('Summary length')
        plt.tick_params()
        plt.tight_layout()
        plt.savefig(chart_file_path)
        plt.close()
        print('charts have been plotted.')
        print('Process finished.')

        with open(report_file_path, 'r', encoding='utf8') as f:
            content = f.read()
            print(content)

        # remove duplicates, instances with longer summaries than articles and instances with <unk>
        # from the original data to the initial data

    def clean_duc_dataset_from_original_to_cleaned(self, article_path,
                                                   model_summary1_path, model_summary2_path,
                                                   model_summary3_path, model_summary4_path,
                                                   new_article_path,
                                                   new_model_summary1_path, new_model_summary2_path,
                                                   new_model_summary3_path, new_model_summary4_path,
                                                   word2vec_path,
                                                   print_per_lines=50):
        print('Cleaning... from orgiginal to cleaned DUC data...')
        start_time = time.time()
        cont = Contractions(word2vec_path)
        article_read_f = open(article_path, 'r', encoding='utf8')
        summary1_read_f = open(model_summary1_path, 'r', encoding='utf8')
        summary2_read_f = open(model_summary2_path, 'r', encoding='utf8')
        summary3_read_f = open(model_summary3_path, 'r', encoding='utf8')
        summary4_read_f = open(model_summary4_path, 'r', encoding='utf8')
        article_write_f = open(new_article_path, 'w+', encoding='utf8')
        summary1_write_f = open(new_model_summary1_path, 'w+', encoding='utf8')
        summary2_write_f = open(new_model_summary2_path, 'w+', encoding='utf8')
        summary3_write_f = open(new_model_summary3_path, 'w+', encoding='utf8')
        summary4_write_f = open(new_model_summary4_path, 'w+', encoding='utf8')
        count_input_lines = 0
        count_output_lines = 0
        for article_line, summary1_line, summary2_line, summary3_line, summary4_line in zip(article_read_f,
                                                                                            summary1_read_f,
                                                                                            summary2_read_f,
                                                                                            summary3_read_f,
                                                                                            summary4_read_f):
            count_input_lines += 1
            article_line_new = self.clean_text(article_line.lower(), cont)
            summary1_line_new = self.clean_text(summary1_line.lower(), cont)
            summary2_line_new = self.clean_text(summary2_line.lower(), cont)
            summary3_line_new = self.clean_text(summary3_line.lower(), cont)
            summary4_line_new = self.clean_text(summary4_line.lower(), cont)
            article_write_f.write(article_line_new + '\n')
            summary1_write_f.write(summary1_line_new + '\n')
            summary2_write_f.write(summary2_line_new + '\n')
            summary3_write_f.write(summary3_line_new + '\n')
            summary4_write_f.write(summary4_line_new + '\n')
            if count_input_lines % print_per_lines == 0:
                print('{}, {}, {} (input line, output line and time for cleaning)'.format(
                    count_input_lines,
                    count_output_lines,
                    datetime.timedelta(
                        seconds=round(
                            time.time() - start_time,
                            0))))
                print('    art__in: {}'.format(article_line), end='')
                print('    art_out: {}'.format(article_line_new), end='\n')
                print('    sum1__in: {}'.format(summary1_line), end='')
                print('    sum1_out: {}'.format(summary1_line_new), end='\n')
                print('    sum2__in: {}'.format(summary2_line), end='')
                print('    sum2_out: {}'.format(summary2_line_new), end='\n')
                print('    sum3__in: {}'.format(summary3_line), end='')
                print('    sum3_out: {}'.format(summary3_line_new), end='\n')
                print('    sum4__in: {}'.format(summary4_line), end='')
                print('    sum4_out: {}'.format(summary4_line_new), end='\n')

        article_read_f.close()
        summary1_read_f.close()
        summary2_read_f.close()
        summary3_read_f.close()
        summary4_read_f.close()
        # print('\ncount_shorter_articles_than_summary: {}'.format(count_shorter_articles_than_summary))
        article_write_f.close()
        summary1_write_f.close()
        summary2_write_f.close()
        summary3_write_f.close()
        summary4_write_f.close()

    # remove duplicates and clean dataset
    def clean_dataset(self, article_path, model_summary_path, new_article_path, new_model_summary_path,
                      word2vec_path, print_per_lines=10000, debug=False):
        print('Cleaning dataset...')
        start_time = time.time()
        article_read_f = open(article_path, 'r', encoding='utf8')
        summary_read_f = open(model_summary_path, 'r', encoding='utf8')
        article_write_f = open(new_article_path, 'w+', encoding='utf8')
        summary_write_f = open(new_model_summary_path, 'w+', encoding='utf8')
        cont = Contractions(word2vec_path)
        print('word2vec vectors have been loaded')
        count_input_lines = 0
        article_summary_set = set()
        for article_line, summary_line in zip(article_read_f, summary_read_f):
            article_summary_set.add((article_line, summary_line))
            count_input_lines += 1
            if count_input_lines % print_per_lines == 0:
                print('{}, {} (input line and time for removing duplicates)'.format(count_input_lines,
                                                                                    datetime.timedelta(
                                                                                        seconds=round(
                                                                                            time.time() - start_time,
                                                                                            0))))
            if debug:
                if count_input_lines == 1000:
                    break
        article_read_f.close()
        summary_read_f.close()
        count_input_lines = 0
        count_output_lines = 0
        count_shorter_articles_than_summary = 0
        for el in article_summary_set:
            article_line = el[0]
            summary_line = el[1]
            count_input_lines += 1
            length_article = len(article_line)
            length_summary = len(summary_line)
            if length_article > length_summary:
                count_output_lines += 1
                article_line_new = self.clean_text(article_line, cont)
                summary_line_new = self.clean_text(summary_line, cont)
                article_write_f.write(article_line_new + '\n')
                summary_write_f.write(summary_line_new + '\n')
                if count_input_lines % print_per_lines == 0:
                    print('{}, {}, {} (input line, output line and time for cleaning)'.format(count_input_lines,
                                                                                              count_output_lines,
                                                                                              datetime.timedelta(
                                                                                                  seconds=round(
                                                                                                      time.time() - start_time,
                                                                                                      0))))
                    print('    art__in: {}'.format(article_line), end='')
                    print('    art_out: {}'.format(article_line_new), end='\n')
                    print('    sum__in: {}'.format(summary_line), end='')
                    print('    sum_out: {}'.format(summary_line_new), end='\n')
            else:
                count_shorter_articles_than_summary += 1
            if debug:
                if count_input_lines == 1000:
                    break
        print('\ncount_shorter_articles_than_summary: {}'.format(count_shorter_articles_than_summary))
        article_write_f.close()
        summary_write_f.close()

        # remove duplicates and clean dataset

    @staticmethod
    def clean_text(text, contractions):
        text = re.sub("-lrb-(.*?)-rrb-", "", text)
        text = text.replace('-lrb-', '').replace('-rrb-', '').replace('.', '')
        table = str.maketrans({key: ' ' for key in '!"$%&()*+,-./:;<=>?@[\]^_`{|}~'})
        text = text.translate(table)
        # text = text.replace('-', '')
        text = re.sub("[0-9]", "#", text)
        text = re.sub(' +', ' ', text)
        # text = "ca n't and wo n't and i 'm and you 'd have and '' it is n't"
        text = text.replace("ca n't", "can not"). \
            replace("wo n't", "will not"). \
            replace("n't", "not"). \
            replace("''", " "). \
            replace(" 'm ", " am "). \
            replace(" 've ", " have "). \
            replace(" 're ", " are ")
        text = re.sub(' +', ' ', text)
        text = text.replace(" '", "'"). \
            replace('were', 'WWEERREE')
        text = list(contractions.expand_texts([text], precise=True, scores=False))[0]
        text = text.replace('WWEERREE', 'were'). \
            replace("'s ", " 's "). \
            replace("' s ", " 's "). \
            replace(" and'#", " and '#"). \
            replace(" the'#", " the '#"). \
            replace("'#s ", " '#s "). \
            replace("' ", " ' "). \
            replace("\r", "").replace("\n", ""). \
            replace("'#", " '#")
        text = text.replace("'s", " 's").replace("'#", " '#")
        text = text.replace("'n", " 'n"). \
            replace("'em", " 'em").replace("'d", " 'd").replace("'m", " 'm").replace("\n", "")
        text = re.sub(' +', ' ', text)
        text = text.strip()
        return text

    def remove_duplicates(self, article_path, model_summary_path, new_article_path, new_model_summary_path,
                          print_per_lines=10000, debug=False):
        print('remove duplicates...')
        start_time = time.time()
        article_read_f = open(article_path, 'r', encoding='utf8')
        summary_read_f = open(model_summary_path, 'r', encoding='utf8')
        article_write_f = open(new_article_path, 'w+', encoding='utf8')
        summary_write_f = open(new_model_summary_path, 'w+', encoding='utf8')
        count_input_lines = 0
        article_summary_set = set()
        for article_line, summary_line in zip(article_read_f, summary_read_f):
            article_summary_set.add((article_line, summary_line))
            count_input_lines += 1
            if count_input_lines % print_per_lines == 0:
                print('{}, {} (input line and time)'.format(count_input_lines,
                                                            datetime.timedelta(
                                                                seconds=round(
                                                                    time.time() - start_time, 0))))
            # Early stopping for testing
            if debug:
                if count_input_lines == 1000:
                    break
        for el in article_summary_set:
            article_write_f.write(el[0])
            summary_write_f.write(el[1])

        # print('count_shorter_articles_than_summary: {}'.format(count_shorter_articles_than_summary))
        article_read_f.close()
        summary_read_f.close()
        article_write_f.close()
        summary_write_f.close()

    @staticmethod
    def clean_es_text(text):
        text = text.replace("'s", " 's").replace("'#", " '#")
        text = text.replace("'n", " 'n"). \
            replace("'em", " 'em").replace("'d", " 'd").replace("'m", " 'm").replace("\n", "")
        text = re.sub(' +', ' ', text)
        text = text.strip()
        return text

    # clean words such as mary's which will be done mery 's
    def clean_es_dataset(self, article_path, model_summary_path, new_article_path, new_model_summary_path,
                         print_per_lines=10000, debug=False):
        print('Cleaning dataset...')
        start_time = time.time()
        article_read_f = open(article_path, 'r', encoding='utf8')
        summary_read_f = open(model_summary_path, 'r', encoding='utf8')
        article_write_f = open(new_article_path, 'w+', encoding='utf8')
        summary_write_f = open(new_model_summary_path, 'w+', encoding='utf8')
        count_input_lines = 0
        for article_line, summary_line in zip(article_read_f, summary_read_f):
            count_input_lines += 1
            article_line_new = self.clean_es_text(article_line)
            summary_line_new = self.clean_es_text(summary_line)
            article_write_f.write(article_line_new + '\n')
            summary_write_f.write(summary_line_new + '\n')
            if count_input_lines % print_per_lines == 0:
                print('{}, {} (line and time for cleaning)'.format(count_input_lines, datetime.timedelta(
                    seconds=round(
                        time.time() - start_time,
                        0))))
                print('    art__in: {}'.format(article_line), end='')
                print('    art_out: {}'.format(article_line_new), end='\n')
                print('    sum__in: {}'.format(summary_line), end='')
                print('    sum_out: {}'.format(summary_line_new), end='\n')
            if debug:
                if count_input_lines == 1000:
                    break
        article_write_f.close()
        summary_write_f.close()

    a = ""

    def noun_word_freq_hypernympaths(self, input_word_pos_freq_pickle_file_path,
                                     input_word_freq_hypernyms_pickle_file_path,
                                     output_noun_freq_pickle_file_path,
                                     output_noun_freq_txt_file_path):
        input_word_pos_freq_dict = utils.read_pickle_file(input_word_pos_freq_pickle_file_path)
        input_word_freq_hypernyms_dict = utils.read_pickle_file(input_word_freq_hypernyms_pickle_file_path)
        output_noun_freq_hypernyms_dict = dict()
        max_freq = 0
        for (word, pos), freq in input_word_pos_freq_dict.items():
            if pos == 'n':
                if freq > max_freq:
                    max_freq = freq
        log_max_freq = np.log10(max_freq + 1)
        for (word, pos), freq in input_word_pos_freq_dict.items():
            if pos == 'n':
                if input_word_freq_hypernyms_dict.get(word, None):
                    _, _, hypernym_detph_list = input_word_freq_hypernyms_dict[word]
                    output_noun_freq_hypernyms_dict[word] = (
                        freq, np.log10(freq + 1) / log_max_freq, hypernym_detph_list)
                else:
                    output_noun_freq_hypernyms_dict[word] = (freq, np.log10(freq + 1) / log_max_freq, [])
        output_noun_freq_hypernyms_list = []
        for word, (freq, norm_freq, hypernyms) in output_noun_freq_hypernyms_dict.items():
            output_noun_freq_hypernyms_list.append((word, freq, norm_freq, hypernyms))
        output_noun_freq_hypernyms_list = utils.sort_by_second(output_noun_freq_hypernyms_list)
        with open(output_noun_freq_txt_file_path, 'w', encoding='utf8') as f:
            for (word, freq, norm_freq, hypernyms) in output_noun_freq_hypernyms_list:
                f.write('{} {} {} {}'.format(word, freq, norm_freq, hypernyms) + '\n')
        with open(output_noun_freq_pickle_file_path, 'wb') as f:
            pickle.dump(output_noun_freq_hypernyms_dict, f)

    def conver_dataset_with_ner_from_wordnet(self, input_article_pos_pickle_file_path,
                                             input_summary_pos_pickle_file_path,
                                             input_word_freq_hypernyms_pickle_file_path,
                                             output_article_file_path, output_summary_file_path,
                                             norm_freq_thresold=1.1,
                                             print_per_line=100000):
        article_word_pos_per_line_list = utils.read_pickle_file(input_article_pos_pickle_file_path)
        summary_word_pos_per_line_list = utils.read_pickle_file(input_summary_pos_pickle_file_path)
        word_freq_hypernyms_dict = utils.read_pickle_file(input_word_freq_hypernyms_pickle_file_path)
        # stanford_ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
        wordnet_ner_tags = ['person', 'location', 'organization']
        output_article_file = open(output_article_file_path, 'w', encoding='utf8')
        output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
        count_article_wordnet_replacements = 0
        count_summary_wordnet_replacements = 0
        line_count = 0
        for article_pos_line_list, summary_pos_line_list in \
                zip(article_word_pos_per_line_list, summary_word_pos_per_line_list):
            line_count += 1
            article_line = ''
            summary_line = ''
            new_article_line = ''
            new_summary_line = ''
            for (token, pos) in article_pos_line_list:
                article_line += token + ' '
                if pos == 'n' and word_freq_hypernyms_dict.get(token, None):
                    (freq, norm_freq, hypernyms_path_list) = word_freq_hypernyms_dict[token]
                    # hypernyms_path_list = word_freq_hypernyms_dict[token][2]
                    flag = True
                    if norm_freq < norm_freq_thresold:
                        for (hyp, depth) in hypernyms_path_list:
                            if hyp in wordnet_ner_tags:
                                new_article_line += hyp + '_ '
                                count_article_wordnet_replacements += 1
                                flag = False
                                break
                    if flag:
                        new_article_line += token + ' '
                else:
                    new_article_line += token + ' '
            for (token, pos) in summary_pos_line_list:
                summary_line += token + ' '
                if pos == 'n' and word_freq_hypernyms_dict.get(token, None):
                    (freq, norm_freq, hypernyms_path_list) = word_freq_hypernyms_dict[token]
                    # hypernyms_path_list = word_freq_hypernyms_dict[token][2]
                    flag = True
                    if norm_freq < norm_freq_thresold:
                        for (hyp, depth) in hypernyms_path_list:
                            if hyp in wordnet_ner_tags:
                                new_summary_line += hyp + '_ '
                                count_summary_wordnet_replacements += 1
                                flag = False
                                break
                    if flag:
                        new_summary_line += token + ' '
                else:
                    new_summary_line += token + ' '

            output_article_file.write(new_article_line.strip() + '\n')
            output_summary_file.write(new_summary_line.strip() + '\n')
            if line_count % print_per_line == 0:
                print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                    line_count, article_line, new_article_line, summary_line, new_summary_line))
                print('\nArticle wordnet replacements: {}\nSummary wordnet replacements: {}\n'
                      'Overall wordnet replacements: {}\n'.format(count_article_wordnet_replacements,
                                                                  count_summary_wordnet_replacements,
                                                                  count_article_wordnet_replacements +
                                                                  count_summary_wordnet_replacements))
        output_article_file.close()
        output_summary_file.close()

        print('\nArticle wordnet replacements: {}\nSummary wordnet replacements: {}\n'
              'Overall wordnet replacements: {}\n'.format(count_article_wordnet_replacements,
                                                          count_summary_wordnet_replacements,
                                                          count_article_wordnet_replacements +
                                                          count_summary_wordnet_replacements))

    def conver_duc_dataset_with_ner_from_stanford_and_wordnet(self, input_article_ner_pickle_file_path,
                                                              input_summary1_ner_pickle_file_path,
                                                              input_summary2_ner_pickle_file_path,
                                                              input_summary3_ner_pickle_file_path,
                                                              input_summary4_ner_pickle_file_path,
                                                              input_article_pos_pickle_file_path,
                                                              input_summary1_pos_pickle_file_path,
                                                              input_summary2_pos_pickle_file_path,
                                                              input_summary3_pos_pickle_file_path,
                                                              input_summary4_pos_pickle_file_path,
                                                              input_word_freq_hypernyms_pickle_file_path,
                                                              output_article_file_path,
                                                              output_summary1_file_path,
                                                              output_summary2_file_path,
                                                              output_summary3_file_path,
                                                              output_summary4_file_path,
                                                              word_freq_thresold=9999990,
                                                              print_per_line=100000):
        word_freq_hypernyms_dict = utils.read_pickle_file(input_word_freq_hypernyms_pickle_file_path)
        article_word_ner_per_line_list = utils.read_pickle_file(input_article_ner_pickle_file_path)
        print('1/9')
        summary1_word_ner_per_line_list = utils.read_pickle_file(input_summary1_ner_pickle_file_path)
        summary2_word_ner_per_line_list = utils.read_pickle_file(input_summary2_ner_pickle_file_path)
        summary3_word_ner_per_line_list = utils.read_pickle_file(input_summary3_ner_pickle_file_path)
        summary4_word_ner_per_line_list = utils.read_pickle_file(input_summary4_ner_pickle_file_path)
        print('2/9')
        stanford_ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
        output_article_file = open(output_article_file_path, 'w', encoding='utf8')
        output_summary1_file = open(output_summary1_file_path, 'w', encoding='utf8')
        output_summary2_file = open(output_summary2_file_path, 'w', encoding='utf8')
        output_summary3_file = open(output_summary3_file_path, 'w', encoding='utf8')
        output_summary4_file = open(output_summary4_file_path, 'w', encoding='utf8')
        count_article_driven_replacements = 0
        count_summary_driven_replacements = 0
        line_count = 0
        for article_line_list, summary1_line_list, summary2_line_list, summary3_line_list, summary4_line_list \
                in zip(article_word_ner_per_line_list, summary1_word_ner_per_line_list, summary2_word_ner_per_line_list,
                       summary3_word_ner_per_line_list, summary4_word_ner_per_line_list):
            line_count += 1
            article_line = ' '
            summary1_line = ' '
            summary2_line = ' '
            summary3_line = ' '
            summary4_line = ' '
            for (article_token, article_ner) in article_line_list:
                article_line += article_token + ' '
            for (summary_token, summary_ner) in summary1_line_list:
                summary1_line += summary_token + ' '
            for (summary_token, summary_ner) in summary2_line_list:
                summary2_line += summary_token + ' '
            for (summary_token, summary_ner) in summary3_line_list:
                summary3_line += summary_token + ' '
            for (summary_token, summary_ner) in summary4_line_list:
                summary4_line += summary_token + ' '
            new_article_line = article_line
            new_summary1_line = summary1_line
            new_summary2_line = summary2_line
            new_summary3_line = summary3_line
            new_summary4_line = summary4_line
            for (article_token, article_ner) in article_line_list:
                if article_ner in stanford_ner_tags:
                    change_flag = True
                    if word_freq_hypernyms_dict.get(article_token, None):
                        (freq, norm_freq, hypernyms_path_list) = word_freq_hypernyms_dict[article_token]
                        if freq > word_freq_thresold:
                            change_flag = False
                    if change_flag:
                        token_find = ' {} '.format(article_token)
                        find_index = max(new_summary1_line.find(token_find), new_summary2_line.find(token_find),
                                         new_summary3_line.find(token_find), new_summary4_line.find(token_find))
                        if find_index > -1:
                            token_replace = ' {} '.format(article_ner)
                            new_summary1_line = new_summary1_line.replace(token_find, token_replace)
                            new_summary2_line = new_summary2_line.replace(token_find, token_replace)
                            new_summary3_line = new_summary3_line.replace(token_find, token_replace)
                            new_summary4_line = new_summary4_line.replace(token_find, token_replace)
                            new_article_line = new_article_line.replace(token_find, token_replace)
                            count_article_driven_replacements += 1

            for summary_line_list, output_summary_file, new_summary_line, summary_line in zip(
                    [summary1_line_list, summary2_line_list, summary3_line_list, summary4_line_list],
                    [output_summary1_file, output_summary2_file, output_summary3_file, output_summary4_file],
                    [new_summary1_line, new_summary2_line, new_summary3_line, new_summary4_line],
                    [summary1_line, summary2_line, summary3_line, summary4_line]):
                for (summary_token, summary_ner) in summary_line_list:
                    if summary_ner in stanford_ner_tags:
                        change_flag = True
                        if word_freq_hypernyms_dict.get(summary_token, None):
                            (freq, norm_freq, hypernyms_path_list) = word_freq_hypernyms_dict[summary_token]
                            if freq > word_freq_thresold:
                                change_flag = False
                        if change_flag:
                            token_find = ' {} '.format(summary_token)
                            find_index = new_article_line.find(token_find)
                            if find_index > -1:
                                token_replace = ' {} '.format(summary_ner)
                                new_article_line = new_article_line.replace(token_find, token_replace)
                                new_summary_line = new_summary_line.replace(token_find, token_replace)
                                count_summary_driven_replacements += 1
                output_summary_file.write(new_summary_line.strip() + '\n')

                if line_count % print_per_line == 0:
                    print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                        line_count, article_line, new_article_line, summary_line, new_summary_line))
                    print(
                        '\tarticle driven replacements: {}\n\tSummary driven replacements: {}\n'
                        '\tOverall replacements: {}'.format(
                            count_article_driven_replacements, count_summary_driven_replacements,
                            count_article_driven_replacements + count_summary_driven_replacements))
            output_article_file.write(new_article_line.strip() + '\n')
        del article_word_ner_per_line_list
        del summary1_word_ner_per_line_list
        del summary2_word_ner_per_line_list
        del summary3_word_ner_per_line_list
        del summary4_word_ner_per_line_list
        output_article_file.close()
        output_summary1_file.close()
        output_summary2_file.close()
        output_summary3_file.close()
        output_summary4_file.close()
        print('3/9')
        print('\narticle driven replacements: {}\nSummary driven replacements: {}\nOverall replacements: {}'.format(
            count_article_driven_replacements, count_summary_driven_replacements,
            count_article_driven_replacements + count_summary_driven_replacements))

        wordnet_ner_tags = ['person', 'location', 'organization']

        article_per_line_list = []
        output_article_file = open(output_article_file_path, 'r', encoding='utf8')
        for line in output_article_file:
            article_per_line_list.append(line.split())
        output_article_file.close()
        summary1_per_line_list = []
        summary2_per_line_list = []
        summary3_per_line_list = []
        summary4_per_line_list = []
        output_summary1_file = open(output_summary1_file_path, 'r', encoding='utf8')
        output_summary2_file = open(output_summary2_file_path, 'r', encoding='utf8')
        output_summary3_file = open(output_summary3_file_path, 'r', encoding='utf8')
        output_summary4_file = open(output_summary4_file_path, 'r', encoding='utf8')
        for line in output_summary1_file:
            summary1_per_line_list.append(line.split())
        output_summary1_file.close()
        for line in output_summary2_file:
            summary2_per_line_list.append(line.split())
        output_summary2_file.close()
        for line in output_summary3_file:
            summary3_per_line_list.append(line.split())
        output_summary3_file.close()
        for line in output_summary4_file:
            summary4_per_line_list.append(line.split())
        output_summary4_file.close()

        print('4/9')
        article_word_pos_per_line_list = utils.read_pickle_file(input_article_pos_pickle_file_path)
        print('5/9')
        output_article_file = open(output_article_file_path, 'w', encoding='utf8')
        output_summary1_file = open(output_summary1_file_path, 'w', encoding='utf8')
        output_summary2_file = open(output_summary2_file_path, 'w', encoding='utf8')
        output_summary3_file = open(output_summary3_file_path, 'w', encoding='utf8')
        output_summary4_file = open(output_summary4_file_path, 'w', encoding='utf8')
        stan_repl_art = 0
        wordnet_repl_art = 0
        line_count = 0

        for article_line_list, summary1_line_list, summary2_line_list, summary3_line_list, summary4_line_list, \
            word_pos_line_list in zip(article_per_line_list, summary1_per_line_list, summary2_per_line_list,
                                      summary3_per_line_list, summary4_per_line_list, article_word_pos_per_line_list):
            # for line_list in article_per_line_list:
            line_count += 1
            new_line_str = ''
            old_line_str = ''
            old_summary1_line_str = ' '
            for word in summary1_line_list:
                old_summary1_line_str += word + ' '
            old_summary2_line_str = ' '
            for word in summary2_line_list:
                old_summary2_line_str += word + ' '
            old_summary3_line_str = ' '
            for word in summary3_line_list:
                old_summary3_line_str += word + ' '
            old_summary4_line_str = ' '
            for word in summary4_line_list:
                old_summary4_line_str += word + ' '
            new_summary1_line_str = old_summary1_line_str
            new_summary2_line_str = old_summary2_line_str
            new_summary3_line_str = old_summary3_line_str
            new_summary4_line_str = old_summary4_line_str
            for word, (word_, pos) in zip(article_line_list, word_pos_line_list):
                old_line_str += word + ' '
                if word in stanford_ner_tags:
                    new_line_str += word + '_ '
                    stan_repl_art += 1
                elif pos == 'n':
                    flag = True
                    if word_freq_hypernyms_dict.get(word, None):
                        (freq, norm_freq, hypernyms_depth_list) = word_freq_hypernyms_dict[word]
                        if freq < word_freq_thresold + 1:
                            for (hyp, depth) in hypernyms_depth_list:
                                if hyp in wordnet_ner_tags:
                                    new_line_str += hyp + '_ '
                                    wordnet_repl_art += 1
                                    token_for_replacemet = ' {} '.format(word)
                                    token_replace = ' {}_ '.format(hyp)
                                    new_summary1_line_str = new_summary1_line_str.replace(token_for_replacemet,
                                                                                          token_replace)
                                    new_summary2_line_str = new_summary2_line_str.replace(token_for_replacemet,
                                                                                          token_replace)
                                    new_summary3_line_str = new_summary3_line_str.replace(token_for_replacemet,
                                                                                          token_replace)
                                    new_summary4_line_str = new_summary4_line_str.replace(token_for_replacemet,
                                                                                          token_replace)
                                    flag = False
                                    break
                    if flag:
                        new_line_str += word + ' '
                else:
                    new_line_str += word + ' '
            output_article_file.write(new_line_str.strip() + '\n')
            output_summary1_file.write(new_summary1_line_str.strip() + '\n')
            output_summary2_file.write(new_summary2_line_str.strip() + '\n')
            output_summary3_file.write(new_summary3_line_str.strip() + '\n')
            output_summary4_file.write(new_summary4_line_str.strip() + '\n')
            if line_count % print_per_line == 0:
                print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}\n\t'
                      'old_sum: {}\n\tnew_sum: {}\n\told_sum: {}\n\tnew_sum: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                    line_count, old_line_str, new_line_str, old_summary1_line_str, new_summary1_line_str,
                    old_summary2_line_str, new_summary2_line_str, old_summary3_line_str, new_summary3_line_str,
                    old_summary4_line_str, new_summary4_line_str))
                print('\tArticle -> stanford, wordnet and Overall replacements: {}, {} & {}'.format(
                    stan_repl_art, wordnet_repl_art, stan_repl_art + wordnet_repl_art))
        output_article_file.close()
        output_summary1_file.close()
        output_summary2_file.close()
        output_summary3_file.close()
        output_summary4_file.close()
        del article_word_pos_per_line_list
        del article_per_line_list
        print('6/9')

        stan_repl_sum = 0
        wordnet_repl_sum = 0
        line_count = 0
        for output_summary_file_path, input_summary_pos_pickle_file_path \
                in zip([output_summary1_file_path, output_summary2_file_path, output_summary3_file_path,
                        output_summary4_file_path], [input_summary1_pos_pickle_file_path,
                                                     input_summary2_pos_pickle_file_path,
                                                     input_summary3_pos_pickle_file_path,
                                                     input_summary4_pos_pickle_file_path]):

            article_per_line_list = []
            output_article_file = open(output_article_file_path, 'r', encoding='utf8')
            for line in output_article_file:
                article_per_line_list.append(line.split())
            output_article_file.close()

            summary_per_line_list = []
            output_summary_file = open(output_summary_file_path, 'r', encoding='utf8')
            for line in output_summary_file:
                summary_per_line_list.append(line.split())
            output_summary_file.close()
            summary_word_pos_per_line_list = utils.read_pickle_file(input_summary_pos_pickle_file_path)
            print('7/9')
            output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
            output_article_file = open(output_article_file_path, 'w', encoding='utf8')

            for summary_line_list, article_line_list, word_pos_line_list in \
                    zip(summary_per_line_list, article_per_line_list, summary_word_pos_per_line_list):
                line_count += 1
                new_line_str = ''
                old_line_str = ''
                old_article_line_str = ' '
                for word in article_line_list:
                    old_article_line_str += word + ' '
                new_article_line_str = old_article_line_str
                for word, (word_, pos) in zip(summary_line_list, word_pos_line_list):
                    old_line_str += word + ' '
                    if word in stanford_ner_tags:
                        new_line_str += word + '_ '
                        stan_repl_sum += 1
                    elif pos == 'n':
                        flag = True
                        if word_freq_hypernyms_dict.get(word, None):
                            (freq, norm_freq, hypernyms_depth_list) = word_freq_hypernyms_dict[word]
                            if freq < word_freq_thresold + 1:
                                for (hyp, depth) in hypernyms_depth_list:
                                    if hyp in wordnet_ner_tags:
                                        new_line_str += hyp + '_ '
                                        token_for_replacement = ' {} '.format(word)
                                        token_replace = ' {}_ '.format(hyp)
                                        new_article_line_str = new_article_line_str.replace(token_for_replacement,
                                                                                            token_replace)
                                        wordnet_repl_sum += 1
                                        flag = False
                                        break
                        if flag:
                            new_line_str += word + ' '
                    else:
                        new_line_str += word + ' '
                output_summary_file.write(new_line_str.strip() + '\n')
                output_article_file.write(new_article_line_str.strip() + '\n')
                if line_count % print_per_line == 0:
                    print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                        line_count, old_article_line_str, new_article_line_str, old_line_str, new_line_str))
                    print('\tSummary -> stanford, wordnet and Overall replacements: {}, {} & {}'.format(
                        stan_repl_sum, wordnet_repl_sum, stan_repl_sum + wordnet_repl_sum))
            output_summary_file.close()
            output_article_file.close()
            del summary_word_pos_per_line_list
            del summary_per_line_list
            print('8/9')
        print(
            '\nOverall replacements:\n\tArticle -> stanford, wordnet and Overall replacements: {}, {} & {}'.format(
                stan_repl_art, wordnet_repl_art, stan_repl_art + wordnet_repl_art))

        print(
            '\tSummary -> stanford, wordnet and overall replacements: {}, {} & {}'.format(
                stan_repl_sum, wordnet_repl_sum, stan_repl_sum + wordnet_repl_sum))

        print('\tArticle and Summary -> stanford, wordnet and overall replacements {}, {}, {}'.format(
            stan_repl_art + stan_repl_sum,
            wordnet_repl_art + wordnet_repl_sum,
            stan_repl_art + stan_repl_sum +
            wordnet_repl_art + wordnet_repl_sum
        ))
        print('Output files of threshold {}:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(
            word_freq_thresold, output_article_file_path, output_summary1_file_path, output_summary2_file_path,
            output_summary3_file_path, output_summary4_file_path))
        print('9/9')

    def conver_dataset_with_ner_from_stanford_and_wordnet(self, input_article_ner_pickle_file_path,
                                                          input_summary_ner_pickle_file_path,
                                                          input_article_pos_pickle_file_path,
                                                          input_summary_pos_pickle_file_path,
                                                          input_word_freq_hypernyms_pickle_file_path,
                                                          output_article_file_path, output_summary_file_path,
                                                          word_freq_thresold=9999990,
                                                          print_per_line=100000):
        word_freq_hypernyms_dict = utils.read_pickle_file(input_word_freq_hypernyms_pickle_file_path)
        article_word_ner_per_line_list = utils.read_pickle_file(input_article_ner_pickle_file_path)
        print('1/9')
        summary_word_ner_per_line_list = utils.read_pickle_file(input_summary_ner_pickle_file_path)
        print('2/9')
        stanford_ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
        output_article_file = open(output_article_file_path, 'w', encoding='utf8')
        output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
        count_article_driven_replacements = 0
        count_summary_driven_replacements = 0
        line_count = 0
        for article_line_list, summary_line_list in zip(article_word_ner_per_line_list, summary_word_ner_per_line_list):
            line_count += 1
            article_line = ' '
            summary_line = ' '
            for (article_token, article_ner) in article_line_list:
                article_line += article_token + ' '
            for (summary_token, summary_ner) in summary_line_list:
                summary_line += summary_token + ' '
            new_article_line = article_line
            new_summary_line = summary_line
            for (article_token, article_ner) in article_line_list:
                if article_ner in stanford_ner_tags:
                    change_flag = True
                    if word_freq_hypernyms_dict.get(article_token, None):
                        (freq, norm_freq, hypernyms_path_list) = word_freq_hypernyms_dict[article_token]
                        if freq > word_freq_thresold:
                            change_flag = False
                    if change_flag:
                        token_find = ' {} '.format(article_token)
                        find_index = new_summary_line.find(token_find)
                        if find_index > -1:
                            token_replace = ' {} '.format(article_ner)
                            new_summary_line = new_summary_line.replace(token_find, token_replace)
                            new_article_line = new_article_line.replace(token_find, token_replace)
                            count_article_driven_replacements += 1
            for (summary_token, summary_ner) in summary_line_list:
                if summary_ner in stanford_ner_tags:
                    change_flag = True
                    if word_freq_hypernyms_dict.get(summary_token, None):
                        (freq, norm_freq, hypernyms_path_list) = word_freq_hypernyms_dict[summary_token]
                        if freq > word_freq_thresold:
                            change_flag = False
                    if change_flag:
                        token_find = ' {} '.format(summary_token)
                        find_index = new_article_line.find(token_find)
                        if find_index > -1:
                            token_replace = ' {} '.format(summary_ner)
                            new_article_line = new_article_line.replace(token_find, token_replace)
                            new_summary_line = new_summary_line.replace(token_find, token_replace)
                            count_summary_driven_replacements += 1
            output_article_file.write(new_article_line.strip() + '\n')
            output_summary_file.write(new_summary_line.strip() + '\n')
            if line_count % print_per_line == 0:
                print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                    line_count, article_line, new_article_line, summary_line, new_summary_line))
                print(
                    '\tarticle driven replacements: {}\n\tSummary driven replacements: {}\n'
                    '\tOverall replacements: {}'.format(
                        count_article_driven_replacements, count_summary_driven_replacements,
                        count_article_driven_replacements + count_summary_driven_replacements))
        del article_word_ner_per_line_list
        del summary_word_ner_per_line_list
        output_article_file.close()
        output_summary_file.close()
        print('3/9')
        print('\narticle driven replacements: {}\nSummary driven replacements: {}\nOverall replacements: {}'.format(
            count_article_driven_replacements, count_summary_driven_replacements,
            count_article_driven_replacements + count_summary_driven_replacements))

        wordnet_ner_tags = ['person', 'location', 'organization']

        article_per_line_list = []
        output_article_file = open(output_article_file_path, 'r', encoding='utf8')
        for line in output_article_file:
            article_per_line_list.append(line.split())
        output_article_file.close()
        summary_per_line_list = []
        output_summary_file = open(output_summary_file_path, 'r', encoding='utf8')
        for line in output_summary_file:
            summary_per_line_list.append(line.split())
        output_summary_file.close()

        print('4/9')
        article_word_pos_per_line_list = utils.read_pickle_file(input_article_pos_pickle_file_path)
        print('5/9')
        output_article_file = open(output_article_file_path, 'w', encoding='utf8')
        output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
        stan_repl_art = 0
        wordnet_repl_art = 0
        line_count = 0
        for article_line_list, summary_line_list, word_pos_line_list in \
                zip(article_per_line_list, summary_per_line_list, article_word_pos_per_line_list):
            # for line_list in article_per_line_list:
            line_count += 1
            new_line_str = ''
            old_line_str = ''
            old_summary_line_str = ' '
            for word in summary_line_list:
                old_summary_line_str += word + ' '
            new_summary_line_str = old_summary_line_str
            for word, (word_, pos) in zip(article_line_list, word_pos_line_list):
                old_line_str += word + ' '
                if word in stanford_ner_tags:
                    new_line_str += word + '_ '
                    stan_repl_art += 1
                elif pos == 'n':
                    flag = True
                    if word_freq_hypernyms_dict.get(word, None):
                        (freq, norm_freq, hypernyms_depth_list) = word_freq_hypernyms_dict[word]
                        if freq < word_freq_thresold + 1:
                            for (hyp, depth) in hypernyms_depth_list:
                                if hyp in wordnet_ner_tags:
                                    new_line_str += hyp + '_ '
                                    wordnet_repl_art += 1
                                    token_for_replacemet = ' {} '.format(word)
                                    token_replace = ' {}_ '.format(hyp)
                                    new_summary_line_str = new_summary_line_str.replace(token_for_replacemet,
                                                                                        token_replace)
                                    flag = False
                                    break
                    if flag:
                        new_line_str += word + ' '
                else:
                    new_line_str += word + ' '
            output_article_file.write(new_line_str.strip() + '\n')
            output_summary_file.write(new_summary_line_str.strip() + '\n')
            if line_count % print_per_line == 0:
                print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                    line_count, old_line_str, new_line_str, old_summary_line_str, new_summary_line_str))
                print('\tArticle -> stanford, wordnet and Overall replacements: {}, {} & {}'.format(
                    stan_repl_art, wordnet_repl_art, stan_repl_art + wordnet_repl_art))
        output_article_file.close()
        output_summary_file.close()
        del article_word_pos_per_line_list
        del article_per_line_list
        print('6/9')

        article_per_line_list = []
        output_article_file = open(output_article_file_path, 'r', encoding='utf8')
        for line in output_article_file:
            article_per_line_list.append(line.split())
        output_article_file.close()
        summary_per_line_list = []
        output_summary_file = open(output_summary_file_path, 'r', encoding='utf8')
        for line in output_summary_file:
            summary_per_line_list.append(line.split())
        output_summary_file.close()
        summary_word_pos_per_line_list = utils.read_pickle_file(input_summary_pos_pickle_file_path)
        print('7/9')
        output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
        output_article_file = open(output_article_file_path, 'w', encoding='utf8')
        stan_repl_sum = 0
        wordnet_repl_sum = 0
        line_count = 0
        for summary_line_list, article_line_list, word_pos_line_list in \
                zip(summary_per_line_list, article_per_line_list, summary_word_pos_per_line_list):
            line_count += 1
            new_line_str = ''
            old_line_str = ''
            old_article_line_str = ' '
            for word in article_line_list:
                old_article_line_str += word + ' '
            new_article_line_str = old_article_line_str
            for word, (word_, pos) in zip(summary_line_list, word_pos_line_list):
                old_line_str += word + ' '
                if word in stanford_ner_tags:
                    new_line_str += word + '_ '
                    stan_repl_sum += 1
                elif pos == 'n':
                    flag = True
                    if word_freq_hypernyms_dict.get(word, None):
                        (freq, norm_freq, hypernyms_depth_list) = word_freq_hypernyms_dict[word]
                        if freq < word_freq_thresold + 1:
                            for (hyp, depth) in hypernyms_depth_list:
                                if hyp in wordnet_ner_tags:
                                    new_line_str += hyp + '_ '
                                    token_for_replacement = ' {} '.format(word)
                                    token_replace = ' {}_ '.format(hyp)
                                    new_article_line_str = new_article_line_str.replace(token_for_replacement,
                                                                                        token_replace)
                                    wordnet_repl_sum += 1
                                    flag = False
                                    break
                    if flag:
                        new_line_str += word + ' '
                else:
                    new_line_str += word + ' '
            output_summary_file.write(new_line_str.strip() + '\n')
            output_article_file.write(new_article_line_str.strip() + '\n')
            if line_count % print_per_line == 0:
                print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                    line_count, old_article_line_str, new_article_line_str, old_line_str, new_line_str))
                print('\tSummary -> stanford, wordnet and Overall replacements: {}, {} & {}'.format(
                    stan_repl_sum, wordnet_repl_sum, stan_repl_sum + wordnet_repl_sum))
        output_summary_file.close()
        del summary_word_pos_per_line_list
        del summary_per_line_list
        print('8/9')
        print(
            '\nOverall replacements:\n\tArticle -> stanford, wordnet and Overall replacements: {}, {} & {}'.format(
                stan_repl_art, wordnet_repl_art, stan_repl_art + wordnet_repl_art))

        print(
            '\tSummary -> stanford, wordnet and overall replacements: {}, {} & {}'.format(
                stan_repl_sum, wordnet_repl_sum, stan_repl_sum + wordnet_repl_sum))

        print('\tArticle and Summary -> stanford, wordnet and overall replacements {}, {}, {}'.format(
            stan_repl_art + stan_repl_sum,
            wordnet_repl_art + wordnet_repl_sum,
            stan_repl_art + stan_repl_sum +
            wordnet_repl_art + wordnet_repl_sum
        ))
        print('Output files of threshold {}:\n\t{}\n\t{}'.format(
            word_freq_thresold, output_article_file_path, output_summary_file_path))
        print('9/9')

    def convert_duc_dataset_based_on_level_of_generalizetion(self,
                                                             article_word_pos_line_list_pickle_file_path,
                                                             ref1_word_pos_line_list_pickle_file_path,
                                                             ref2_word_pos_line_list_pickle_file_path,
                                                             ref3_word_pos_line_list_pickle_file_path,
                                                             ref4_word_pos_line_list_pickle_file_path,
                                                             word_hypernym_dict_pickle_file_path,
                                                             output_articles_file_path,
                                                             output_ref1_file_path,
                                                             output_ref2_file_path,
                                                             output_ref3_file_path,
                                                             output_ref4_file_path):
        word_hypernym_dict = utils.read_pickle_file(word_hypernym_dict_pickle_file_path)

        output_articles_file = open(output_articles_file_path, 'w', encoding='utf8')
        articles_word_pos_list_per_line = utils.read_pickle_file(article_word_pos_line_list_pickle_file_path)
        article_changes = 0
        words_set = set()
        hypernyms_set = set()
        for line_list in articles_word_pos_list_per_line:
            line_text = ''
            for (word, pos) in line_list:
                if pos is 'n':
                    words_set.add(word)
                    try:
                        hypernym = word_hypernym_dict[word]
                        hypernyms_set.add(hypernym)
                        if word != hypernym:
                            line_text += hypernym + '_ '
                            article_changes += 1
                        else:
                            line_text += word + ' '
                    except KeyError:
                        line_text += word + ' '
                else:
                    line_text += word + ' '
            output_articles_file.write(line_text.strip() + '\n')
        output_articles_file.close()
        del articles_word_pos_list_per_line

        print('(Articles) Distinct nouns: {}, Changes with distinct hypernyms: {}'.format(len(words_set),
                                                                                          len(hypernyms_set)))

        summary1_changes = 0
        output_summaries_file = open(output_ref1_file_path, 'w', encoding='utf8')
        summaries_word_pos_list_per_line = utils.read_pickle_file(ref1_word_pos_line_list_pickle_file_path)
        for line_list in summaries_word_pos_list_per_line:
            line_text = ''
            for (word, pos) in line_list:
                if pos is 'n':
                    words_set.add(word)
                    try:
                        hypernym = word_hypernym_dict[word]
                        hypernyms_set.add(hypernym)
                        if word != hypernym:
                            line_text += hypernym + '_ '
                            summary1_changes += 1

                        else:
                            line_text += word + ' '
                    except KeyError:
                        line_text += word + ' '
                else:
                    line_text += word + ' '
            output_summaries_file.write(line_text.strip() + '\n')

        print('(Articles & summaries) Distinct nouns: {}, Changes with distinct hypernyms: {}'.format(len(words_set),
                                                                                                      len(
                                                                                                          hypernyms_set)))

        del summaries_word_pos_list_per_line
        output_summaries_file.close()

        summary2_changes = 0
        output_summaries_file = open(output_ref2_file_path, 'w', encoding='utf8')
        summaries_word_pos_list_per_line = utils.read_pickle_file(ref2_word_pos_line_list_pickle_file_path)
        for line_list in summaries_word_pos_list_per_line:
            line_text = ''
            for (word, pos) in line_list:
                if pos is 'n':
                    words_set.add(word)
                    try:
                        hypernym = word_hypernym_dict[word]
                        hypernyms_set.add(hypernym)
                        if word != hypernym:
                            line_text += hypernym + '_ '
                            summary2_changes += 1

                        else:
                            line_text += word + ' '
                    except KeyError:
                        line_text += word + ' '
                else:
                    line_text += word + ' '
            output_summaries_file.write(line_text.strip() + '\n')
        del summaries_word_pos_list_per_line
        output_summaries_file.close()

        summary3_changes = 0
        output_summaries_file = open(output_ref3_file_path, 'w', encoding='utf8')
        summaries_word_pos_list_per_line = utils.read_pickle_file(ref3_word_pos_line_list_pickle_file_path)
        for line_list in summaries_word_pos_list_per_line:
            line_text = ''
            for (word, pos) in line_list:
                if pos is 'n':
                    words_set.add(word)
                    try:
                        hypernym = word_hypernym_dict[word]
                        hypernyms_set.add(hypernym)
                        if word != hypernym:
                            line_text += hypernym + '_ '
                            summary3_changes += 1

                        else:
                            line_text += word + ' '
                    except KeyError:
                        line_text += word + ' '
                else:
                    line_text += word + ' '
            output_summaries_file.write(line_text.strip() + '\n')
        del summaries_word_pos_list_per_line
        output_summaries_file.close()

        summary4_changes = 0
        output_summaries_file = open(output_ref4_file_path, 'w', encoding='utf8')
        summaries_word_pos_list_per_line = utils.read_pickle_file(ref4_word_pos_line_list_pickle_file_path)
        for line_list in summaries_word_pos_list_per_line:
            line_text = ''
            for (word, pos) in line_list:
                if pos is 'n':
                    words_set.add(word)
                    try:
                        hypernym = word_hypernym_dict[word]
                        hypernyms_set.add(hypernym)
                        if word != hypernym:
                            line_text += hypernym + '_ '
                            summary4_changes += 1

                        else:
                            line_text += word + ' '
                    except KeyError:
                        line_text += word + ' '
                else:
                    line_text += word + ' '
            output_summaries_file.write(line_text.strip() + '\n')
        del summaries_word_pos_list_per_line
        output_summaries_file.close()

        print('article changes: {}\n'
              'Summary1 changes {}\nSummary2 changes {}\nSummary3 changes {}\nSummary4 changes {}\n'
              'Overall changes: {}'.format(article_changes,
                                           summary1_changes, summary2_changes, summary3_changes, summary4_changes,
                                           article_changes + summary1_changes + summary2_changes +
                                           summary3_changes + summary4_changes))

    def convert_dataset_to_general(self, article_word_pos_line_list_pickle_file_path,
                                   summary_word_pos_line_list_pickle_file_path,
                                   word_hypernym_dict_pickle_file_path,
                                   output_articles_file_path, output_summaries_file_path):
        word_hypernym_dict = utils.read_pickle_file(word_hypernym_dict_pickle_file_path)

        output_articles_file = open(output_articles_file_path, 'w', encoding='utf8')
        articles_word_pos_list_per_line = utils.read_pickle_file(article_word_pos_line_list_pickle_file_path)
        article_changes = 0
        words_set = set()
        hypernyms_set = set()
        for line_list in articles_word_pos_list_per_line:
            line_text = ''
            for (word, pos) in line_list:
                if pos is 'n':
                    words_set.add(word)
                    try:
                        hypernym = word_hypernym_dict[word]
                        hypernyms_set.add(hypernym)
                        if word != hypernym:
                            line_text += hypernym + '_ '
                            article_changes += 1
                        else:
                            line_text += word + ' '
                    except KeyError:
                        line_text += word + ' '
                else:
                    line_text += word + ' '
            output_articles_file.write(line_text.strip() + '\n')
        output_articles_file.close()
        del articles_word_pos_list_per_line

        print('(Articles) Distinct nouns: {}, Changes with distinct hypernyms: {}'.format(len(words_set),
                                                                                          len(hypernyms_set)))

        summary_changes = 0
        output_summaries_file = open(output_summaries_file_path, 'w', encoding='utf8')
        summaries_word_pos_list_per_line = utils.read_pickle_file(summary_word_pos_line_list_pickle_file_path)
        for line_list in summaries_word_pos_list_per_line:
            line_text = ''
            for (word, pos) in line_list:
                if pos is 'n':
                    words_set.add(word)
                    try:
                        hypernym = word_hypernym_dict[word]
                        hypernyms_set.add(hypernym)
                        if word != hypernym:
                            line_text += hypernym + '_ '
                            summary_changes += 1

                        else:
                            line_text += word + ' '
                    except KeyError:
                        line_text += word + ' '
                else:
                    line_text += word + ' '
            output_summaries_file.write(line_text.strip() + '\n')

        print('(Articles & summaries) Distinct nouns: {}, Changes with distinct hypernyms: {}'.format(len(words_set),
                                                                                                      len(
                                                                                                          hypernyms_set)))

        del summaries_word_pos_list_per_line
        output_summaries_file.close()
        print('article changes: {}\nSummary changes {}\n'
              'Overall changes: {}'.format(article_changes, summary_changes, article_changes + summary_changes))

    def vocab_based_on_hypernyms(self, input_word_freq_hypernyms_pickle_file_path,
                                 output_hypernym_freq_wordlist_remaininghypernyms_txt_file_path,
                                 output_hypernym_freq_wordlist_remaininghypernyms_pickle_file_path,
                                 output_word_hypernym_dict_pickle_file,
                                 output_word_hypernym_dict_txt_file,
                                 upper_word_freq_thres=9999000, min_depth=5):
        word_freq_hypernyms_dict = utils.read_pickle_file(input_word_freq_hypernyms_pickle_file_path)
        hypernym_freq_wordlist_remaininghypernyms_dict = dict()
        for word, (freq, norm_freq, hypernyms_list) in word_freq_hypernyms_dict.items():
            hypernym_freq_wordlist_remaininghypernyms_dict[word] = (freq, norm_freq, [word], hypernyms_list)
        flag = True
        while flag:
            flag = False
            temp_dict = dict()
            # count_words = 0
            for k, v in hypernym_freq_wordlist_remaininghypernyms_dict.items():
                temp_dict[k] = v
                # count_words += len(v[2])
            # print(count_words)
            # count_changes = 0
            for word, (freq, norm_freq, wordlist, hypernyms_list) in temp_dict.items():
                if freq < upper_word_freq_thres + 1:
                    (hypernym, depth) = hypernyms_list[0]
                    if depth >= min_depth:
                        flag = True
                        # count_changes += 1
                        if hypernym_freq_wordlist_remaininghypernyms_dict.get(hypernym, None):
                            (new_freq, new_norm_freq, new_word_list, new_hypernyms_list) = \
                                hypernym_freq_wordlist_remaininghypernyms_dict[hypernym]
                            (freq, norm_freq, wordlist, hypernyms_list) = \
                                hypernym_freq_wordlist_remaininghypernyms_dict.pop(word)
                            log_max_freq = np.log10(new_freq + 1) / new_norm_freq
                            new_word_list = list(set([word] + wordlist + new_word_list))
                            new_freq += freq
                            # hypernyms_list = []
                            # hypernyms_list.remove((hypernym, depth))
                            # if new_norm_freq > 0.99999:
                            #    new_norm_freq = 1.0
                            # else:
                            new_norm_freq = np.log10(new_freq + 1) / log_max_freq
                            merged_hypernyms_list = list(set(hypernyms_list + new_hypernyms_list))
                            merged_hypernyms_list = utils.sort_by_second(merged_hypernyms_list, descending=True)
                            try:
                                merged_hypernyms_list.remove((hypernym, depth))
                            except ValueError:
                                _ = None
                            hypernym_freq_wordlist_remaininghypernyms_dict[hypernym] = (
                                new_freq, new_norm_freq, new_word_list, merged_hypernyms_list)
                        else:

                            (freq, norm_freq, wordlist, hypernyms_list) = \
                                hypernym_freq_wordlist_remaininghypernyms_dict.pop(word)
                            hypernyms_list.remove((hypernym, depth))
                            # hypernym_freq_wordlist_remaininghypernyms_dict.pop(word)
                            hypernym_freq_wordlist_remaininghypernyms_dict[hypernym] = (
                                freq, norm_freq, list(set([word] + wordlist)), hypernyms_list)
            # print(count_changes)
            # count_words = 0
            ##############################################
            hypernym_set = set()
            word_list_set = set()
            for word, (new_freq, new_norm_freq, new_word_list,
                       new_hypernyms_list) in hypernym_freq_wordlist_remaininghypernyms_dict.items():
                hypernym_set.add(word)
                for el in new_word_list:
                    word_list_set.add(el)
            print('Generalized & overall words: {} {}'.format(len(hypernym_set), len(word_list_set)))
            ####################################

        hypernym_freq_wordlist_remaininghypernyms_list = []
        for word, (new_freq, new_norm_freq, new_word_list,
                   new_hypernyms_list) in hypernym_freq_wordlist_remaininghypernyms_dict.items():
            # count_words += len(new_word_list)
            hypernym_freq_wordlist_remaininghypernyms_list.append(
                (word, new_freq, new_norm_freq, new_word_list, new_hypernyms_list))

        # print(count_words)
        hypernym_freq_wordlist_remaininghypernyms_list = utils.sort_by_second(
            hypernym_freq_wordlist_remaininghypernyms_list, descending=True)
        with open(output_hypernym_freq_wordlist_remaininghypernyms_txt_file_path, 'w', encoding='utf8') as f:
            for (word, new_freq, new_norm_freq, new_word_list,
                 new_hypernyms_list) in hypernym_freq_wordlist_remaininghypernyms_list:
                f.write('{} {} {} {} {}\n'.format(word, new_freq, new_norm_freq, new_word_list, new_hypernyms_list))
        with open(output_hypernym_freq_wordlist_remaininghypernyms_pickle_file_path, 'wb') as f:
            pickle.dump(hypernym_freq_wordlist_remaininghypernyms_dict, f)

        word_hypernym_dict = dict()
        count_words = 0
        for hypernym, (freq, norm_freq, word_list,
                       hypernyms_list) in hypernym_freq_wordlist_remaininghypernyms_dict.items():
            # count_words += len(word_list)
            for w in word_list:
                if word_hypernym_dict.get(w, None):
                    count_words += 1
                    hyp1 = word_hypernym_dict[w]
                    (freq1, norm_freq, _, _) = hypernym_freq_wordlist_remaininghypernyms_dict[hyp1]
                    (freq2, norm_freq, _, _) = hypernym_freq_wordlist_remaininghypernyms_dict[hypernym]
                    if freq2 > freq1:
                        word_hypernym_dict[w] = hypernym
                else:
                    word_hypernym_dict[w] = hypernym
        print('same w: ', count_words)
        with open(output_word_hypernym_dict_pickle_file, 'wb') as f:
            pickle.dump(word_hypernym_dict, f)
        with open(output_word_hypernym_dict_txt_file, 'w', encoding='utf8') as f:
            for w, h in word_hypernym_dict.items():
                f.write('{} {}\n'.format(w, h))
        # print('words: {}'.format(count_words))

    def word_freq_hypernym_paths(self, input_word_pos_freq_dict_file_path,
                                 output_word_freq_hypernyms_txt_file_path,
                                 output_word_freq_hypernyms_pickle_file_path):

        word_pos_freq_dict = utils.read_pickle_file(input_word_pos_freq_dict_file_path)
        word_freq_hypernyms_dict = dict()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        max_freq = 0
        for (token, pos), freq in word_pos_freq_dict.items():
            if pos == 'n':
                token_lemma = lemmatizer.lemmatize(token, pos='n')
                synset = self.make_synset(token_lemma, category='n')
                if synset is not None:
                    synset.max_depth()
                    merged_synset_list = self.merge_lists(synset.hypernym_paths())
                    sorted_synsets = self.syncet_sort_accornding_max_depth(merged_synset_list)
                    word_depth_list = self.word_depth_of_synsents(sorted_synsets)
                    if word_depth_list[0][0] != token_lemma:
                        word_depth_list = [(token_lemma, word_depth_list[0][1] + 1)] + word_depth_list
                    word_freq_hypernyms_dict[token] = (freq, word_depth_list)
                    if freq > max_freq:
                        max_freq = freq
        word_freq_hypernyms_list = []
        word_freq_normfreq_hypernyms_dict = dict()
        for word, (freq, hypernym_depth_list) in word_freq_hypernyms_dict.items():
            norm_freq = np.log10(freq + 1) / np.log10(max_freq + 1)
            word_freq_hypernyms_list.append((word, freq, norm_freq, hypernym_depth_list))
            word_freq_normfreq_hypernyms_dict[word] = (freq, norm_freq, hypernym_depth_list)
        del word_freq_hypernyms_dict
        word_freq_hypernyms_list = utils.sort_by_second(word_freq_hypernyms_list, descending=True)
        with open(output_word_freq_hypernyms_txt_file_path, 'w', encoding='utf8') as f:
            for (word, freq, norm_freq, hypernym_depth_list) in word_freq_hypernyms_list:
                f.write('{} {} {} {}\n'.format(word, freq, norm_freq, hypernym_depth_list))
        with open(output_word_freq_hypernyms_pickle_file_path, 'wb') as f:
            pickle.dump(word_freq_normfreq_hypernyms_dict, f)

    def conver_dataset_with_ner(self, input_article_ner_pickle_file_path, input_summary_ner_pickle_file_path,
                                output_article_file_path, output_summary_file_path,
                                print_per_line=200):
        article_word_ner_per_line_list = utils.read_pickle_file(input_article_ner_pickle_file_path)
        summary_word_ner_per_line_list = utils.read_pickle_file(input_summary_ner_pickle_file_path)
        ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
        output_article_file = open(output_article_file_path, 'w', encoding='utf8')
        output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
        count_article_driven_replacements = 0
        count_summary_driven_replacements = 0
        line_count = 0
        for article_line_list, summary_line_list in zip(article_word_ner_per_line_list, summary_word_ner_per_line_list):
            line_count += 1
            article_line = ' '
            summary_line = ' '
            for (article_token, article_ner) in article_line_list:
                article_line += article_token + ' '
            for (summary_token, summary_ner) in summary_line_list:
                summary_line += summary_token + ' '
            new_article_line = article_line
            new_summary_line = summary_line
            for (article_token, article_ner) in article_line_list:
                if article_ner in ner_tags:
                    # summary_line_list = ['alpha', 'beta', 'gama', 'delta']
                    token_find = ' {} '.format(article_token)
                    find_index = new_summary_line.find(token_find)
                    if find_index > -1:
                        token_replace = ' {}_ '.format(article_ner)
                        new_summary_line = new_summary_line.replace(token_find, token_replace)
                        new_article_line = new_article_line.replace(token_find, token_replace)
                        count_article_driven_replacements += 1
            for (summary_token, summary_ner) in summary_line_list:
                if summary_ner in ner_tags:
                    # summary_line_list = ['alpha', 'beta', 'gama', 'delta']
                    token_find = ' {} '.format(summary_token)
                    find_index = new_article_line.find(token_find)
                    if find_index > -1:
                        token_replace = ' {}_ '.format(summary_ner)
                        new_article_line = new_article_line.replace(token_find, token_replace)
                        new_summary_line = new_summary_line.replace(token_find, token_replace)
                        count_summary_driven_replacements += 1
            output_article_file.write(new_article_line.strip() + '\n')
            output_summary_file.write(new_summary_line.strip() + '\n')
            if line_count % print_per_line == 0:
                print('{} line:\n\told_art: {}\n\tnew_art: {}\n\told_sum: {}\n\tnew_sum: {}'.format(
                    line_count, article_line, new_article_line, summary_line, new_summary_line))
                print(
                    '\tarticle driven replacements: {}\n\tSummary driven replacements: {}\n'
                    '\tOverall replacements: {}'.format(
                        count_article_driven_replacements, count_summary_driven_replacements,
                        count_article_driven_replacements + count_summary_driven_replacements))
        output_article_file.close()
        output_summary_file.close()
        print('\narticle driven replacements: {}\nSummary driven replacements: {}\nOverall replacements: {}'.format(
            count_article_driven_replacements, count_summary_driven_replacements,
            count_article_driven_replacements + count_summary_driven_replacements))

    def ner_of_dataset_and_vocabulary_of_ner_words(self, input_article_file_path,
                                                   input_summary_file_path,
                                                   output_article_word_ner_line_list_sample_txt_file,
                                                   output_summary_word_ner_line_list_sample_txt_file,
                                                   output_article_word_ner_line_list_pickle_file,
                                                   output_summary_word_ner_line_list_pickle_file,
                                                   output_word_ner_freq_dict_txt_file,
                                                   output_word_ner_freq_dict_pickle_file,
                                                   lines_per_ner_application, print_per_line=100,
                                                   lines_of_sample_files=1000):
        t0 = time.time()
        input_article_batch_text_list = []
        input_summary_batch_text_list = []
        ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
        print('Reading files:\n\t{}\n\t{}'.format(input_article_file_path, input_summary_file_path))
        with open(input_article_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_ner_application:
                    input_article_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_article_batch_text_list.append(input_temp_list)
            f.close()
        print('Article data have been loaded on batch list')

        word_ner_freq_dict = dict()
        article_word_ner_per_line_list = []
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_article_batch_text_list:
            line_index += lines_per_ner_application
            word_ner_list = self.stanford_ner(batch_text_list)
            line_word_ner_list = []
            for (word, ner) in word_ner_list:
                if word != 'NL_':
                    line_word_ner_list.append((word, ner))
                    if ner in ner_tags:
                        try:
                            new_freq = word_ner_freq_dict[(word, ner)] + 1
                            word_ner_freq_dict[(word, ner)] = new_freq
                        except KeyError:
                            word_ner_freq_dict[(word, ner)] = 1
                else:
                    article_word_ner_per_line_list.append(line_word_ner_list)
                    line_word_ner_list = []
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        del input_article_batch_text_list
        print('Article ner tags have been obtained.')

        with open(output_article_word_ner_line_list_sample_txt_file, 'w', encoding='utf8') as f:
            index = 0
            for word_ner_line_list in article_word_ner_per_line_list:
                f.write('{}\n'.format(word_ner_line_list))
                index += 1
                if index == lines_of_sample_files:
                    break
        with open(output_article_word_ner_line_list_pickle_file, 'wb') as f:
            pickle.dump(article_word_ner_per_line_list, f)
        del article_word_ner_per_line_list
        print('Article files have been created:\n\t{}\n\t{}'.format(
            output_article_word_ner_line_list_sample_txt_file, output_article_word_ner_line_list_pickle_file))

        with open(input_summary_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_ner_application:
                    input_summary_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_summary_batch_text_list.append(input_temp_list)
            f.close()
        print('Summary data have been loaded on batch list')

        t1 = time.time()
        line_index = 0
        summary_word_ner_per_line_list = []
        for batch_text_list in input_summary_batch_text_list:
            line_index += lines_per_ner_application
            word_ner_list = self.stanford_ner(batch_text_list)
            line_word_ner_list = []
            for (word, ner) in word_ner_list:
                if word != 'NL_':
                    line_word_ner_list.append((word, ner))
                    if ner in ner_tags:
                        try:
                            new_freq = word_ner_freq_dict[(word, ner)] + 1
                            word_ner_freq_dict[(word, ner)] = new_freq
                        except KeyError:
                            word_ner_freq_dict[(word, ner)] = 1
                else:
                    summary_word_ner_per_line_list.append(line_word_ner_list)
                    line_word_ner_list = []
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        print('Summary ner tags have been obtained.')
        del input_summary_batch_text_list
        with open(output_summary_word_ner_line_list_sample_txt_file, 'w', encoding='utf8') as f:
            index = 0
            for word_pos_line_list in summary_word_ner_per_line_list:
                f.write('{}\n'.format(word_pos_line_list))
                index += 1
                if index == lines_of_sample_files:
                    break
        with open(output_summary_word_ner_line_list_pickle_file, 'wb') as f:
            pickle.dump(summary_word_ner_per_line_list, f)
        print('Summary files have been created:\n\t{}\n\t{}'.format(
            output_summary_word_ner_line_list_sample_txt_file, output_summary_word_ner_line_list_pickle_file))
        del summary_word_ner_per_line_list

        word_pos_freq_list = []
        for k, v in word_ner_freq_dict.items():
            word_pos_freq_list.append((k, v))
        word_pos_freq_list = sorted(word_pos_freq_list, key=lambda tup: -tup[1])
        with open(output_word_ner_freq_dict_txt_file, 'w', encoding='utf8') as f:
            for ((w, p), freq) in word_pos_freq_list:
                f.write('{} {} {}\n'.format(w, p, freq))
        with open(output_word_ner_freq_dict_pickle_file, 'wb') as f:
            pickle.dump(word_ner_freq_dict, f)
        print('Word-pos-freq files have been created:\n\t{}\n\t{}'.format(
            output_word_ner_freq_dict_txt_file, output_word_ner_freq_dict_pickle_file))

    def ner_of_duc_dataset_and_vocab_of_ne(self, input_article_file_path,
                                           input_summary1_file_path,
                                           input_summary2_file_path,
                                           input_summary3_file_path,
                                           input_summary4_file_path,
                                           output_article_word_ner_line_list_sample_txt_file,
                                           output_summary1_word_ner_line_list_sample_txt_file,
                                           output_summary2_word_ner_line_list_sample_txt_file,
                                           output_summary3_word_ner_line_list_sample_txt_file,
                                           output_summary4_word_ner_line_list_sample_txt_file,
                                           output_article_word_ner_line_list_pickle_file,
                                           output_summary1_word_ner_line_list_pickle_file,
                                           output_summary2_word_ner_line_list_pickle_file,
                                           output_summary3_word_ner_line_list_pickle_file,
                                           output_summary4_word_ner_line_list_pickle_file,
                                           output_word_ner_freq_dict_txt_file,
                                           output_word_ner_freq_dict_pickle_file,
                                           lines_per_ner_application, print_per_line=100,
                                           lines_of_sample_files=1000):
        t0 = time.time()
        input_article_batch_text_list = []
        ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
        print('Reading files:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(input_article_file_path, input_summary1_file_path,
                                                                    input_summary2_file_path,
                                                                    input_summary3_file_path,
                                                                    input_summary4_file_path))
        with open(input_article_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_ner_application:
                    input_article_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_article_batch_text_list.append(input_temp_list)
            f.close()
        print('Article data have been loaded on batch list')

        word_ner_freq_dict = dict()
        article_word_ner_per_line_list = []
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_article_batch_text_list:
            line_index += lines_per_ner_application
            word_ner_list = self.stanford_ner(batch_text_list)
            line_word_ner_list = []
            for (word, ner) in word_ner_list:
                if word != 'NL_':
                    line_word_ner_list.append((word, ner))
                    if ner in ner_tags:
                        try:
                            new_freq = word_ner_freq_dict[(word, ner)] + 1
                            word_ner_freq_dict[(word, ner)] = new_freq
                        except KeyError:
                            word_ner_freq_dict[(word, ner)] = 1
                else:
                    article_word_ner_per_line_list.append(line_word_ner_list)
                    line_word_ner_list = []
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        del input_article_batch_text_list
        print('Article ner tags have been obtained.')

        with open(output_article_word_ner_line_list_sample_txt_file, 'w', encoding='utf8') as f:
            index = 0
            for word_ner_line_list in article_word_ner_per_line_list:
                f.write('{}\n'.format(word_ner_line_list))
                index += 1
                if index == lines_of_sample_files:
                    break
        with open(output_article_word_ner_line_list_pickle_file, 'wb') as f:
            pickle.dump(article_word_ner_per_line_list, f)
        del article_word_ner_per_line_list
        print('Article files have been created:\n\t{}\n\t{}'.format(
            output_article_word_ner_line_list_sample_txt_file, output_article_word_ner_line_list_pickle_file))

        for input_summary_file_path, output_summary_word_ner_line_list_sample_txt_file, \
            output_summary_word_ner_line_list_pickle_file \
                in zip([input_summary1_file_path, input_summary2_file_path,
                        input_summary3_file_path, input_summary4_file_path],
                       [output_summary1_word_ner_line_list_sample_txt_file,
                        output_summary2_word_ner_line_list_sample_txt_file,
                        output_summary3_word_ner_line_list_sample_txt_file,
                        output_summary4_word_ner_line_list_sample_txt_file],
                       [output_summary1_word_ner_line_list_pickle_file, output_summary2_word_ner_line_list_pickle_file,
                        output_summary3_word_ner_line_list_pickle_file,
                        output_summary4_word_ner_line_list_pickle_file]):
            input_summary_batch_text_list = []
            with open(input_summary_file_path, 'r', encoding='utf8') as f:
                line_index = 0
                input_temp_list = []
                for line in f:
                    line_index += 1
                    input_temp_list += line.split() + ['NL_']
                    if line_index == lines_per_ner_application:
                        input_summary_batch_text_list.append(input_temp_list)
                        input_temp_list = []
                        line_index = 0
                if line_index > 0:
                    input_summary_batch_text_list.append(input_temp_list)
                f.close()
            print('Summary data have been loaded on batch list')

            t1 = time.time()
            line_index = 0
            summary_word_ner_per_line_list = []
            for batch_text_list in input_summary_batch_text_list:
                line_index += lines_per_ner_application
                word_ner_list = self.stanford_ner(batch_text_list)
                line_word_ner_list = []
                for (word, ner) in word_ner_list:
                    if word != 'NL_':
                        line_word_ner_list.append((word, ner))
                        if ner in ner_tags:
                            try:
                                new_freq = word_ner_freq_dict[(word, ner)] + 1
                                word_ner_freq_dict[(word, ner)] = new_freq
                            except KeyError:
                                word_ner_freq_dict[(word, ner)] = 1
                    else:
                        summary_word_ner_per_line_list.append(line_word_ner_list)
                        line_word_ner_list = []
                if line_index % print_per_line == 0:
                    t = time.time()
                    print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                        line_index, 1000, datetime.timedelta(seconds=t - t0),
                        (t - t1) * 1000 / print_per_line))
                    t1 = t
            print('Summary ner tags have been obtained.')
            del input_summary_batch_text_list
            with open(output_summary_word_ner_line_list_sample_txt_file, 'w', encoding='utf8') as f:
                index = 0
                for word_pos_line_list in summary_word_ner_per_line_list:
                    f.write('{}\n'.format(word_pos_line_list))
                    index += 1
                    if index == lines_of_sample_files:
                        break
            with open(output_summary_word_ner_line_list_pickle_file, 'wb') as f:
                pickle.dump(summary_word_ner_per_line_list, f)
            print('Summary files have been created:\n\t{}\n\t{}'.format(
                output_summary_word_ner_line_list_sample_txt_file, output_summary_word_ner_line_list_pickle_file))
            del summary_word_ner_per_line_list

        word_pos_freq_list = []
        for k, v in word_ner_freq_dict.items():
            word_pos_freq_list.append((k, v))
        word_pos_freq_list = sorted(word_pos_freq_list, key=lambda tup: -tup[1])
        with open(output_word_ner_freq_dict_txt_file, 'w', encoding='utf8') as f:
            for ((w, p), freq) in word_pos_freq_list:
                f.write('{} {} {}\n'.format(w, p, freq))
        with open(output_word_ner_freq_dict_pickle_file, 'wb') as f:
            pickle.dump(word_ner_freq_dict, f)
        print('Word-pos-freq files have been created:\n\t{}\n\t{}'.format(
            output_word_ner_freq_dict_txt_file, output_word_ner_freq_dict_pickle_file))

    def pos_tagging_of_duc_dataset_and_vocab_pos_frequent(self,
                                                          input_article_file_path,
                                                          input_summary1_file_path,
                                                          input_summary2_file_path,
                                                          input_summary3_file_path,
                                                          input_summary4_file_path,
                                                          output_article_word_pos_line_list_sample_txt_file,
                                                          output_summary1_word_pos_line_list_sample_txt_file,
                                                          output_summary2_word_pos_line_list_sample_txt_file,
                                                          output_summary3_word_pos_line_list_sample_txt_file,
                                                          output_summary4_word_pos_line_list_sample_txt_file,
                                                          output_article_word_pos_line_list_pickle_file,
                                                          output_summary1_word_pos_line_list_pickle_file,
                                                          output_summary2_word_pos_line_list_pickle_file,
                                                          output_summary3_word_pos_line_list_pickle_file,
                                                          output_summary4_word_pos_line_list_pickle_file,
                                                          output_word_pos_freq_dict_txt_file,
                                                          output_word_pos_freq_dict_pickle_file,
                                                          lines_per_pos_application, print_per_line=50,
                                                          lines_of_sample_files=1000):
        t0 = time.time()
        input_article_batch_text_list = []

        print('Reading files:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(input_article_file_path, input_summary1_file_path,
                                                                    input_summary2_file_path, input_summary3_file_path,
                                                                    input_summary4_file_path))
        with open(input_article_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_pos_application:
                    input_article_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_article_batch_text_list.append(input_temp_list)
            f.close()
        print('Article data have been loaded to  batch list')

        word_pos_freq_dict = dict()
        article_word_pos_per_line_list = []
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_article_batch_text_list:
            line_index += lines_per_pos_application
            word_pos_list = self.wordnet_pos_tag(batch_text_list)
            line_word_pos_list = []
            for (word, pos) in word_pos_list:
                if word != 'NL_':
                    line_word_pos_list.append((word, pos))
                    try:
                        new_freq = word_pos_freq_dict[(word, pos)] + 1
                        word_pos_freq_dict[(word, pos)] = new_freq
                    except KeyError:
                        word_pos_freq_dict[(word, pos)] = 1
                else:
                    article_word_pos_per_line_list.append(line_word_pos_list)
                    line_word_pos_list = []
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        del input_article_batch_text_list
        print('Article pos tags have been obtained.')

        with open(output_article_word_pos_line_list_sample_txt_file, 'w', encoding='utf8') as f:
            index = 0
            for word_pos_line_list in article_word_pos_per_line_list:
                f.write('{}\n'.format(word_pos_line_list))
                index += 1
                if index == lines_of_sample_files:
                    break
        with open(output_article_word_pos_line_list_pickle_file, 'wb') as f:
            pickle.dump(article_word_pos_per_line_list, f)
        del article_word_pos_per_line_list
        print('Article files have been created:\n\t{}\n\t{}'.format(
            output_article_word_pos_line_list_sample_txt_file, output_article_word_pos_line_list_pickle_file))

        for summary, output_summary_word_pos_line_list_sample_txt_file, output_summary_word_pos_line_list_pickle_file \
                in zip([input_summary1_file_path, input_summary2_file_path,
                        input_summary3_file_path, input_summary4_file_path],
                       [output_summary1_word_pos_line_list_sample_txt_file,
                        output_summary2_word_pos_line_list_sample_txt_file,
                        output_summary3_word_pos_line_list_sample_txt_file,
                        output_summary4_word_pos_line_list_sample_txt_file],
                       [output_summary1_word_pos_line_list_pickle_file, output_summary2_word_pos_line_list_pickle_file,
                        output_summary3_word_pos_line_list_pickle_file,
                        output_summary4_word_pos_line_list_pickle_file]):
            input_summary_batch_text_list = []
            with open(summary, 'r', encoding='utf8') as f:
                line_index = 0
                input_temp_list = []
                for line in f:
                    line_index += 1
                    input_temp_list += line.split() + ['NL_']
                    if line_index == lines_per_pos_application:
                        input_summary_batch_text_list.append(input_temp_list)
                        input_temp_list = []
                        line_index = 0
                if line_index > 0:
                    input_summary_batch_text_list.append(input_temp_list)
                f.close()
            print('Summary data have been loaded to  batch list')

            t1 = time.time()
            line_index = 0
            summary_word_pos_per_line_list = []
            for batch_text_list in input_summary_batch_text_list:
                line_index += lines_per_pos_application
                word_pos_list = self.wordnet_pos_tag(batch_text_list)
                line_word_pos_list = []
                for (word, pos) in word_pos_list:
                    if word != 'NL_':
                        line_word_pos_list.append((word, pos))
                        try:
                            new_freq = word_pos_freq_dict[(word, pos)] + 1
                            word_pos_freq_dict[(word, pos)] = new_freq
                        except KeyError:
                            word_pos_freq_dict[(word, pos)] = 1
                    else:
                        summary_word_pos_per_line_list.append(line_word_pos_list)
                        line_word_pos_list = []
                if line_index % print_per_line == 0:
                    t = time.time()
                    print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                        line_index, 1000, datetime.timedelta(seconds=t - t0),
                        (t - t1) * 1000 / print_per_line))
                    t1 = t
            print('Summary pos tags have been obtained.')
            del input_summary_batch_text_list
            with open(output_summary_word_pos_line_list_sample_txt_file, 'w', encoding='utf8') as f:
                index = 0
                for word_pos_line_list in summary_word_pos_per_line_list:
                    f.write('{}\n'.format(word_pos_line_list))
                    index += 1
                    if index == lines_of_sample_files:
                        break
            with open(output_summary_word_pos_line_list_pickle_file, 'wb') as f:
                pickle.dump(summary_word_pos_per_line_list, f)
            print('Summary files have been created:\n\t{}\n\t{}'.format(
                output_summary_word_pos_line_list_sample_txt_file, output_summary_word_pos_line_list_pickle_file))
            del summary_word_pos_per_line_list

        word_pos_freq_list = []
        for k, v in word_pos_freq_dict.items():
            word_pos_freq_list.append((k, v))
        word_pos_freq_list = sorted(word_pos_freq_list, key=lambda tup: -tup[1])
        with open(output_word_pos_freq_dict_txt_file, 'w', encoding='utf8') as f:
            for ((w, p), freq) in word_pos_freq_list:
                f.write('{} {} {}\n'.format(w, p, freq))
        with open(output_word_pos_freq_dict_pickle_file, 'wb') as f:
            pickle.dump(word_pos_freq_dict, f)
        print('Word-pos-freq files have been created:\n\t{}\n\t{}'.format(
            output_word_pos_freq_dict_txt_file, output_word_pos_freq_dict_pickle_file))

    def pos_tagging_of_dataset_and_vocabulary_of_words_pos_frequent(self,
                                                                    input_article_file_path,
                                                                    input_summary_file_path,
                                                                    output_article_word_pos_line_list_sample_txt_file,
                                                                    output_summary_word_pos_line_list_sample_txt_file,
                                                                    output_article_word_pos_line_list_pickle_file,
                                                                    output_summary_word_pos_line_list_pickle_file,
                                                                    output_word_pos_freq_dict_txt_file,
                                                                    output_word_pos_freq_dict_pickle_file,
                                                                    lines_per_pos_application, print_per_line=100,
                                                                    lines_of_sample_files=1000):

        t0 = time.time()
        input_article_batch_text_list = []
        input_summary_batch_text_list = []
        print('Reading files:\n\t{}\n\t{}'.format(input_article_file_path, input_summary_file_path))
        with open(input_article_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_pos_application:
                    input_article_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_article_batch_text_list.append(input_temp_list)
            f.close()
        print('Article data have been loaded to  batch list')

        word_pos_freq_dict = dict()
        article_word_pos_per_line_list = []
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_article_batch_text_list:
            line_index += lines_per_pos_application
            word_pos_list = self.wordnet_pos_tag(batch_text_list)
            line_word_pos_list = []
            for (word, pos) in word_pos_list:
                if word != 'NL_':
                    line_word_pos_list.append((word, pos))
                    try:
                        new_freq = word_pos_freq_dict[(word, pos)] + 1
                        word_pos_freq_dict[(word, pos)] = new_freq
                    except KeyError:
                        word_pos_freq_dict[(word, pos)] = 1
                else:
                    article_word_pos_per_line_list.append(line_word_pos_list)
                    line_word_pos_list = []
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        del input_article_batch_text_list
        print('Article pos tags have been obtained.')

        with open(output_article_word_pos_line_list_sample_txt_file, 'w', encoding='utf8') as f:
            index = 0
            for word_pos_line_list in article_word_pos_per_line_list:
                f.write('{}\n'.format(word_pos_line_list))
                index += 1
                if index == lines_of_sample_files:
                    break
        with open(output_article_word_pos_line_list_pickle_file, 'wb') as f:
            pickle.dump(article_word_pos_per_line_list, f)
        del article_word_pos_per_line_list
        print('Article files have been created:\n\t{}\n\t{}'.format(
            output_article_word_pos_line_list_sample_txt_file, output_article_word_pos_line_list_pickle_file))

        with open(input_summary_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_pos_application:
                    input_summary_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_summary_batch_text_list.append(input_temp_list)
            f.close()
        print('Summary data have been loaded to  batch list')

        t1 = time.time()
        line_index = 0
        summary_word_pos_per_line_list = []
        for batch_text_list in input_summary_batch_text_list:
            line_index += lines_per_pos_application
            word_pos_list = self.wordnet_pos_tag(batch_text_list)
            line_word_pos_list = []
            for (word, pos) in word_pos_list:
                if word != 'NL_':
                    line_word_pos_list.append((word, pos))
                    try:
                        new_freq = word_pos_freq_dict[(word, pos)] + 1
                        word_pos_freq_dict[(word, pos)] = new_freq
                    except KeyError:
                        word_pos_freq_dict[(word, pos)] = 1
                else:
                    summary_word_pos_per_line_list.append(line_word_pos_list)
                    line_word_pos_list = []
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        print('Summary pos tags have been obtained.')
        del input_summary_batch_text_list
        with open(output_summary_word_pos_line_list_sample_txt_file, 'w', encoding='utf8') as f:
            index = 0
            for word_pos_line_list in summary_word_pos_per_line_list:
                f.write('{}\n'.format(word_pos_line_list))
                index += 1
                if index == lines_of_sample_files:
                    break
        with open(output_summary_word_pos_line_list_pickle_file, 'wb') as f:
            pickle.dump(summary_word_pos_per_line_list, f)
        print('Summary files have been created:\n\t{}\n\t{}'.format(
            output_summary_word_pos_line_list_sample_txt_file, output_summary_word_pos_line_list_pickle_file))
        del summary_word_pos_per_line_list

        word_pos_freq_list = []
        for k, v in word_pos_freq_dict.items():
            word_pos_freq_list.append((k, v))
        word_pos_freq_list = sorted(word_pos_freq_list, key=lambda tup: -tup[1])
        with open(output_word_pos_freq_dict_txt_file, 'w', encoding='utf8') as f:
            for ((w, p), freq) in word_pos_freq_list:
                f.write('{} {} {}\n'.format(w, p, freq))
        with open(output_word_pos_freq_dict_pickle_file, 'wb') as f:
            pickle.dump(word_pos_freq_dict, f)
        print('Word-pos-freq files have been created:\n\t{}\n\t{}'.format(
            output_word_pos_freq_dict_txt_file, output_word_pos_freq_dict_pickle_file))

        # it returns an dictionayr of hyperonym paths of its word

    def convert_dataset_with_hyperonyms(self, input_article_file_path, input_summary_file_path,
                                        output_article_file_path, output_summary_file_path,
                                        output_hypernyms_dict_pickle_file_path, output_hypernyms_dict_txt_file_path,
                                        print_per_line=2, max_depth=6,
                                        lines_per_pos_application=2000):
        t0 = time.time()
        general_hypernyms = ['abstraction', 'entity', 'attribute', 'whole', 'physical',
                             'entity', 'physical_entity', 'matter', 'object', 'relation', 'natural_object',
                             'psychological_feature']
        stopword_list = nltk.corpus.stopwords.words('english')
        general_categories = []  # ['PERSON_', 'LOCATION_', 'ORGANIZATION_']
        except_words = stopword_list + general_categories
        word_set = set()
        article_pos_list = []
        summary_pos_list = []
        print('Building dataset with hypernyms...')
        print('Input files:\n\t{}\n\t{}'.format(input_article_file_path, input_summary_file_path))
        print('Building dictionary...')

        input_article_batch_text_list = []
        input_summary_batch_text_list = []
        with open(input_article_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_pos_application:
                    input_article_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_article_batch_text_list.append(input_temp_list)
            f.close()
        with open(input_summary_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_pos_application:
                    input_summary_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_summary_batch_text_list.append(input_temp_list)
            f.close()

        count_words = 0
        # with open(input_article_file_path, 'r', encoding='utf8') as f:
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_article_batch_text_list:
            line_index += lines_per_pos_application
            pos_list = self.wordnet_pos_n_tag(batch_text_list)
            article_pos_list.append(pos_list)
            for (word, pos) in pos_list:
                if pos == 'n' and word not in except_words and word != 'NL_':
                    word_set.add(word)
                    count_words += 1
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        del input_article_batch_text_list
        print('Article vocabulary has been loaded.')
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_summary_batch_text_list:
            line_index += lines_per_pos_application
            pos_list = self.wordnet_pos_n_tag(batch_text_list)
            summary_pos_list.append(pos_list)
            for (word, pos) in pos_list:
                if pos == 'n' and word not in except_words and word != 'NL_':
                    word_set.add(word)
                    count_words += 1
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t

        del input_summary_batch_text_list
        print('Summary vocabulary has been loaded.')
        print('Dictionary has been built (count_words: {}).'.format(count_words))
        print('Extracting hypernyms...')
        word_hypernym_path_dict = dict()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for token_ in word_set:
            token_lemma = lemmatizer.lemmatize(token_, pos='n')
            synset = self.make_synset(token_lemma, category='n')
            if synset is not None:
                synset.max_depth()
                merged_synset_list = self.merge_lists(synset.hypernym_paths())
                sorted_synsets = self.syncet_sort_accornding_max_depth(merged_synset_list)
                word_depth_list = self.word_depth_of_synsents(sorted_synsets)
                if word_depth_list[0][0] != token_lemma:
                    word_depth_list = [(token_lemma, word_depth_list[0][1] + 1)] + word_depth_list
                word_hypernym_path_dict[token_lemma] = word_depth_list
        del word_set
        ###############
        with open(output_hypernyms_dict_txt_file_path, 'w', encoding='utf8') as f:
            for k, v in word_hypernym_path_dict.items():
                f.write('{} {}\n'.format(k, v))
        with open(output_hypernyms_dict_pickle_file_path, 'wb') as f:
            pickle.dump(word_hypernym_path_dict, f)
        print('Hypernyms have been written to files:\n\t{}\n\t{}'.format(
            output_hypernyms_dict_pickle_file_path, output_hypernyms_dict_txt_file_path))

        hypernym_changed_words_list_dict = dict()

        print('writing article file...')
        t1 = time.time()
        # depth_greater_than = min_depth - 1
        with open(output_article_file_path, 'w', encoding='utf8') as f:
            line_index = 0
            for pos_list in article_pos_list:
                line_index += lines_per_pos_application
                for (word, pos) in pos_list:
                    if word == 'NL_':
                        f.write('\n')
                    elif pos == 'n' and word not in except_words and word != None:
                        try:
                            hypernyms_path_list = word_hypernym_path_dict[word]
                            # print(hypernyms_path_list)
                            # hypernym_token = hypernyms_path_list[hypernym_offset][0]
                            depth = hypernyms_path_list[0][1]
                            if depth > max_depth:
                                for el in hypernyms_path_list:
                                    if el[1] <= max_depth:
                                        # print(el[2])
                                        f.write(el[0] + '_ ')
                                        if hypernym_changed_words_list_dict.get(el[0], None):

                                            new_list = hypernym_changed_words_list_dict[el[0]][1] + [word]
                                            new_freq = hypernym_changed_words_list_dict[el[0]][0] + 1
                                            hypernym_changed_words_list_dict[el[0]] = (new_freq, new_list)
                                        else:
                                            hypernym_changed_words_list_dict[el[0]] = (1, [word])
                                        break
                            else:
                                f.write(word + ' ')
                        except KeyError:
                            f.write(word + ' ')
                        except IndexError:
                            f.write(word + ' ')
                    else:
                        f.write(word + ' ')
                # f.write('\n')
                if line_index % print_per_line == 0:
                    t = time.time()
                    print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                        line_index, 1000, datetime.timedelta(seconds=t - t0),
                        (t - t1) * 1000 / print_per_line))
                    t1 = t
        del article_pos_list
        print('writing summary file...')
        t1 = time.time()
        with open(output_summary_file_path, 'w', encoding='utf8') as f:
            line_index = 0
            for pos_list in summary_pos_list:
                line_index += lines_per_pos_application
                for (word, pos) in pos_list:
                    if word == 'NL_':
                        f.write('\n')
                    elif pos == 'n' and word not in except_words and word != None:
                        try:
                            hypernyms_path_list = word_hypernym_path_dict[word]
                            # hypernym_token = hypernyms_path_list[hypernym_offset][0]
                            depth = hypernyms_path_list[0][1]
                            if depth > max_depth:
                                for el in hypernyms_path_list:
                                    if el[1] <= max_depth:
                                        f.write(el[0] + '_ ')
                                        if hypernym_changed_words_list_dict.get(el[0], None):
                                            new_list = hypernym_changed_words_list_dict[el[0]][1] + [word]
                                            new_freq = hypernym_changed_words_list_dict[el[0]][0] + 1
                                            hypernym_changed_words_list_dict[el[0]] = (new_freq, new_list)
                                        else:
                                            hypernym_changed_words_list_dict[el[0]] = (1, [word])
                                        break
                            else:
                                f.write(word + ' ')
                        except KeyError:
                            f.write(word + ' ')
                        except IndexError:
                            f.write(word + ' ')
                    else:
                        f.write(word + ' ')
                # f.write('\n')
                if line_index % print_per_line == 0:
                    t = time.time()
                    print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                        line_index, 1000, datetime.timedelta(seconds=t - t0),
                        (t - t1) * 1000 / print_per_line))
                    t1 = t

        print('Hypernyms and the words that they have replaced:')
        for k, v in hypernym_changed_words_list_dict.items():
            print(k, v[0], v[1])
        print('Output files:\n\t{}\n\t{}'.format(output_article_file_path, output_summary_file_path))
        # return word_hypernym_path_dict

        # it returns an dictionayr of hyperonym paths of its word

    def convert_dataset_with_ner_and_hyperonyms(self, input_article_file_path, input_summary_file_path,
                                                output_article_file_path, output_summary_file_path,
                                                output_hypernyms_dict_pickle_file_path,
                                                output_hypernyms_dict_txt_file_path,
                                                print_per_line=2, hypernym_offset=1, min_depth=5, max_depth=6,
                                                lines_per_pos_application=2000):
        t0 = time.time()
        general_hypernyms = ['abstraction', 'entity', 'attribute', 'whole', 'physical',
                             'entity', 'physical_entity', 'matter', 'object', 'relation', 'natural_object',
                             'psychological_feature']
        stopword_list = nltk.corpus.stopwords.words('english')
        general_categories = ['PERSON_', 'LOCATION_', 'ORGANIZATION_']
        except_words = stopword_list + general_categories
        word_set = set()
        article_pos_list = []
        summary_pos_list = []
        print('Building dataset with hypernyms...')
        print('Input files:\n\t{}\n\t{}'.format(input_article_file_path, input_summary_file_path))
        print('Building dictionary...')

        input_article_batch_text_list = []
        input_summary_batch_text_list = []
        with open(input_article_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_pos_application:
                    input_article_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_article_batch_text_list.append(input_temp_list)
            f.close()
        with open(input_summary_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_pos_application:
                    input_summary_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_summary_batch_text_list.append(input_temp_list)
            f.close()

        count_words = 0
        # with open(input_article_file_path, 'r', encoding='utf8') as f:
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_article_batch_text_list:
            line_index += lines_per_pos_application
            pos_list = self.wordnet_pos_n_tag(batch_text_list)
            article_pos_list.append(pos_list)
            for (word, pos) in pos_list:
                if pos == 'n' and word not in except_words and word != 'NL_':
                    word_set.add(word)
                    count_words += 1
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t
        del input_article_batch_text_list
        print('Article vocabulary has been loaded.')
        t1 = time.time()
        line_index = 0
        for batch_text_list in input_summary_batch_text_list:
            line_index += lines_per_pos_application
            pos_list = self.wordnet_pos_n_tag(batch_text_list)
            summary_pos_list.append(pos_list)
            for (word, pos) in pos_list:
                if pos == 'n' and word not in except_words and word != 'NL_':
                    word_set.add(word)
                    count_words += 1
            if line_index % print_per_line == 0:
                t = time.time()
                print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                    line_index, 1000, datetime.timedelta(seconds=t - t0),
                    (t - t1) * 1000 / print_per_line))
                t1 = t

        del input_summary_batch_text_list
        print('Summary vocabulary has been loaded.')
        print('Dictionary has been built (count_words: {}).'.format(count_words))
        print('Extracting hypernyms...')
        word_hypernym_path_dict = dict()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for token_ in word_set:
            token_lemma = lemmatizer.lemmatize(token_, pos='n')
            synset = self.make_synset(token_lemma, category='n')
            if synset is not None:
                synset.max_depth()
                merged_synset_list = self.merge_lists(synset.hypernym_paths())
                sorted_synsets = self.syncet_sort_accornding_max_depth(merged_synset_list)
                word_depth_list = self.word_depth_of_synsents(sorted_synsets)
                if word_depth_list[0][0] != token_lemma:
                    word_depth_list = [(token_lemma, word_depth_list[0][1] + 1)] + word_depth_list
                word_hypernym_path_dict[token_lemma] = word_depth_list
        del word_set
        ###############
        with open(output_hypernyms_dict_txt_file_path, 'w', encoding='utf8') as f:
            for k, v in word_hypernym_path_dict.items():
                f.write('{} {}\n'.format(k, v))
        with open(output_hypernyms_dict_pickle_file_path, 'wb') as f:
            pickle.dump(word_hypernym_path_dict, f)
        print('Hypernyms have been written to files:\n\t{}\n\t{}'.format(
            output_hypernyms_dict_pickle_file_path, output_hypernyms_dict_txt_file_path))

        hypernym_changed_words_list_dict = dict()

        print('writing article file...')
        t1 = time.time()
        # depth_greater_than = min_depth - 1
        with open(output_article_file_path, 'w', encoding='utf8') as f:
            line_index = 0
            for pos_list in article_pos_list:
                line_index += lines_per_pos_application
                for (word, pos) in pos_list:
                    if word == 'NL_':
                        f.write('\n')
                    elif pos == 'n' and word not in except_words and word != None:
                        try:
                            hypernyms_path_list = word_hypernym_path_dict[word]
                            # print(hypernyms_path_list)
                            # hypernym_token = hypernyms_path_list[hypernym_offset][0]
                            depth = hypernyms_path_list[0][1]
                            if depth > max_depth:
                                for el in hypernyms_path_list:
                                    if el[1] <= max_depth:
                                        # print(el[2])
                                        f.write(el[0] + '_ ')
                                        if hypernym_changed_words_list_dict.get(el[0], None):

                                            new_list = hypernym_changed_words_list_dict[el[0]][1] + [word]
                                            new_freq = hypernym_changed_words_list_dict[el[0]][0] + 1
                                            hypernym_changed_words_list_dict[el[0]] = (new_freq, new_list)
                                        else:
                                            hypernym_changed_words_list_dict[el[0]] = (1, [word])
                                        break
                            else:
                                f.write(word + ' ')
                        except KeyError:
                            f.write(word + ' ')
                        except IndexError:
                            f.write(word + ' ')
                    else:
                        f.write(word + ' ')
                # f.write('\n')
                if line_index % print_per_line == 0:
                    t = time.time()
                    print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                        line_index, 1000, datetime.timedelta(seconds=t - t0),
                        (t - t1) * 1000 / print_per_line))
                    t1 = t
        del article_pos_list
        print('writing summary file...')
        t1 = time.time()
        with open(output_summary_file_path, 'w', encoding='utf8') as f:
            line_index = 0
            for pos_list in summary_pos_list:
                line_index += lines_per_pos_application
                for (word, pos) in pos_list:
                    if word == 'NL_':
                        f.write('\n')
                    elif pos == 'n' and word not in except_words and word != None:
                        try:
                            hypernyms_path_list = word_hypernym_path_dict[word]
                            # hypernym_token = hypernyms_path_list[hypernym_offset][0]
                            depth = hypernyms_path_list[0][1]
                            if depth > max_depth:
                                for el in hypernyms_path_list:
                                    if el[1] <= max_depth:
                                        f.write(el[0] + '_ ')
                                        if hypernym_changed_words_list_dict.get(el[0], None):
                                            new_list = hypernym_changed_words_list_dict[el[0]][1] + [word]
                                            new_freq = hypernym_changed_words_list_dict[el[0]][0] + 1
                                            hypernym_changed_words_list_dict[el[0]] = (new_freq, new_list)
                                        else:
                                            hypernym_changed_words_list_dict[el[0]] = (1, [word])
                                        break
                            else:
                                f.write(word + ' ')
                        except KeyError:
                            f.write(word + ' ')
                        except IndexError:
                            f.write(word + ' ')
                    else:
                        f.write(word + ' ')
                # f.write('\n')
                if line_index % print_per_line == 0:
                    t = time.time()
                    print('{} line, Time (overall and per {} lines): {} & {:.2f}'.format(
                        line_index, 1000, datetime.timedelta(seconds=t - t0),
                        (t - t1) * 1000 / print_per_line))
                    t1 = t

        print('Hypernyms and the words that they have replaced:')
        for k, v in hypernym_changed_words_list_dict.items():
            print(k, v[0], v[1])
        print('Output files:\n\t{}\n\t{}'.format(output_article_file_path, output_summary_file_path))
        # return word_hypernym_path_dict

    @staticmethod
    # it returns a list of tuples: (word, pos)
    def wordnet_pos_n_tag(text_list):
        # wordnet pos: (ADJ, ADJ_SAT, ADV, NOUN, VERB) = ('a', 's', 'r', 'n', 'v')
        pos_tag_list = nltk.tag.pos_tag(text_list)
        word_pos_list = []
        for (w, pos) in pos_tag_list:
            wordnet_pos = 'other'
            if pos.startswith('N'):
                wordnet_pos = 'n'  # nltk.corpus.wordnet.NOUN
            word_pos_list.append((w, wordnet_pos))
        return word_pos_list

    def min_common_hyperonym_of_vocabulary(self, input_file_path='path/to/file',
                                           output_dictionary_pickle_file="", output_dict_txt_file="",
                                           time_of_pass=1):
        stopword_list = nltk.corpus.stopwords.words('english')
        general_categories = ['PERSON_', 'LOCATION_', 'ORGANIZATION_']
        general_hypernyms = ['abstraction', 'entity', 'attribute', 'whole', 'physical',
                             'entity', 'physical_entity', 'matter', 'object', 'relation', 'natural_object']
        except_words = stopword_list + general_categories
        # print(except_words)

        word_pos_freq_dict = dict()

        print('Building dictionary')
        count_words = 0
        with open(input_file_path, 'r', encoding='utf8') as f:
            for line in f:
                line_list = line.split()
                pos_list, _ = self.wordnet_pos_tag(line_list)
                for word, pos in pos_list:
                    if pos == 'n' and word not in except_words:
                        if word_pos_freq_dict.get((word, pos), None):
                            new_freq = word_pos_freq_dict[(word, pos)] + 1
                            word_pos_freq_dict[(word, pos)] = new_freq
                        else:
                            word_pos_freq_dict[(word, pos)] = 1
                        count_words += 1
        print('dictionary is built. count_words: {}'.format(count_words))

        word_pos_list = []
        for k, v in word_pos_freq_dict.items():
            word_pos_list.append((k, v))
            ##################
            # print(k, v)
        word_pos_list = sorted(word_pos_list, key=lambda tup: -tup[1])

        #############
        # print(word_pos_list)

        del word_pos_freq_dict
        word_hypernym_dict = dict()
        # word2_start_index = 0
        for ((word1, pos1), freq1) in word_pos_list:
            # word2_start_index += 1
            hypernym_freq_dict = dict()
            try:
                synset1 = self.make_synset(word1)

                for ((word2, pos2), freq2) in word_pos_list:

                    synset2 = self.make_synset(word2)
                    # try:
                    common_hypernyms = synset1.lowest_common_hypernyms(synset2)
                    # except Exception:
                    if common_hypernyms != []:
                        for ch in common_hypernyms:
                            ch_word = self.synset_word(ch)
                            if ch_word not in general_hypernyms:
                                if hypernym_freq_dict.get(ch_word, None):
                                    new_freq = hypernym_freq_dict[ch_word] + freq2
                                    hypernym_freq_dict[ch_word] = new_freq
                                else:
                                    hypernym_freq_dict[ch_word] = freq2
                    else:
                        word_hypernym_dict[word1] = word1
            except nltk.corpus.reader.wordnet.WordNetError:
                word_hypernym_dict[word1] = word1
            max_freq = 0
            for k, v in hypernym_freq_dict.items():
                ############
                # print(k, v)
                if v > max_freq:
                    word_hypernym_dict[word1] = k

        ################
        for k, v in word_hypernym_dict.items():
            if k != v:
                print(k, v)

    def convert_dataset_with_hypernyms(self, input_article_file_path, output_article_file_path,
                                       input_summary_file_path, output_summary_file_path,
                                       lines_per_ner_application=2500, print_per_lines=10000):
        for (input_file_path, output_file_path) in [(input_article_file_path, output_article_file_path),
                                                    (input_summary_file_path, output_summary_file_path)]:
            self.convert_text_with_hypernyms(input_file_path, output_file_path,
                                             lines_per_ner_application=lines_per_ner_application,
                                             print_per_lines=print_per_lines)

    def convert_dataset_with_ner(self, input_article_file_path, input_summary_file_path,
                                 output_article_file_path, output_summary_file_path,
                                 lines_per_ner_application=2500, print_per_lines=10000):
        for (input_file_path, output_file_path) in [(input_article_file_path, output_article_file_path),
                                                    (input_summary_file_path, output_summary_file_path)]:
            self.convert_text_with_ner(input_file_path, output_file_path,
                                       lines_per_ner_application=lines_per_ner_application,
                                       print_per_lines=print_per_lines)

    def convert_text_with_hypernyms(self, input_file_path, output_file_path,
                                    lines_per_ner_application=2500, print_per_lines=10000):
        print('Named Entity Recognition and convert the dataset')
        print('Input file: {}\n'.format(input_file_path))
        input_batch_text_list = []
        with open(input_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_ner_application:
                    input_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_batch_text_list.append(input_temp_list)
            f.close()
        print('Input data loaded.')
        ner_list = []
        lines_index = 0
        t0 = time.time()
        for el_list in input_batch_text_list:
            ner_list += self.stanford_ner(el_list)
            lines_index += lines_per_ner_application
            if lines_index % print_per_lines == 0:
                dt = time.time() - t0
                print('NER: {} lines, Time (total and avg per 1000 lines) {} & {:.3f} sec,'.format(
                    lines_index, datetime.timedelta(seconds=dt), dt * 1000 / lines_index))
        del input_batch_text_list
        print('NER have been run')
        ner_freq_dict = dict()
        ner_tag_list = ['LOCATION', 'PERSON', 'ORGANIZATION']
        output_file = open(output_file_path, 'w', encoding='utf8')
        previous_text = ''
        for (token, ner) in ner_list:
            # temp_text = token
            if token == 'NL_':
                output_file.write('\n')
                previous_text = ''
            elif ner in ner_tag_list:
                if ner != previous_text:
                    output_file.write(ner + '_ ')
                    previous_text = ner
                    if ner_freq_dict.get(ner, None):
                        new_freq = ner_freq_dict[ner] + 1
                        ner_freq_dict[ner] = new_freq
                    else:
                        ner_freq_dict[ner] = 1
            else:
                output_file.write(token + ' ')
                previous_text = token
        output_file.close()
        del ner_list
        for k, v in ner_freq_dict.items():
            print(k, v)
        print('Output file: {}\n'.format(output_file_path))

    def convert_text_with_ner(self, input_file_path, output_file_path,
                              lines_per_ner_application=2500, print_per_lines=10000):
        print('Named Entity Recognition and convert the dataset')
        print('Input file: {}\n'.format(input_file_path))
        input_batch_text_list = []
        with open(input_file_path, 'r', encoding='utf8') as f:
            line_index = 0
            input_temp_list = []
            for line in f:
                line_index += 1
                input_temp_list += line.split() + ['NL_']
                if line_index == lines_per_ner_application:
                    input_batch_text_list.append(input_temp_list)
                    input_temp_list = []
                    line_index = 0
            if line_index > 0:
                input_batch_text_list.append(input_temp_list)
            f.close()
        print('Input data loaded.')
        ner_list = []
        lines_index = 0
        t0 = time.time()
        stanford_ner_tagger_dir = "C:/Stanford_NLP_Tools/stanford-ner-2018-10-16/"
        model = ['english.all.3class.distsim.crf.ser.gz', 'english.all.3class.distsim.crf.ser.gz']
        ner = nltk.tag.stanford.StanfordNERTagger(
            stanford_ner_tagger_dir + 'classifiers/' + model[0],
            stanford_ner_tagger_dir + 'stanford-ner-3.9.2.jar')
        for el_list in input_batch_text_list:
            ner_list += ner.tag(el_list)  # self.stanford_ner(el_list)
            lines_index += lines_per_ner_application
            if lines_index % print_per_lines == 0:
                dt = time.time() - t0
                print('NER: {} lines, Time (total and avg per 1000 lines) {} & {:.3f} sec,'.format(
                    lines_index, datetime.timedelta(seconds=dt), dt * 1000 / lines_index))
        del input_batch_text_list
        print('NER have been run')
        ner_freq_dict = dict()
        ner_tag_list = ['LOCATION', 'PERSON', 'ORGANIZATION']
        output_file = open(output_file_path, 'w', encoding='utf8')
        previous_text = ''
        for (token, ner) in ner_list:
            # temp_text = token
            if token == 'NL_':
                output_file.write('\n')
                previous_text = ''
            elif ner in ner_tag_list:
                if ner != previous_text:
                    output_file.write(ner + '_ ')
                    previous_text = ner
                    if ner_freq_dict.get(ner, None):
                        new_freq = ner_freq_dict[ner] + 1
                        ner_freq_dict[ner] = new_freq
                    else:
                        ner_freq_dict[ner] = 1
            else:
                output_file.write(token + ' ')
                previous_text = token
        output_file.close()
        del ner_list
        for k, v in ner_freq_dict.items():
            print(k, v)
        print('Output file: {}\n'.format(output_file_path))

    def stanford_ner(self, text_list):
        # http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford
        # https://nlp.stanford.edu/software/CRF-NER.html#Starting
        java_path = 'C:\Program Files (x86)\Java\jre1.8.0_201/bin/java.exe'
        os.environ['JAVAHOME'] = java_path
        stanford_ner_tagger_dir = "C:/Stanford_NLP_Tools/stanford-ner-2018-10-16/"
        model = ['english.all.3class.distsim.crf.ser.gz', 'english.conll.4class.distsim.crf.ser.gz',
                 'english.all.3class.distsim.crf.ser.gz']
        ner = nltk.tag.stanford.StanfordNERTagger(
            stanford_ner_tagger_dir + 'classifiers/' + model[0],
            stanford_ner_tagger_dir + 'stanford-ner-3.9.2.jar')
        return ner.tag(text_list)

    @staticmethod
    # it returns a list of tuples: (word, pos)
    def wordnet_pos_tag(text_list):
        # wordnet pos: (ADJ, ADJ_SAT, ADV, NOUN, VERB) = ('a', 's', 'r', 'n', 'v')
        pos_tag_list = nltk.tag.pos_tag(text_list)
        # print(pos_tag_list)
        word_pos_list = []
        # size = 0
        for (w, pos) in pos_tag_list:
            # size += 1
            wordnet_pos = pos
            if pos.startswith('J'):
                wordnet_pos = 'a'  # nltk.corpus.wordnet.ADJ
            elif pos.startswith('V'):
                wordnet_pos = 'v'  # nltk.corpus.wordnet.VERB
            elif pos.startswith('N'):
                wordnet_pos = 'n'  # nltk.corpus.wordnet.NOUN
            elif pos.startswith('R'):
                wordnet_pos = 'r'  # nltk.corpus.wordnet.ADV
            word_pos_list.append((w, wordnet_pos))
            # lemma = lemmatizer.lemmatize(w, pos=wordnet_pos)
            # new_text += lemma + ' '
        # print(word_pos_list)
        return word_pos_list

    def nltk_pos_of_sentence(self, sentence):
        return nltk.tag.pos_tag(sentence.split())

    def stanford_pos(self, text_list):
        # http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford
        # https://nlp.stanford.edu/software/tagger.shtml
        # java_path = 'C:\Program Files (x86)\Java\jre1.8.0_191/bin/java.exe'
        # os.environ['JAVAHOME'] = java_path
        stanford_pos_tagger_dir = "C:/Stanford_NLP_Tools/stanford-postagger-full-2018-10-16/"
        pos = nltk.tag.stanford.StanfordPOSTagger(
            stanford_pos_tagger_dir + 'models/english-bidirectional-distsim.tagger',
            stanford_pos_tagger_dir + 'stanford-postagger-3.9.2.jar')
        return pos.tag(text_list)

    def hyperonyms_paths(self, sentence, stopword_list):
        # gensim_model = gensim.models.KeyedVectors.load_word2vec_format(param.word2vec_file_path, binary=False)

        ############
        print(sentence)
        word_hyperonym_list = []
        token_list = sentence.split()
        word_pos_list, size = self.wordnet_pos_tag(token_list)

        for (token, pos) in word_pos_list:
            if token in stopword_list or pos == 'other_pos':
                word_hyperonym_list.append([token])
            else:
                synset = self.make_synset(token, category=pos)
                synset.max_depth()
                merged_synset_list = self.merge_lists(synset.hypernym_paths())
                sorted_synsets = self.syncet_sort_accornding_max_depth(merged_synset_list)
                word_depth_list = self.word_depth_of_synsents(sorted_synsets)
                print(word_depth_list)
                # hypernyms = self.all_hypernyms(synset)
                # print(hypernyms)

    def word_depth_of_synsents(self, synset_depth_list):
        word_depth_list = []
        for (s, d) in synset_depth_list:
            word_depth_list.append((self.synset_word(s), d))
        return word_depth_list

    def merge_lists(self, list_of_list):
        merged_set = set()
        for el in list_of_list:
            for e in el:
                merged_set.add(e)
        return list(merged_set)

    def syncet_sort_accornding_max_depth(self, synsets_list):
        sorted_synset_list = []
        for synset in synsets_list:
            sorted_synset_list.append((synset, synset.max_depth()))
        return sorted(sorted_synset_list, key=lambda tup: -tup[1])

    def hypernym_path(self, synset):
        return synset.hypernym_paths()

    @staticmethod
    def make_synset(word, category='n', number='01'):
        """Make a synset"""
        try:
            return wordnet.synset('{}.{}.{}'.format(word, category, number))
        except nltk.corpus.reader.wordnet.WordNetError:
            return None

    def _recurse_all_hypernyms(self, synset, all_hypernyms):
        synset_hypernyms = synset.hypernyms()
        if synset_hypernyms:
            all_hypernyms += synset_hypernyms
            for hypernym in synset_hypernyms:
                self._recurse_all_hypernyms(hypernym, all_hypernyms)

    def all_hypernyms(self, synset):
        """Get the set of hypernyms of the hypernym of the synset etc.
           Nouns can have multiple hypernyms, so we can't just create a depth-sorted
           list."""
        hypernyms = []
        self._recurse_all_hypernyms(synset, hypernyms)
        return set(hypernyms)

    def depth_of_synset(self, synset):
        return

    def _recurse_leaf_hyponyms(self, synset, leaf_hyponyms):
        synset_hyponyms = synset.hyponyms()
        if synset_hyponyms:
            for hyponym in synset_hyponyms:
                self._recurse_all_hyponyms(hyponym, leaf_hyponyms)
        else:
            leaf_hyponyms += synset

    def leaf_hyponyms(self, synset):
        """Get the set of leaf nodes from the tree of hyponyms under the synset"""
        hyponyms = []
        self._recurse_leaf_hyponyms(synset, hyponyms)
        return set(hyponyms)

    def all_peers(self, synset):
        """Get the set of all peers of the synset (including the synset).
           If the synset has multiple hypernyms then the peers will be hyponyms of
           multiple synsets."""
        hypernyms = synset.hypernyms()
        peers = []
        for hypernym in hypernyms:
            peers += hypernym.hyponyms()
        return set(peers)

    def synset_synonyms(self, synset):
        """Get the synonyms for the synset"""
        return set([lemma.synset for lemma in synset.lemmas])

    def synset_antonyms(self, synset):
        """Get the antonyms for [the first lemma of] the synset"""
        return set([lemma.synset for lemma in synset.lemmas[0].antonyms()])

    def _recurse_all_hyponyms(self, synset, all_hyponyms):
        synset_hyponyms = synset.hyponyms()
        if synset_hyponyms:
            all_hyponyms += synset_hyponyms
            for hyponym in synset_hyponyms:
                self._recurse_all_hyponyms(hyponym, all_hyponyms)

    def all_hyponyms(self, synset):
        """Get the set of the tree of hyponyms under the synset"""
        hyponyms = []
        self._recurse_all_hyponyms(synset, hyponyms)
        return set(hyponyms)

    def synsets_words(self, synsets):
        """Get the set of strings for the words represented by the synsets"""
        return list([self.synset_word(synset) for synset in synsets])

    def synset_word(self, synset):
        name = synset.name()
        return name.split(sep='.')[0]

    # Compute the Word Mover’s Distance between two sentences.
    def sentence_similarity(self, sentence_1, sentence_2, gensim_model):
        distance = gensim_model.wmdistance(sentence_1, sentence_2)
        # print('distance = %.4f' % distance)
        return distance


if __name__ == "__main__":
    DataPreprocessing()
