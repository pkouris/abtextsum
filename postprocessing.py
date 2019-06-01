from nltk.corpus import wordnet
import numpy as np
# from nltk import pos_tag
import nltk
import time
import datetime
import pickle
import parameters as param
import os
import gensim
import sklearn.metrics.pairwise
import sklearn
import utils
import testing
from scipy.stats import wasserstein_distance
import dataset_path
from ortools.graph import pywrapgraph
# sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer


# from stanfordcorenlp import StanfordCoreNLP
# from nltk.tag.stanford import NERTagger

# It converst general summary to specific summary, giving the specific article and general summary
class PostProcessing:

    def __init__(self):
        start_time = time.time()
        mode = 'lg'

        if mode == 'neg':
            similarity_methods = {0: 'freq', 1: 'tfidf', 2: 'w2v', 3: 'edit', 4: 'jacard', 5: 'avg_w2v', 6: 'mix'}
            center_word = {0: False, 1: True}
            self.neg_postprocessing(input_general_summary_file_path='path/to/generalized_system_summaries',
                                    input_general_article_file_path='path/to/generalized_articles',
                                    input_specific_article_file_path='path/to/articles',
                                    # input_golden_summary_file_path='docs/golden_sum',
                                    output_summary_file_path='path/to/output_summaries',
                                    word_embedding_file_path='path/to/word embeddings',
                                    sum_window=3, art_window=5,
                                    similarity_method=similarity_methods[6],
                                    art_center_word=center_word[0],
                                    sum_center_word=center_word[0],
                                    # testing_mode='rouge_of_individual_files'
                                    )
        elif mode == 'lg':
            similarity_methods = {0: 'freq', 1: 'tfidf', 2: 'w2v', 3: 'edit', 4: 'jacard', 5: 'avg_w2v', 6: 'mix'}
            center_word = {0: False, 1: True}
            self.lg_postprocessing(input_general_summary_file_path='path/to/generalized_system_summaries',
                                   input_general_article_file_path='path/to/generalized_articles',
                                   input_specific_article_file_path='path/to/articles',
                                   # input_golden_summary_file_path='path/to/golden_sum',
                                   output_summary_file_path='path/to/output_summaries',
                                   word_embedding_file_path='path/to/word embeddings',
                                   hypernyms_dict_file_path='path/to/hypernym_paths_pickle_file',
                                   sum_window=3, art_window=5,
                                   similarity_method=similarity_methods[6],
                                   art_center_word=center_word[0],
                                   sum_center_word=center_word[0],
                                   # testing_mode='rouge_of_individual_files'
                                   )
        dt = time.time() - start_time
        print('Time: {}sec'.format(dt))
        print('Time: {}\n\nprocess finished'.format(datetime.timedelta(seconds=dt)))

    def neg_postprocessing(self, input_general_summary_file_path,
                           input_general_article_file_path,
                           input_specific_article_file_path,
                           # input_golden_summary_file_path,
                           output_summary_file_path,
                           word_embedding_file_path,
                           sum_window, art_window,
                           similarity_method='w2v',
                           art_center_word=False, sum_center_word=False,
                           # testing_mode='rouge_of_individual_files'
                           ):
        t0 = time.time()
        general_summary_per_line_list = []
        general_article_per_line_list = []
        specific_article_per_line_list = []
        # specific_text_line_list = []
        # specific_text_list = []
        output_summary_text = ''
        general_summary_text = ''
        # specific_summary_text = ''
        word2vec = None
        if similarity_method == 'w2v' or similarity_method == 'avg_w2v' or similarity_method == 'mix':
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_file_path, binary=False)
        stopword_list = nltk.corpus.stopwords.words('english')
        stemmer = nltk.stem.snowball.EnglishStemmer()
        # stemmer = nltk.stem.LancasterStemmer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer()
        with open(input_general_summary_file_path, 'r', encoding='utf8') as f:
            for line in f:
                general_summary_per_line_list.append(line.split())
                general_summary_text += line
        with open(input_general_article_file_path, 'r', encoding='utf8') as f:
            for line in f:
                general_article_per_line_list.append(line.split())
        with open(input_specific_article_file_path, 'r', encoding='utf8') as f:
            for line in f:
                specific_article_per_line_list.append(line.split())

        # general_summary_text += line
        print('Files have been read')

        # article_word_name_entity_list = self.word_ner_list(
        #    input_specific_article_file_path=input_specific_article_file_path,
        #    stopword_list=stopword_list, lemmatizer=lemmatizer,
        #    print_per_line=100000)
        stanford_ner_tags = ['PERSON_', 'LOCATION_', 'ORGANIZATION_']
        wordnet_ner_tags = ['person_', 'location_', 'organization_']
        ner_tags = stanford_ner_tags + wordnet_ner_tags
        # count_gen_words = 0
        output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
        line_index = 0
        for summary_line_word_list, general_article_line_word_list, specific_article_line_word_list in \
                zip(general_summary_per_line_list, general_article_per_line_list, specific_article_per_line_list):
            line_index += 1
            print_flag = False
            if line_index < 10:
                print_flag = True

            summary_line_word_list_ = [w for w in summary_line_word_list]

            generalized_consepts = []
            candidates_for_replacement = []
            sum_index = -1
            for token_s in summary_line_word_list:
                sum_index += 1
                if token_s in ner_tags:
                    generalized_consepts.append(sum_index)
                    art_index = -1
                    for token_a in general_article_line_word_list:
                        art_index += 1
                        if token_a == token_s:
                            similarity = self.similarity_of_text_v2(word2vec, lemmatizer, stemmer,
                                                                    art_center_word, sum_center_word,
                                                                    summary_line_word_list_,
                                                                    specific_article_line_word_list,
                                                                    sum_window, art_window,
                                                                    art_index, sum_index,
                                                                    stopword_list,
                                                                    similarity_method=similarity_method,
                                                                    decay_factor=0.2)
                            candidates_for_replacement.append((sum_index, art_index, similarity))
            candidates_for_replacement = utils.sort_by_third(candidates_for_replacement, descending=True)
            for (sum_index, art_index, similarity) in candidates_for_replacement:
                if sum_index in generalized_consepts:
                    summary_line_word_list_[sum_index] = specific_article_line_word_list[art_index]
                    generalized_consepts.remove(sum_index)

                    #############
            if print_flag:
                print(general_article_line_word_list)
                print(specific_article_line_word_list)
                print(summary_line_word_list)
                print(summary_line_word_list_)
                # print(' '.join([w.replace('_', ' ') for w in summary_line_word_list_]))
            output_summary_line_text = ''
            for w in summary_line_word_list_:
                if w not in ner_tags:
                    output_summary_line_text += w.replace('_', ' ') + ' '
                else:
                    output_summary_line_text += w + ' '
            if print_flag:
                print(output_summary_line_text)
                print()
            output_summary_file.write(output_summary_line_text.strip() + '\n')
            output_summary_text += output_summary_line_text + '\n'

        output_summary_file.close()
        print('Output file have been written.')
        # t1 = time.time()
        # with open(input_golden_summary_file_path, 'r', encoding='utf8') as f:
        #    specific_summary_text = f.read()
        # cos_sim_of_initial_file = self.cos_similarity_based_on_tfidf(specific_summary_text, general_summary_text)[0][0]
        # cos_sim_of_spec_and_output_file = \
        #    self.cos_similarity_based_on_tfidf(specific_summary_text, output_summary_text)[0][0]
        # cos_sim_of_output_output_file = \
        #    self.cos_similarity_based_on_tfidf(output_summary_text, output_summary_text)[0][0]
        # testing.Testing(testing_mode=testing_mode)
        # print('\ncosine similarities:\n')
        # print('\tcos_sim between golden and general summary: ', cos_sim_of_initial_file)
        # print('\tcos_sim between golde and output summary:   ', cos_sim_of_spec_and_output_file)
        # print('\tcos_sim between the same files (checking):  ', cos_sim_of_output_output_file)
        # t2 = time.time()
        # print('Time for converting, testing & overall: {}, {} & {}'.format(t1 - t0, t2 - t1, t2 - t0))
        print('Time: {}'.format(time.time() - t0))

    def lg_postprocessing(self, input_general_summary_file_path,
                          input_general_article_file_path,
                          input_specific_article_file_path,
                          # input_golden_summary_file_path,
                          output_summary_file_path,
                          word_embedding_file_path,
                          hypernyms_dict_file_path,
                          sum_window, art_window,
                          similarity_method='w2v',
                          art_center_word=False, sum_center_word=False,
                          # testing_mode='rouge_of_individual_files'
                          ):
        t0 = time.time()
        general_summary_per_line_list = []
        general_article_per_line_list = []
        specific_article_per_line_list = []
        # specific_text_line_list = []
        # specific_text_list = []
        output_summary_text = ''
        general_summary_text = ''
        # specific_summary_text = ''
        word2vec = None
        if similarity_method == 'w2v' or similarity_method == 'avg_w2v' or similarity_method == 'mix':
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_file_path, binary=False)
        stopword_list = nltk.corpus.stopwords.words('english')
        stemmer = nltk.stem.snowball.EnglishStemmer()
        # stemmer = nltk.stem.LancasterStemmer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer()
        with open(input_general_summary_file_path, 'r', encoding='utf8') as f:
            for line in f:
                general_summary_per_line_list.append(line.split())
                general_summary_text += line
        with open(input_general_article_file_path, 'r', encoding='utf8') as f:
            for line in f:
                general_article_per_line_list.append(line.split())
        with open(input_specific_article_file_path, 'r', encoding='utf8') as f:
            for line in f:
                specific_article_per_line_list.append(line.split())
        hypernyms_depth_dict = utils.read_pickle_file(hypernyms_dict_file_path)
        hypernyms_dict = dict()
        for key in hypernyms_depth_dict.keys():
            hypernym_list = []
            for (hyp, depth) in hypernyms_depth_dict[key]:
                hypernym_list.append(hyp)
            hypernyms_dict[key] = hypernym_list
        del hypernyms_depth_dict

        # general_summary_text += line
        print('Files have been read')

        # article_word_name_entity_list = self.word_ner_list(
        #    input_specific_article_file_path=input_specific_article_file_path,
        #    stopword_list=stopword_list, lemmatizer=lemmatizer,
        #    print_per_line=100000)
        stanford_ner_tags = ['PERSON_', 'LOCATION_', 'ORGANIZATION_']
        wordnet_ner_tags = ['person_', 'location_', 'organization_']
        ner_tags = stanford_ner_tags + wordnet_ner_tags
        # count_gen_words = 0
        output_summary_file = open(output_summary_file_path, 'w', encoding='utf8')
        line_index = 0
        for summary_line_word_list, general_article_line_word_list, specific_article_line_word_list in \
                zip(general_summary_per_line_list, general_article_per_line_list, specific_article_per_line_list):
            line_index += 1
            print_flag = False
            if line_index < 10:
                print_flag = True

            summary_line_word_list_ = [w for w in summary_line_word_list]

            generalized_consepts = []
            candidates_for_replacement = []
            sum_index = -1
            for token_s in summary_line_word_list:
                sum_index += 1
                if token_s.find('_') > -1:  # if token_s is generalized
                    generalized_consepts.append(sum_index)
                    art_index = -1
                    for token_a in general_article_line_word_list:
                        art_index += 1
                        hypernyms_list = hypernyms_dict.get(token_a, None)
                        if hypernyms_list and token_s in hypernyms_list:
                            similarity = self.similarity_of_text_v2(word2vec, lemmatizer, stemmer,
                                                                    art_center_word, sum_center_word,
                                                                    summary_line_word_list_,
                                                                    specific_article_line_word_list,
                                                                    sum_window, art_window,
                                                                    art_index, sum_index,
                                                                    stopword_list,
                                                                    similarity_method=similarity_method,
                                                                    decay_factor=0.2)
                            candidates_for_replacement.append((sum_index, art_index, similarity))
            candidates_for_replacement = utils.sort_by_third(candidates_for_replacement, descending=True)
            for (sum_index, art_index, similarity) in candidates_for_replacement:
                if sum_index in generalized_consepts:
                    summary_line_word_list_[sum_index] = specific_article_line_word_list[art_index]
                    generalized_consepts.remove(sum_index)

                    #############
            if print_flag:
                print(general_article_line_word_list)
                print(specific_article_line_word_list)
                print(summary_line_word_list)
                print(summary_line_word_list_)
                # print(' '.join([w.replace('_', ' ') for w in summary_line_word_list_]))
            output_summary_line_text = ''
            for w in summary_line_word_list_:
                if w not in ner_tags:
                    output_summary_line_text += w.replace('_', ' ') + ' '
                else:
                    output_summary_line_text += w + ' '
            if print_flag:
                print(output_summary_line_text)
                print()
            output_summary_file.write(output_summary_line_text.strip() + '\n')
            output_summary_text += output_summary_line_text + '\n'

        output_summary_file.close()
        print('Output file have been written.')
        # t1 = time.time()
        # with open(input_golden_summary_file_path, 'r', encoding='utf8') as f:
        #      specific_summary_text = f.read()
        # cos_sim_of_initial_file = self.cos_similarity_based_on_tfidf(specific_summary_text, general_summary_text)[0][0]
        # cos_sim_of_spec_and_output_file = \
        #    self.cos_similarity_based_on_tfidf(specific_summary_text, output_summary_text)[0][0]
        # cos_sim_of_output_output_file = \
        #   self.cos_similarity_based_on_tfidf(output_summary_text, output_summary_text)[0][0]
        # testing.Testing(testing_mode=testing_mode)
        # print('\ncosine similarities:\n')
        # print('\tcos_sim between golden and general summary: ', cos_sim_of_initial_file)
        # print('\tcos_sim between golde and output summary:   ', cos_sim_of_spec_and_output_file)
        # print('\tcos_sim between the same files (checking):  ', cos_sim_of_output_output_file)
        # t2 = time.time()
        # print('Time for converting, testing & overall: {}, {} & {}'.format(t1 - t0, t2 - t1, t2 - t0))
        print('Time: {}'.format(time.time() - t0))

    def similarity_of_text_v2(self, word2vec, lemmatizer, stemmer,
                              art_center_word, sum_center_word,
                              summary_line_word_list, specific_article_line_word_list,
                              sum_window, art_window,
                              art_index, sum_index, stopword_list, similarity_method,
                              decay_factor=0.2):
        sum_end_index = len(summary_line_word_list) - 1
        art_end_index = len(specific_article_line_word_list) - 1
        similarity = 0.0
        # pre_similarity = 0.0
        for new_left_window in range(1):
            # new_right_window = new_left_window
            art_left_index = art_index - art_window  # new_left_window
            art_right_index = art_index + art_window  # new_right_window
            sum_left_index = sum_index - sum_window  # new_left_window
            sum_right_index = sum_index + sum_window  # new_right_window
            if art_left_index < 0:
                art_left_index = 0
            if art_right_index > art_end_index:
                art_right_index = art_end_index
            if sum_left_index < 0:
                sum_left_index = 0
            if sum_right_index > sum_end_index:
                sum_right_index = sum_end_index
            # art_text = ''
            art_text1 = ''
            # art_text2 = ''
            art_offset_of_center_word = 1
            sum_offset_of_center_word = 1
            if art_center_word:
                art_offset_of_center_word = 0
            if sum_center_word:
                sum_offset_of_center_word = 0
            for w in specific_article_line_word_list[
                     art_left_index:art_index] + specific_article_line_word_list[
                                                 art_index + 1:art_right_index + 1]:
                art_text1 += w + ' '

            art_central_w = specific_article_line_word_list[art_index]

            # art_text = art_text.replace('_', ' ')
            art_text1 = art_text1.replace('_', ' ')
            # art_text2 = art_text2.replace('_', ' ')
            # sum_text = ''
            sum_text1 = ''
            # sum_text2 = ''
            for w in summary_line_word_list[
                     sum_left_index:sum_index] + summary_line_word_list[
                                                 sum_index + 1:sum_right_index + 1]:
                sum_text1 += w + ' '

            sum_text1 = sum_text1.replace('_', ' ')

            temp_txt = ''
            for w in art_text1.split():
                temp_txt += lemmatizer.lemmatize(w) + ' '

            art_text1 = temp_txt

            temp_txt = ''
            for w in sum_text1.split():
                temp_txt += lemmatizer.lemmatize(w) + ' '
            sum_text1 = temp_txt

            similarity = None
            if similarity_method == 'freq':
                similarity = self.cos_similarity_based_on_freq(sum_text1, art_text1)[0][0]
            elif similarity_method == 'tfidf':
                similarity = self.cos_similarity_based_on_tfidf(sum_text1, art_text1)[0][0]
            elif similarity_method == 'w2v':
                similarity = self.word_mover_distance(word2vec, sum_text1, art_text1)
            elif similarity_method == 'edit':
                similarity = self.edit_similarity(sum_text1, art_text1)
                # Levenshtein edit-distance between two strings
            elif similarity_method == 'jacard':
                similarity = nltk.jaccard_distance(set(art_text1.split()), set(sum_text1.split()))
            elif similarity_method == 'avg_w2v':
                similarity = self.cos_sim_of_avg_w2v(word2vec, art_text1, sum_text1,
                                                     art_index - art_left_index,
                                                     art_right_index - art_index,
                                                     sum_index - sum_left_index,
                                                     sum_right_index - sum_index,
                                                     art_center=False, sum_center=False)

                print(similarity)
            elif similarity_method == 'mix':
                similarity = max(self.cos_similarity_based_on_freq(sum_text1, art_text1)[0][0],
                                 self.cos_similarity_based_on_freq(sum_text1, art_text1 + ' ' + art_central_w)[0][0]) + \
                             0.1 * max(self.cos_sim_of_avg_w2v(word2vec, art_text1, sum_text1,
                                                               art_index - art_left_index,
                                                               art_right_index - art_index,
                                                               sum_index - sum_left_index,
                                                               sum_right_index - sum_index,
                                                               art_center=False, sum_center=False),
                                       self.cos_sim_of_avg_w2v(word2vec, art_text1 + ' ' + art_central_w,
                                                               sum_text1,
                                                               art_index - art_left_index,
                                                               art_right_index - art_index,
                                                               sum_index - sum_left_index,
                                                               sum_right_index - sum_index,
                                                               art_center=False, sum_center=False))

        return similarity


    def cos_sim_of_avg_w2v(self, w2v_model, art_text, sum_text,
                           art_left_window, art_right_window, sum_left_window, sum_right_window,
                           art_center, sum_center, decay=0.0):
        text1_list = art_text.split()
        text2_list = sum_text.split()
        w2v_1_list = []
        w2v_2_list = []
        count_unk = 0
        index = -1
        right = art_right_window
        for w in text1_list:
            index += 1
            if art_left_window > 0:
                weight = (1 - decay) ** art_left_window
                art_left_window -= 1
            elif art_left_window == 0 and art_center:
                art_left_window -= 1
                weight = (1 - decay)
            else:
                weight = (1 - decay) ** (right - art_right_window + 1)
                art_right_window -= 1
            try:
                d = np.multiply(w2v_model.word_vec(w), weight)
                w2v_1_list.append(d)  # append(d)
            except KeyError:
                count_unk += 1
                _ = None
                if count_unk > 1:
                    print('Edit Sim')
                    return self.edit_similarity(art_text, sum_text)
        index = -1
        right = sum_right_window
        for w in text2_list:
            index += 1
            # weight = 1.0
            if sum_left_window > 0:
                weight = (1 - decay) ** sum_left_window
                sum_left_window -= 1
            elif sum_left_window == 0 and sum_center:
                sum_left_window -= 1
                weight = (1 - decay)
            else:
                weight = (1 - decay) ** (right - sum_right_window + 1)
                sum_right_window -= 1
            # weight = 1.0 / (abs(c2 - index)*2 + 1)
            try:
                d = np.multiply(w2v_model.word_vec(w), weight)
                w2v_2_list.append(d)
            except KeyError:
                count_unk += 1
                _ = None
                if count_unk > 1:
                    print('Edit Sim')
                    return self.edit_similarity(art_text, sum_text)
        avg_1 = np.average(w2v_1_list, axis=0)
        avg_2 = np.average(w2v_2_list, axis=0)
        try:
            return sklearn.metrics.pairwise.cosine_similarity(avg_1.reshape(1, -1), avg_2.reshape(1, -1))[0][0]
        except ValueError:
            print('Except', art_text, sum_text)
            return self.edit_similarity(art_text, sum_text)

    @staticmethod
    def edit_similarity(art_text, sum_text):
        str_len = max(len(art_text), len(sum_text))
        dist = nltk.edit_distance(art_text, sum_text, substitution_cost=1, transpositions=False)
        return 1.0 - (dist / str_len)



    def word_mover_distance(self, model, text1, text2):
        # model = gensim.models.KeyedVectors.load_word2vec_format(param.word2vec_file_path, binary=False)
        return model.wmdistance(text1, text2)

    @staticmethod
    # it returns a list of tuples: (word, pos)
    def wordnet_pos_tag(text_list):
        pos_tag_list = nltk.tag.pos_tag(text_list)
        word_pos_list = []
        for (w, pos) in pos_tag_list:
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
        return word_pos_list

    def word_ner_list(self, input_specific_article_file_path,
                      stopword_list, lemmatizer,
                      print_per_line=100000):

        # word_freq_hypernyms_dict = utils.read_pickle_file(input_word_freq_hypernyms_pickle_file_path)
        stanford_ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
        wordnet_ner_tags = ['person', 'location', 'organization']
        ner_tags = stanford_ner_tags + wordnet_ner_tags
        # lemmatizer = nltk.stem.WordNetLemmatizer()
        specific_word_ner_list = []
        specific_word_pos_list = []
        text_word_list = []
        with open(input_specific_article_file_path, 'r', encoding='utf8') as f:
            for line in f:
                text_word_list += line.split() + ['_NL_']

        # stanford_ner_tagger_dir = "C:/Stanford_NLP_Tools/stanford-ner-2018-10-16/"
        stanford_ner_tagger_dir = dataset_path.stanford_nlp_tools + 'stanford-ner-2018-10-16/'
        # os.environ['CLASSPATH'] = "/home/pkouris/Stanford_NLP_Tools/stanford-ner-2018-10-16/*"
        model = ['english.all.3class.distsim.crf.ser.gz', 'english.all.3class.distsim.crf.ser.gz']
        ner = nltk.tag.stanford.StanfordNERTagger(
            stanford_ner_tagger_dir + 'classifiers/' + model[0],
            stanford_ner_tagger_dir + 'stanford-ner-3.9.2.jar')
        specific_word_ner_list = ner.tag(text_word_list)
        output_word_name_entity_list = []
        output_line_list = []

        for (word, ner) in specific_word_ner_list:
            flag = True
            index_lineIndex = (-1, -1)
            if word == '_NL_':
                output_word_name_entity_list.append(output_line_list)
                output_line_list = []
                flag = False
            elif word not in stopword_list:
                if ner in stanford_ner_tags:  # and word not in stopword_list:
                    output_line_list.append((word, ner.lower()))
                    flag = False
                else:
                    token_lemma = lemmatizer.lemmatize(word, pos='n')
                    synset = self.make_synset(token_lemma, category='n')
                    if synset is not None:
                        synset.max_depth()
                        merged_synset_list = self.merge_lists(synset.hypernym_paths())
                        hypernym_list = []
                        for synset in merged_synset_list:
                            hypernym_list.append(self.synset_word(synset))
                        for hyp in hypernym_list:
                            if hyp in wordnet_ner_tags:  # and word not in stopword_list:
                                output_line_list.append((word, hyp))
                                flag = False
                                break
            if flag:
                output_line_list.append((word, ''))

        new_output_word_name_entity_list = []
        for line in output_word_name_entity_list:
            index = -1
            new_line_list = []
            prev_ner = ''
            prev_word = ''
            for (word, ner) in line:
                index += 1
                new_word = word
                if ner in ner_tags:
                    if ner == prev_ner:
                        new_word = prev_word + '_' + word
                        new_line_list.remove((prev_word, prev_ner))
                        # print('ner == prev_ner ', new_word)
                prev_ner = ner
                prev_word = new_word
                new_line_list.append((new_word, ner))
            # print(new_line_list)
            new_output_word_name_entity_list.append(new_line_list)

        #############################
        # for line in output_word_name_entity_list:
        #    print(line)
        return new_output_word_name_entity_list


    def cos_similarity_based_on_tfidf(self, text1, text2):
        text_list = [text1, text2]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\w+\b|['#/]")
        tfidf_vectors_list = vectorizer.fit_transform(text_list)
        vector1 = tfidf_vectors_list[0].A[0]
        vector2 = tfidf_vectors_list[1].A[0]
        cos_sim = sklearn.metrics.pairwise.cosine_similarity([vector1], [vector2])
        # print(vectorizer.get_feature_names())
        # print(vector1)
        # print(vector2)
        # print(cos_sim)
        return cos_sim

    def cos_similarity_based_on_freq(self, text1, text2):
        # import sklearn.feature_extraction
        # import sklearn.metrics.pairwise
        # text1 = 'w bush leave for residence_ 2 $ weekday_'
        # text2 = 'president george w'
        # token_pattern = r"(?u)\w+\b" # ignore the punctuation
        text_list = [text1, text2]
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(token_pattern=r"(?u)\w+\b|['#/]")
        freq_vectors_list = vectorizer.fit_transform(text_list)
        # vector1 = freq_vectors_list[0].A
        # vector2 = freq_vectors_list[1].A
        cos_sim = sklearn.metrics.pairwise.cosine_similarity(freq_vectors_list[0].A, freq_vectors_list[1].A)
        # print(vectorizer.get_feature_names())
        # print(vector1)
        # print(vector2)
        # print(cos_sim)
        return cos_sim

    # it returns an dictionayr of hyperonym paths of its word
    def hyperonyms_paths_dict(self, input_article_file_path, input_summary_file_path,
                              output_article_file_path, output_summary_file_path,
                              output_hypernyms_dict_pickle_file_path, output_hypernyms_dict_txt_file_path,
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
        for token in word_set:
            synset = self.make_synset(token, category='n')
            if synset is not None:
                synset.max_depth()
                merged_synset_list = self.merge_lists(synset.hypernym_paths())
                sorted_synsets = self.syncet_sort_accornding_max_depth(merged_synset_list)
                word_depth_list = self.word_depth_of_synsents(sorted_synsets)
                if word_depth_list[0][0] != token:
                    word_depth_list = [(token, word_depth_list[0][1] + 1)] + word_depth_list
                word_hypernym_path_dict[token] = word_depth_list
        del word_set
        ###############
        with open(output_hypernyms_dict_txt_file_path, 'w', encoding='utf8') as f:
            for k, v in word_hypernym_path_dict.items():
                f.write('{} {}\n'.format(k, v))
        with open(output_hypernyms_dict_pickle_file_path, 'wb') as f:
            pickle.dump(word_hypernym_path_dict, f)
        print('Hypernyms have been written to files:\n\t{}\n\t{}'.format(
            output_hypernyms_dict_pickle_file_path, output_hypernyms_dict_txt_file_path))

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
                    elif pos == 'n' and word not in except_words:
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
                    elif pos == 'n' and word not in except_words:
                        try:
                            hypernyms_path_list = word_hypernym_path_dict[word]
                            # hypernym_token = hypernyms_path_list[hypernym_offset][0]
                            depth = hypernyms_path_list[0][1]
                            if depth > max_depth:
                                for el in hypernyms_path_list:
                                    if el[1] <= max_depth:
                                        f.write(el[0] + '_ ')
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
                                           output_dictionary_pickle_file="", output_dict_txt_file="", time_of_pass=1):
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

    def convert_dataset_with_ner(self, input_article_file_path, output_article_file_path,
                                 input_summary_file_path, output_summary_file_path,
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

    def stanford_ner(self, text_list):
        stanford_ner_tagger_dir = "C:/Stanford_NLP_Tools/stanford-ner-2018-10-16/"
        model = ['english.all.3class.distsim.crf.ser.gz', 'english.all.3class.distsim.crf.ser.gz']
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
        size = 0
        for (w, pos) in pos_tag_list:
            size += 1
            wordnet_pos = 'other_pos'
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
        return word_pos_list, size

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


if __name__ == '__main__':
    PostProcessing()
