import dataset_path
import parameters as param
dataset_dir = dataset_path.dataset_dir
model_id = param.model_id
duc_model_id = param.duc_model_id

# word embeddings
word2vec_file_path = "/word2vec/file/path"
word_embendings = 'filename of word embendings'


# Train paths
train_dir = dataset_dir + 'train/'  # path to initial train files
train_data_dir = train_dir + 'train_{}_data/'.format(model_id)  # path to x, y, vocabulary etc.
train_article_file_path = train_dir + 'train_{}_article.txt'.format(model_id)
train_title_file_path = train_dir + 'train_{}_title.txt'.format(model_id)

train_baselinegigaword_article_file_path = train_dir + 'train_baselinegigaword_article.txt'
train_baselinegigaword_title_file_path = train_dir + 'train_baselinegigaword_title.txt'
train_neg100gigaword_article_file_path = train_dir + 'train_neg100gigaword_article.txt'
train_neg100gigaword_title_file_path = train_dir + 'train_neg100gigaword_title.txt'
train_neg200gigaword_article_file_path = train_dir + 'train_neg200gigaword_article.txt'
train_neg200gigaword_title_file_path = train_dir + 'train_neg200gigaword_title.txt'
train_neg500gigaword_article_file_path = train_dir + 'train_neg500gigaword_article.txt'
train_neg500gigaword_title_file_path = train_dir + 'train_neg500gigaword_title.txt'
train_neg1000gigaword_article_file_path = train_dir + 'train_neg1000gigaword_article.txt'
train_neg1000gigaword_title_file_path = train_dir + 'train_neg1000gigaword_title.txt'
train_negAllgigaword_article_file_path = train_dir + 'train_negAllgigaword_article.txt'
train_negAllgigaword_title_file_path = train_dir + 'train_negAllgigaword_title.txt'
train_lg100d5gigaword_article_file_path = train_dir + 'train_lg100d5gigaword_article.txt' # generalized word_freq=100, min_depth=5
train_lg100d5gigaword_title_file_path = train_dir + 'train_lg100d5gigaword_title.txt'
train_lg200d5gigaword_article_file_path = train_dir + 'train_lg200d5gigaword_article.txt' # generalized word_freq=200, min_depth=5
train_lg200d5gigaword_title_file_path = train_dir + 'train_lg200d5gigaword_title.txt'
train_lg500d5gigaword_article_file_path = train_dir + 'train_lg500d5gigaword_article.txt' # generalized word_freq=500, min_depth=5
train_lg500d5gigaword_title_file_path = train_dir + 'train_lg500d5gigaword_title.txt'
train_lg1000d5gigaword_article_file_path = train_dir + 'train_lg1000d5gigaword_article.txt'
train_lg1000d5gigaword_title_file_path = train_dir + 'train_lg1000d5gigaword_title.txt'
train_lgAlld5gigaword_article_file_path = train_dir + 'train_lgAlld5gigaword_article.txt'
train_lgAlld5gigaword_title_file_path = train_dir + 'train_lgAlld5gigaword_title.txt'


x_chunks_dir = train_data_dir + 'x_chunks/'
y_chunks_dir = train_data_dir + 'y_chunks/'
x_chunks_sample_dir = train_data_dir + 'x_chunks_sample/'
y_chunks_sample_dir = train_data_dir + 'y_chunks_sample/'
x_txt_dir = train_data_dir + 'x_txt/'
y_txt_dir = train_data_dir + 'y_txt/'
word2int_file_path = train_data_dir + 'word2int_dict.pickle'
int2word_file_path = train_data_dir + 'int2word_dict.pickle'
maxlen_file_path = train_data_dir + 'maxlen.pickle'
word_freq_file_path = train_data_dir + 'word_freq_dict.pickle'
word2int_txt_file_path = train_data_dir + 'word2int_dict.txt'
int2word_txt_file_path = train_data_dir + 'int2word_dict.txt'
maxlen_txt_file_path = train_data_dir + 'maxlen.txt'
vocab_txt_file_path = train_data_dir + 'vocab.txt'
synonyms_dict_pickle_file_path = train_data_dir + 'synonyms_dict.pickle'
synonyms_dict_txt_file_path = train_data_dir + 'synonyms_dict.txt'

if param.sample_training_dataset:
    x_dir = x_chunks_sample_dir
    y_dir = y_chunks_sample_dir
else:
    x_dir = x_chunks_dir
    y_dir = y_chunks_dir

# testing files
validation_dir = dataset_dir + 'validation/'  # path to initial train files
validation_data_dir = validation_dir + '{}/'.format(model_id)  # path to x, y, vocabulary etc.

validation_baselinegigaword_article_file_path = validation_dir + 'validation_baselinegigaword_articles.txt'
validation_baselinegigaword_title_file_path = validation_dir + 'validation_baselinegigaword_titles.txt'

validation_neg100gigaword_article_file_path = validation_dir + 'validation_neg100gigaword_articles.txt'
validation_neg100gigaword_title_file_path = validation_dir + 'validation_neg100gigaword_titles.txt'
validation_neg200gigaword_article_file_path = validation_dir + 'validation_neg200gigaword_articles.txt'
validation_neg200gigaword_title_file_path = validation_dir + 'validation_neg200gigaword_titles.txt'
validation_neg500gigaword_article_file_path = validation_dir + 'validation_neg500gigaword_articles.txt'
validation_neg500gigaword_title_file_path = validation_dir + 'validation_neg500gigaword_titles.txt'
validation_neg1000gigaword_article_file_path = validation_dir + 'validation_neg1000gigaword_articles.txt'
validation_neg1000gigaword_title_file_path = validation_dir + 'validation_neg1000gigaword_titles.txt'
validation_negAllgigaword_article_file_path = validation_dir + 'validation_negAllgigaword_articles.txt'
validation_negAllgigaword_title_file_path = validation_dir + 'validation_negAllgigaword_titles.txt'
validation_lg100d5gigaword_article_file_path = validation_dir + 'validation_lg100d5gigaword_articles.txt'
validation_lg100d5gigaword_title_file_path = validation_dir + 'validation_lg100d5gigaword_titles.txt'
validation_lg200d5gigaword_article_file_path = validation_dir + 'validation_lg200d5gigaword_articles.txt'
validation_lg200d5gigaword_title_file_path = validation_dir + 'validation_lg200d5gigaword_titles.txt'
validation_lg500d5gigaword_article_file_path = validation_dir + 'validation_lg500d5gigaword_articles.txt'
validation_lg500d5gigaword_title_file_path = validation_dir + 'validation_lg500d5gigaword_titles.txt'
validation_lg1000d5gigaword_article_file_path = validation_dir + 'validation_lg1000d5gigaword_articles.txt'
validation_lg1000d5gigaword_title_file_path = validation_dir + 'validation_lg1000d5gigaword_titles.txt'
validation_lgAlld5gigaword_article_file_path = validation_dir + 'validation_lgAlld5gigaword_articles.txt'
validation_lgAlld5gigaword_title_file_path = validation_dir + 'validation_lgAlld5gigaword_titles.txt'

validation_system_summaries_dir = validation_data_dir + 'validation_system_summaries/'
validation_system_summaries_filename_id = 'val_system_summaries.txt'
validation_rouge_results_dir = validation_data_dir + 'validation_rouge_results/'
validation_rouge_results_filename_suffix = 'val_rouge_results.txt'
validation_official_perl_rouge = True
validation_rouge_system_summaries_dir = validation_data_dir + 'rouge_files/rouge_system_summaries'
validation_rouge_model_summaries_dir = validation_data_dir + 'rouge_files/rouge_model_summaries'

validation_article_file_path = validation_dir + 'validation_{}_articles.txt'.format(model_id)
validation_summary_file_path = validation_dir + 'validation_{}_titles.txt'.format(model_id)

# testing files
test_dir = dataset_dir + 'test/'  # path to initial train files
test_data_dir = test_dir + '{}/'.format(model_id)  # path to x, y, vocabulary etc.
test_system_summaries_dir = test_data_dir + 'test_system_summaries/'
test_system_summaries_file_suffix = 'test_system_summaries.txt'
test_x_file_path = test_data_dir + 'x_test.pickle'
test_maxlen_file_path = test_data_dir + 'test_article_max_len.pickle'
test_rouge_files_dir = test_data_dir + 'test_rouge_files/'
test_rouge_system_summaries_dir = test_rouge_files_dir + 'rouge_system_summaries/'
test_rouge_model_summaries_dir = test_rouge_files_dir + 'rouge_model_summaries/'
test_rouge_results_dir = test_data_dir + 'test_rouge_results/'
test_rouge_results_file_suffix = '{}_test_rouge_results.txt'.format(model_id)

test_baselinegigaword_article_file_path = test_dir + 'test_baselinegigaword_articles.txt'
test_baselinegigaword_title_file_path = test_dir + 'test_baselinegigaword_titles.txt'



test_neg100gigaword_article_file_path = test_dir + 'test_neg100gigaword_articles.txt' # min_norm_freq=0.8
test_neg100gigaword_title_file_path = test_dir + 'test_neg100gigaword_titles.txt'
test_neg200gigaword_article_file_path = test_dir + 'test_neg200gigaword_articles.txt'
test_neg200gigaword_title_file_path = test_dir + 'test_neg200gigaword_titles.txt'
test_neg500gigaword_article_file_path = test_dir + 'test_neg500gigaword_articles.txt'
test_neg500gigaword_title_file_path = test_dir + 'test_neg500gigaword_titles.txt'
test_neg1000gigaword_article_file_path = test_dir + 'test_neg1000gigaword_articles.txt'
test_neg1000gigaword_title_file_path = test_dir + 'test_neg1000gigaword_titles.txt'
test_negAllgigaword_article_file_path = test_dir + 'test_negAllgigaword_articles.txt'
test_negAllgigaword_title_file_path = test_dir + 'test_negAllgigaword_titles.txt'
test_neg100d5gigaword_article_file_path = test_dir + 'test_neg100d5gigaword_articles.txt'
test_neg100d5gigaword_title_file_path = test_dir + 'test_neg100d5gigaword_titles.txt'
test_neg200d5gigaword_article_file_path = test_dir + 'test_neg200d5gigaword_articles.txt'
test_neg200d5gigaword_title_file_path = test_dir + 'test_neg200d5gigaword_titles.txt'
test_neg500d5gigaword_article_file_path = test_dir + 'test_neg500d5gigaword_articles.txt'
test_neg500d5gigaword_title_file_path = test_dir + 'test_neg500d5gigaword_titles.txt'
test_neg1000d5gigaword_article_file_path = test_dir + 'test_neg1000d5gigaword_articles.txt'
test_neg1000d5gigaword_title_file_path = test_dir + 'test_neg1000d5gigaword_titles.txt'
test_negAlld5gigaword_article_file_path = test_dir + 'test_negAlld5gigaword_articles.txt'
test_negAlld5gigaword_title_file_path = test_dir + 'test_negAlld5gigaword_titles.txt'


#DUC dataset
duc_dir = 'C:/datasets/textsum/duc2004/'
test_duc_dir = duc_dir + 'test_duc/'  # path to initial train files
test_duc_data_dir = test_duc_dir + '{}/'.format(duc_model_id)  # path to x, y, vocabulary etc.
test_duc_system_summaries_dir = test_duc_data_dir + 'system_summaries/'
test_duc_rouge_files_dir = test_duc_data_dir + 'rouge_files/'
test_duc_rouge_system_summaries_dir = test_duc_rouge_files_dir + 'system_summaries/'
test_duc_rouge_model_summaries_dir = test_duc_rouge_files_dir + 'model_summaries/'
test_duc_golden_summary_file_path_list = \
    [test_duc_dir + 'test_{}_ref1.txt'.format(duc_model_id),
     test_duc_dir + 'test_{}_ref2.txt'.format(duc_model_id),
     test_duc_dir + 'test_{}_ref3.txt'.format(duc_model_id),
     test_duc_dir + 'test_{}_ref4.txt'.format(duc_model_id)]
test_duc_x_file_path = test_duc_data_dir + 'x_test.pickle'
test_duc_rouge_results_dir = test_duc_data_dir + 'rouge_results/'
test_duc_system_summaries_file_suffix = 'system_summaries.txt'
test_duc_rouge_results_file_suffix = 'rouge_results.txt'
test_duc_article_file_path = test_duc_dir + 'test_{}_articles.txt'.format(duc_model_id)
test_duc_maxlen_file_path = test_duc_data_dir + 'test_article_max_len.pickle'
test_duc_ref1 = test_duc_dir + 'test_{}_ref1.txt'.format(duc_model_id)
test_duc_ref2 = test_duc_dir + 'test_{}_ref2.txt'.format(duc_model_id)
test_duc_ref3 = test_duc_dir + 'test_{}_ref3.txt'.format(duc_model_id)
test_duc_ref4 = test_duc_dir + 'test_{}_ref4.txt'.format(duc_model_id)
test_baselineduc_ref1_file_path = test_duc_dir + 'test_baselineduc_ref1.txt'
test_baselineduc_ref2_file_path = test_duc_dir + 'test_baselineduc_ref2.txt'
test_baselineduc_ref3_file_path = test_duc_dir + 'test_baselineduc_ref3.txt'
test_baselineduc_ref4_file_path = test_duc_dir + 'test_baselineduc_ref4.txt'
test_neg100duc_article_file_path = test_duc_dir + 'test_neg100duc_articles.txt'
test_neg100duc_ref1_file_path = test_duc_dir + 'test_neg100duc_ref1.txt'
test_neg100duc_ref2_file_path = test_duc_dir + 'test_neg100duc_ref2.txt'
test_neg100duc_ref3_file_path = test_duc_dir + 'test_neg100duc_ref3.txt'
test_neg100duc_ref4_file_path = test_duc_dir + 'test_neg100duc_ref4.txt'
test_neg200duc_article_file_path = test_duc_dir + 'test_neg200duc_articles.txt'
test_neg200duc_ref1_file_path = test_duc_dir + 'test_neg200duc_ref1.txt'
test_neg200duc_ref2_file_path = test_duc_dir + 'test_neg200duc_ref2.txt'
test_neg200duc_ref3_file_path = test_duc_dir + 'test_neg200duc_ref3.txt'
test_neg200duc_ref4_file_path = test_duc_dir + 'test_neg200duc_ref4.txt'
test_neg500duc_article_file_path = test_duc_dir + 'test_neg500duc_articles.txt'
test_neg500duc_ref1_file_path = test_duc_dir + 'test_neg500duc_ref1.txt'
test_neg500duc_ref2_file_path = test_duc_dir + 'test_neg500duc_ref2.txt'
test_neg500duc_ref3_file_path = test_duc_dir + 'test_neg500duc_ref3.txt'
test_neg500duc_ref4_file_path = test_duc_dir + 'test_neg500duc_ref4.txt'
test_neg1000duc_article_file_path = test_duc_dir + 'test_neg1000duc_articles.txt'
test_neg1000duc_ref1_file_path = test_duc_dir + 'test_neg1000duc_ref1.txt'
test_neg1000duc_ref2_file_path = test_duc_dir + 'test_neg1000duc_ref2.txt'
test_neg1000duc_ref3_file_path = test_duc_dir + 'test_neg1000duc_ref3.txt'
test_neg1000duc_ref4_file_path = test_duc_dir + 'test_neg1000duc_ref4.txt'
test_negAllduc_article_file_path = test_duc_dir + 'test_negAllduc_articles.txt'
test_negAllduc_ref1_file_path = test_duc_dir + 'test_negAllduc_ref1.txt'
test_negAllduc_ref2_file_path = test_duc_dir + 'test_negAllduc_ref2.txt'
test_negAllduc_ref3_file_path = test_duc_dir + 'test_negAllduc_ref3.txt'
test_negAllduc_ref4_file_path = test_duc_dir + 'test_negAllduc_ref4.txt'
test_lg100d5duc_article_file_path = test_duc_dir + 'test_lg100d5duc_articles.txt'  # generalized freq=50, min_depth=5
test_lg100d5duc_ref1_file_path = test_duc_dir + 'test_lg100d5duc_ref1.txt'
test_lg100d5duc_ref2_file_path = test_duc_dir + 'test_lg100d5duc_ref2.txt'
test_lg100d5duc_ref3_file_path = test_duc_dir + 'test_lg100d5duc_ref3.txt'
test_lg100d5duc_ref4_file_path = test_duc_dir + 'test_lg100d5duc_ref4.txt'
test_lg200d5duc_article_file_path = test_duc_dir + 'test_lg200d5duc_articles.txt'
test_lg200d5duc_ref1_file_path = test_duc_dir + 'test_lg200d5duc_ref1.txt'
test_lg200d5duc_ref2_file_path = test_duc_dir + 'test_lg200d5duc_ref2.txt'
test_lg200d5duc_ref3_file_path = test_duc_dir + 'test_lg200d5duc_ref3.txt'
test_lg200d5duc_ref4_file_path = test_duc_dir + 'test_lg200d5duc_ref4.txt'
test_lg500d5duc_article_file_path = test_duc_dir + 'test_lg500d5duc_articles.txt'
test_lg500d5duc_ref1_file_path = test_duc_dir + 'test_lg500d5duc_ref1.txt'
test_lg500d5duc_ref2_file_path = test_duc_dir + 'test_lg500d5duc_ref2.txt'
test_lg500d5duc_ref3_file_path = test_duc_dir + 'test_lg500d5duc_ref3.txt'
test_lg500d5duc_ref4_file_path = test_duc_dir + 'test_lg500d5duc_ref4.txt'
test_lg1000d5duc_article_file_path = test_duc_dir + 'test_lg1000d5duc_articles.txt'
test_lg1000d5duc_ref1_file_path = test_duc_dir + 'test_lg1000d5duc_ref1.txt'
test_lg1000d5duc_ref2_file_path = test_duc_dir + 'test_lg1000d5duc_ref2.txt'
test_lg1000d5duc_ref3_file_path = test_duc_dir + 'test_lg1000d5duc_ref3.txt'
test_lg1000d5duc_ref4_file_path = test_duc_dir + 'test_lg1000d5duc_ref4.txt'
test_lgAlld5duc_article_file_path = test_duc_dir + 'test_lgAlld5duc_articles.txt'
test_lgAlld5duc_ref1_file_path = test_duc_dir + 'test_lgAlld5duc_ref1.txt'
test_lgAlld5duc_ref2_file_path = test_duc_dir + 'test_lgAlld5duc_ref2.txt'
test_lgAlld5duc_ref3_file_path = test_duc_dir + 'test_lgAlld5duc_ref3.txt'
test_lgAlld5duc_ref4_file_path = test_duc_dir + 'test_lgAlld5duc_ref4.txt'

test_article_file_path = test_dir + 'test_subset_{}_articles.txt'.format(model_id)
test_summary_file_path = test_dir + 'test_subset_{}_titles.txt'.format(model_id)

# files with random numbers for creating the test and validation set
test_random_numbers_pickle_file_path = test_dir + 'test_random_numbers.pickle'
test_random_numbers_txt_file_path = test_dir + 'test_random_numbers.txt'
val_random_numbers_pickle_file_path = validation_dir + 'val_random_numbers.pickle'
val_random_numbers_txt_file_path = validation_dir + 'val_random_numbers.txt'

# models
model_dir = dataset_dir + 'models/{}/'.format(model_id)

# reports
reports_dir = dataset_dir + 'reports/'  # {}/'.format(data_id)

# Logs and charts
# Logs
logs_and_charts_dir = dataset_dir + 'logs/{}/'.format(model_id)
logs_dir = logs_and_charts_dir
# Charts
chart_dir = logs_and_charts_dir