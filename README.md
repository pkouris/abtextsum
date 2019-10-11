### abtextsum
This source code has been used in the experimental procedure of the following paper:

Panagiotis Kouris, Georgios Alexandridis, Andreas Stafylopatis. 2019. Abstractive text summarization based on deep learning and semantic content generalization. _In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_, pp. 5082-5092.<br/>
<br/>


This paper is accessible in the [Proceedings of the 57th ACL Annual Meeting (2019)](https://www.aclweb.org/anthology/events/acl-2019/) or directly from [here](https://www.aclweb.org/anthology/P19-1501). 
<br/> For citing, the BibTex follows:
```
@inproceedings{kouris2019abstractive,
  title={Abstractive text summarization based on deep learning and semantic content generalization},
  author={Kouris, Panagiotis and Alexandridis, Georgios and Stafylopatis, Andreas},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  month = jul,
  year={2019}
  address = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
  url = {https://www.aclweb.org/anthology/P19-1501},
  pages={5082--5092},
}
```
<br/><br/>

### Code Description
The code described below follows the methodology and the assumptions which are described in detail in the aforementioned paper.
The experimental procedure, as it is described in the paper, requires as initial dataset for training, validation and testing the _Gigaword_ dataset as it is described by Rush et. al. 2015 (see references in the paper). Also for testing, the _DUC 2004_ dataset is used as this is also described in the paper.
<br/>
According to the paper, the initial dataset is preprocessed furthermore and generalized to one of the proposed text generalization strategies (e.g. _NEG100_ or _LG200d5_). Then the generalized dataset is used for training where the deep learning model learns to predict a generalized summary.
<br/>
In the phase of testing, a generalized article (e.g. an article of the test set) is given as input to the deep learning model which predicts the respective generalized summary. Then, in the phase of post-processing, the generalized concepts of the generalized summary are replaced by the specific concepts of the original (preprocessed) article producing the final summary.<br/><br/>  

More specifically, the main steps of this framework are as follows:

1. Preprocessing of the dataset<br/> 
The task of preprocessing of the dataset is performed by _DataPreprocessing_ class (_preprocessing.py_ file). The method _clean_dataset()_ is used for preprocessing the _Gigaword_ dataset while the method _clean_duc_dataset_from_original_to_cleaned()_ is used for _DUC_ dataset. 


1. Text generalization<br/>
Both text generalization tasks, _NEG_ and _LG_, are performed by _DataPreprocessing_ class (_preprocessing.py_ file).<br/>
Firstly, part-of-speach tagging is required which is performed by _pos_tagging_of_dataset_and_vocabulary_of_words_pos_frequent()_ method for _Gigaword_ dataset and _pos_tagging_of_duc_dataset_and_vocab_pos_frequent()_ method for _DUC_ dataset. Then the _NEG_ and _LG_ strategy can be applied as follows:
   1. NEG Strategy<br/>
The annotation of named entities is performed by _ner_of_dataset_and_vocabulary_of_ner_words()_ method for _Gigaword_ dataset and _ner_of_duc_dataset_and_vocab_of_ne()_ method for _DUC_ dataset. Then the methods _conver_dataset_with_ner_from_stanford_and_wordnet()_ for _Gigaword_ dataset and _conver_duc_dataset_with_ner_from_stanford_and_wordnet()_ for _DUC_ dataset generalize these datasets according to _NEG_ strategy having set the parameters accordingly. 

   1. LG Strategy<br/> 
The _word_freq_hypernym_paths()_ method produces a file that contains a vocabulary with the frequency and the hypernym path of each word. Then this file is used by _vocab_based_on_hypernyms()_ method in order to produce a file that contains a vocabulary with those words that are candidates for generalization. Finally, for the _Gigaword_ dataset, the _convert_dataset_to_general()_ method produces the files with summary-article pairs which constitute the generalized dataset, while for _DUC_ dataset the _convert_duc_dataset_based_on_level_of_generalizetion()_ method is used. The hyperparameters of these methods should be set accordingly.


1. Building dataset for training, validation and testing<br/>
The _BuildDataset_ class (_build_dataset.py_ file) creates the files which are given as input to the deep learning model for training, validation or testing. The hyperparameters should be set accordingly.
</br>For creating the dataset, the appropriate files paths should be set in the _\_\_inint\_\_()_ of _BuildDataset_ class executing the following commands: for building the training dataset ```python build_dataset.py -mode train -model lg100d5g```, for building the validation dataset ```python build_dataset.py -mode validation -model lg100d5g``` and for building the testing dataset ```python build_dataset.py -mode test -model lg100d5g```, where the argument _-model_ specifies the employed generalization strategy (e.g. lg100d5 or neg100).



1. Training<br/>
The process of training is performed by _Train_ Class (file _train_v2.py_) having set the hyperparameters accordingly. The files which are produced from the previous step of Building dataset are used as input in this phase of training.
The process of training is performed by the command: ```python train.py -model neg100```, where the argument -model specifies the employed generalization strategy (e.g. lg100d5 or neg100).

1. Post-processing of generalized summaries<br/>
In the phase of testing, the task of post-processing of the generalized summaries, which are produced by the deep learning model, is required to replace the generalized concepts of the generalized summary with the specific ones from the corresponding original articles. This task is performed by _PostProcessing_ class by setting the parameters in _\_\_init\_\_()_ method accordingly. More specifically, the mode should be set to _"lg"_ or _"neg"_ according to the employed text generalization strategy. Also, the hyperparameters of _neg_postprocessing()_ and _lg_postprocessing()_ methods for file paths, text similarity function and the context window should be set accordingly.


1. Testing<br/>
The _Testing_ class (file _testing.py_) performs the process of testing of this framework. For the _Gigaword_ dataset, a subset of its test set (e.g. 4000 instances) should be used in order to evaluate the framework while for the _DUC_ dataset, the whole set of instances is used. The _Testing_ class requires the official _ROUGE package_ for measuring the performance of the proposed framework.<br/>
In order to perform the task of testing, the appropriate file paths should be set in the _\_\_init\_\_()_ of _Testing_ class running one of the following modes; testing for gigaword: ```python testing.py -mode gigaword```, testing for duc: ```python testing.py -mode duc``` and testing for duc capped to 75 bytes: ```python testing.py -mode duc75b```<br/><br/>



**Setting parameters and paths**<br/>
The values of hyperparameters should be specified in the file _parameters.py_, while the paths of the corresponding files should be set in the file _paths.py_, 
Additionally, a file with word embeddings (e.g. _word2vec_) is required where its file path and the dimension of the vectors (e.g. 300) should be specified in the files _paths.py_ and _parameters.py_, respectively.<br/><br/>

The project was developed in python 3.5 and the required python packages are included in the file _requirements.txt_.


The above described code includes the functionality that was used in the experimental procedure of the corresponding paper. However, the proposed framework is not limited by the current implementation as it is based on a well defined theoretical model that may provide the possibility of enhancing its performance by extending or improving this implementation (e.g. using a better taxonomy of concepts, a different machine learning model or an alternative similarity method for the post-processing task). 






<!--
```
@inproceedings{kouris-etal-2019-abstractive,
    title = "Abstractive Text Summarization Based on Deep Learning and Semantic Content Generalization",
    author = "Kouris, Panagiotis  and Alexandridis, Georgios  and Stafylopatis, Andreas",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1501",
    pages = "5082--5092",
    abstract = "This work proposes a novel framework for enhancing abstractive text summarization based on the combination of deep learning techniques along with semantic data transformations. Initially, a theoretical model for semantic-based text generalization is introduced and used in conjunction with a deep encoder-decoder architecture in order to produce a summary in generalized form. Subsequently, a methodology is proposed which transforms the aforementioned generalized summary into human-readable form, retaining at the same time important informational aspects of the original text and addressing the problem of out-of-vocabulary or rare words. The overall approach is evaluated on two popular datasets with encouraging results.",
}
```
-->
