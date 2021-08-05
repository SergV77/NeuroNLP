#!/usr/bin/venv python*
# -*- coding: utf-8 -*-


from testFunction import *
from postFunction import *
from testSettings import *


path_download_dataset = 'baseW/dataframe/dataset_umkb'
path_download_data = 'baseW/db_word/'

path_save_model = "models"
path_save_dataset = "dataset"
path_save_train_info = "trainInfo"
path_save_dataset_allWord = "dataset/datasetAllWords/"


stop_words = nltk.corpus.stopwords.words('russian')
text, classes = open_text(path_download_data)
print_border('ЗАГРУЗКА ФАЙЛОВ ОКОНЧЕНА')

#######################################################################
#######################################################################
print_border('ТОКЕНИЗАЦИЯ И ЛЕММАТИЗАЦИЯ ТЕКСТА')
processed_data = []
processed_data_all = []
for doc in text:
    tokens = preprocess_text(doc, stop_words)
    processed_data.append(tokens)
    processed_data_all += tokens

# temp_data_count = create_count_dictionary(processed_data, classes)
# dictionary = Counter(processed_data_all)
# print(dictionary)
# print(len(dictionary))
#
# print_border('СОЗДАНИЕ ДАТАСЕТА ИЗ ЛЕММАТИЗИРОВАННЫХ ДАННЫХ')
#
# dataset_tok_lem_one, dataSet = make_dataset_full(temp_data_count, dictionary)
# print(dataSet)
# print(dataset_tok_lem_one)
#
#
# print_border('СОХРАНЕНИЕ ГОТОВОГО ДАТАСЕТА ИЗ ЛЕММАТИЗИРОВАННЫХ ДАННЫХ')
#
# save_dataset(dataSet, path_save_dataset_allWord, 'dataset_token_lemm')
# save_dataset(dataset_tok_lem_one, path_save_dataset_allWord, 'dataset_token_lemm_one')

########################################################################################
########################################################################################
print_border('ТОКЕНИЗАЦИЯ ТЕКСТА')
words = []
allwords = []
for i in range(len(text)):
  words.append(text2Words(text[i]))
  print(i, classes[i], len(words[i]))
  allwords += words[i]


# temp_data_count = create_count_dictionary(words, classes)
# dictionary = Counter(allwords)
# print(dictionary)
# print(len(dictionary))
#
# print_border('СОЗДАНИЕ ДАТАСЕТА ИЗ ДАННЫХ')
#
# dataset_tok_one, dataSet = make_dataset_full(temp_data_count, dictionary)
# print(dataSet)
# print(dataset_tok_one)
#
# print_border('СОХРАНЕНИЕ ГОТОВОГО ДАТАСЕТА ИЗ ОРИГЕНАЛЬНЫХ ДАННЫХ')
#
# save_dataset(dataSet, path_save_dataset_allWord, 'dataset_token')
# save_dataset(dataset_tok_one, path_save_dataset_allWord, 'dataset_token_one')
#
# print_border('ДАТАСЕТ СОХРАНЕН')


df = pd.read_pickle(path_download_dataset)

print(df)
