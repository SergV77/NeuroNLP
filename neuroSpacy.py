#!/usr/bin/python
# -*- coding: utf-8 -*-

from testFunction import *
from modelFunction import *
from testSettings import *

"""
    ВХОДНОЙ ТЕКСТ   
    ТОКЕНИЗАЦИЯ 
    ЛЕММАТИЗАЦИЯ 
    РРАЗМЕТКА 
    СИНТАКТИЧЕСКИЙ РАЗБОР  
    РАСПОЗНОВАНИЕ СУЩНОСТЕЙ  
    РЕЗУЛЬТАТ
"""
#nlp0 = Russian()
nlp = spacy.load("ru_core_news_lg")
stop_words = nltk.corpus.stopwords.words('russian')

path_download_data = 'baseW/db_word/'
path_save_model = "models"
path_save_dataset = "dataset"
path_save_train_info = "trainInfo"


text = []
classes = []
n = 0
codecs_list = ['UTF-8', 'Windows-1251']

for filename in tqdm(os.listdir(path_download_data)): # Проходим по всем файлам в директории
  n +=1
  for codec_s in codecs_list:
    try:
      text.append(readText(path_download_data+'/'+filename, codec_s)) # Преобразуем файл в одну строку и добавляем в agreements
      classes.append(filename.replace(".txt", ""))
      print('Файл прочитался: ', path_download_data+'/'+filename, 'Кодировка: ', codec_s)
      break
    except UnicodeDecodeError:
       print('Не прочитался файл: ', path_download_data+'/'+filename, 'Кодировка: ', codec_s)
    else:
       next

print_border('ЗАГРУЗКА ФАЙЛОВ ОКОНЧЕНА')
#
# for el in text:
#   print(el)
print_border('ФУНКЦИЯ ТОКЕНАЙЗЕР')

processed_data = [];
for doc in text:
    tokens = preprocess_text(doc, stop_words)
    processed_data.append(tokens)

count = 0
for el in processed_data:
    count += len(el)
    print(len(el))
print(count)

print_border('РАЗМЕТКА')

comb_of_words = []
for sent in text:
    sent = sent.split('  ')
    temp = []
    for el in sent:
        el = " ".join(el.split())
        temp.append(el)
    comb_of_words.append(temp)

count = 0
for el in comb_of_words:
    count += len(el)
    print(len(el))
print(count)

print_border('СИНТАКТИЧЕСКИЙ РАЗБОР SPACY')

comb_token = []
for sent in text:
    doc = nlp(sent)
    temp_list = []
    for token in doc:
        chunk = ''
        if token.pos_ == 'NOUN':
            for w in token.children:
                if w.pos_ == 'ADJ' or w.pos_ == 'DET' or w.pos_ == 'ADP' or w.pos_ == 'ADV':
                    chunk = chunk + w.text + ' '
            chunk = chunk + token.text

        if chunk != '':
            temp_list.append(chunk)

    comb_token.append(temp_list)

count = 0
for el in comb_token:
    count += len(el)
    print(len(el))
print(count)

print_border('ОБЪЕДИНЕНИЕ ТОКЕНОВ 2')

comb_token2 = []
for sent in comb_of_words:
    temp_doc = []
    for words in sent:
        doc = nlp(words)
        temp_list = []
        for token in doc:
            chunk = ''
            if token.pos_ == 'NOUN':
                for w in token.children:
                    if w.pos_ == 'ADJ' or w.pos_ == 'DET' or w.pos_ == 'ADP' or w.pos_ == 'ADV':
                        chunk = chunk + w.text + ' '
                chunk = chunk + token.text

            if chunk != '':
                temp_list.append(chunk)

        temp_doc += temp_list

    comb_token2.append(temp_doc)


count = 0
for el in comb_token2:
    count += len(el)
    print(len(el))
print(count)

print_border('ОБЩЕЕ ОБЪЕДИНЕНИЕ ТОКЕНОВ')


big_temp_list = []
for i, item_i in enumerate(processed_data):
    temp = []
    for j, item_j in enumerate(comb_of_words):
        if i == j:
            temp += item_i
            for k, item_k in enumerate(comb_token):
                if j == k:
                    temp += item_j
                    for l, item_l in enumerate(comb_token2):
                        if k == l:
                            temp += item_k
                            temp += item_l

    big_temp_list.append(temp)


count = 0
for el in big_temp_list:
    count += len(el)
    print(len(el))
print(count)

print_border('ПРОВЕРКА СЛОВАРЯ')
list_all_words = [ item for doc in big_temp_list for item in doc ]
# print(list_all_words)
print(len(list_all_words))

dictionary = Counter(list_all_words)
# print(dictionary)
print(len(dictionary))

vocabulary = createVocabulary(list_all_words)
# print(vocabulary)
print(len(vocabulary))

maxConceptsCount = len(vocabulary)
xLen = 50
step = 2

conceptIndexes = []
for i in range(len(big_temp_list)):
  conceptIndexes.append(words2Indexes(big_temp_list[i], vocabulary, maxConceptsCount))
  print(i, classes[i], len(conceptIndexes[i]))


#
# libs = {'vocabulary': vocabulary, 'classes': classes, 'xLen': xLen, 'step': step}
# with open(path_save_train_info + '/train_info_202107261.pickle', 'wb') as outfile:
#     pickle.dump(libs, outfile)
# print("[+] Служебная информация сохранена")
#
# xTrainIndex = []
# xTestIndex = []
# for i in range(len(conceptIndexes)):
#   (xTrain, yTrain, xTest, yTest) = createTestsClasses(conceptIndexes[i], i, 0.8)
#   xTrainIndex.append(xTrain)
#   xTestIndex.append(xTest)
#
# (xTrain, yTrain) = createSetsMultiClasses(xTrainIndex, xLen, step)
# xTrain01 = changeSetTo01(xTrain, maxConceptsCount)
#
# (xVal, yVal) = createSetsMultiClasses(xTestIndex, xLen, step)
# xVal01 = changeSetTo01(xVal, maxConceptsCount)
#
#
# model_BOW = build_model_bow(maxConceptsCount)
# mcp_save = ModelCheckpoint('models/model_BOW_best202107262', save_best_only=True, monitor='val_loss', mode='min')
# history_model_BOW = model_BOW.fit(xTrain01, yTrain, epochs=40, batch_size=64, callbacks=[mcp_save], validation_data=(xVal01, yVal))
#
#
# plt.plot(history_model_BOW.history['accuracy'],
#          label='Доля верных ответов на обучающем наборе')
# plt.plot(history_model_BOW.history['val_accuracy'],
#          label='Доля верных ответов на проверочном наборе')
# plt.xlabel('Эпоха обучения')
# plt.ylabel('Доля верных ответов')
# plt.legend()
# plt.show()
#



# normolized_concept = []
# for element in conceptIndexes:
#     x_array = np.array(element)
#     normolized_concept.append(preprocessing.normalize([x_array]))
#



# tfidf_dataset = compute_tfidf(big_temp_list)

# with open(path_save_dataset + "/dataset_tfidf_202107261.pickle", "wb") as filesave:
#     pickle.dump(len(tfidf_dataset), filesave)
#     for value in tfidf_dataset:
#         pickle.dump(value, filesave)
#
# print("[+] Файл с датасетом создан")

# with open(PIK, "rb") as f:
#     for _ in range(pickle.load(f)):
#         data2.append(pickle.load(f))
# print data2




# gensim_dictionary = corpora.Dictionary(big_temp_list)
# gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in big_temp_list]
#
# print(gensim_dictionary)

# pickle.dump(gensim_corpus, open('models/gensim_models/gensim_corpus_corpus.pkl', 'wb'))
# gensim_dictionary.save('models/gensim_models/gensim_dictionary.gensim')





#
#
# token_data = []
# for doc in text:
#     doc = nlp(doc)
#     token_data.append([(w.text, w.lemma_) for w in doc])
#
# # for el in token_data:
# #   print(el)
#
# print('*'*150)
# print('*'*75 + 'ЧАСТИРЕЧНАЯ РАЗМЕТКА SPACY' + '*'*75)
# print('*'*150)

# tag_data = []
# for doc in text:
#     doc = nlp(doc)
#     tag_data.append([(w.text, w.pos_, w.tag_, w.dep_) for w in doc])
#
# for el in tag_data[0]:
#     # if el[2] == 'VBG' or el[2] == 'VBG':
#     print(el)
#
# head_data = []
# for doc in text:
#     doc = nlp(doc)
#     head_data.append([(w.head.text, w.dep_, w.text) for w in doc])
#
# for el in head_data[0]:
#     # if el[2] == 'VBG' or el[2] == 'VBG':
#     print(el)

# doc = nlp(text[0])
# for sent in doc.sents:
#     # if el[2] == 'VBG' or el[2] == 'VBG':
#     print([w.text for w in sent if w.dep_ == 'ROOT' or w.dep_ == 'obl' or w.dep_ == 'nmod'  ])


# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# X_train_counts.shape
# (2257, 35788)
# count_vect.vocabulary_.get(u'algorithm')
#
# from sklearn.feature_extraction.text import TfidfTransformer
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
# X_train_tf.shape
#
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_train_tfidf.shape
#
#

