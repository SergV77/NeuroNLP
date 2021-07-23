from testFunction import *
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
nlp0 = Russian()
nlp = spacy.load("ru_core_news_lg")
stop_words = nltk.corpus.stopwords.words('russian')

path = 'baseW/db_word/'
text = []
classes = []
n = 0
codecs_list = ['UTF-8', 'Windows-1251']

for filename in tqdm(os.listdir(path)): # Проходим по всем файлам в директории
  n +=1
  for codec_s in codecs_list:
    try:
      text.append(readText(path+'/'+filename, codec_s)) # Преобразуем файл в одну строку и добавляем в agreements
      classes.append(filename.replace(".txt", ""))
      print('Файл прочитался: ', path+'/'+filename, 'Кодировка: ', codec_s)
      break
    except UnicodeDecodeError:
       print('Не прочитался файл: ', path+'/'+filename, 'Кодировка: ', codec_s)
    else:
       next

print('*'*150)
print('*'*75 + 'ЗАГРУЗКА ФАЙЛОВ ОКОНЧЕНА' + '*'*75)
print('*'*150)
#
# for el in text:
#   print(el)

print('*'*150)
print('*'*75 + 'ФУНКЦИЯ ТОКЕНАЙЗЕР' + '*'*75)
print('*'*150)
processed_data = [];
for doc in text:
    tokens = preprocess_text(doc, stop_words)
    processed_data.append(tokens)

for el in processed_data:
  print(el)

print('*'*150)
print('*'*75 + 'РРАЗМЕТКА' + '*'*75)
print('*'*150)

comb_of_words = []
for sent in text:
    sent = sent.split('  ')
    temp = []
    for el in sent:
        el = " ".join(el.split())
        temp.append(el)
    comb_of_words.append(temp)

for el in comb_of_words:
  print(el)

print('*'*150)
print('*'*75 + 'СИНТАКТИЧЕСКИЙ РАЗБОР SPACY' + '*'*75)
print('*'*150)

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


for el in comb_token:
  print(el)

print('*'*150)
print('*'*75 + 'ОБЪЕДИНЕНИЕ ТОКЕНОВ 2' + '*'*75)
print('*'*150)

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


for el in comb_token2:
  print(el)


print('*'*150)
print('*'*75 + 'ОБЩЕЕ ОБЪЕДИНЕНИЕ ТОКЕНОВ' + '*'*75)
print('*'*150)



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

for el in big_temp_list:
  print(el)








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


