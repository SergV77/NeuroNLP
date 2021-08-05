from testSettings import *
from testFunction import *


################################РАЗМЕТКА##############################
def make_markup(text):
    comb_of_words = []
    for sent in text:
        sent = sent.split('  ')
        temp = []
        for el in sent:
            el = " ".join(el.split())
            temp.append(el)
        comb_of_words.append(temp)

    return comb_of_words

#######################СИНТАКТИЧЕСКИЙ РАЗБОР SPACY####################
def syntax_parsing(text, nlp):
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

    return comb_token

def syntax_parsing_comb(comb_of_words, nlp):
    comb_token = []
    for sent in comb_of_words:
        temp_doc = []
        for words in sent:
            doc = nlp(words)
            temp_list = []
            for token in doc:
                chunk = ''
                if token.pos_ == 'NOUN':
                    for w in token.children:
                        if w.pos_ == 'ADJ' or w.pos_ == 'DET' or w.pos_ == 'ADP':
                            chunk = chunk + w.text + ' '
                    chunk = chunk + token.text

                if chunk != '':
                    temp_list.append(chunk)

            temp_doc += temp_list

        comb_token.append(temp_doc)

    return comb_token

def concat_tokens(list_doc, nlp):
    list_doc
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

    return big_temp_list
