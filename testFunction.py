from testSettings import *


###########################
# Чтение файла в текст
##########################
def readText(fileName, encod):
    f = open(fileName, 'r', encoding=encod)
    text = f.read()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("•", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("\xa0", " ")

    return text

###########################
# Очистка текста и превращение в набор слов
##########################
def text2Words(text):  # функция будет принимать в себя текст и разбивать его на слова, избавляясь от лишних знаков

    # Удаляем лишние знаки
    text = text.replace(".", " ")
    text = text.replace("—", " ")
    text = text.replace(",", " ")
    text = text.replace("!", " ")
    text = text.replace("?", " ")
    text = text.replace("…", " ")
    text = text.replace("-", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace(";", " ")
    text = text.replace(",", " ")
    text = text.lower()  # переводим в нижний регистр

    words = []  # создаем пустой список, в который будем записывать все слова
    currWord = ""  # здесь будет накапливаться текущее слово (между двумя пробелами)

    for symbol in text:  # проходим по каждому символу в тексте

        if (symbol != "\ufeff"):  # игнорируем системный символ в начале строки
            if (symbol != " "):  # если символ не является пробелом
                currWord += symbol  # то добавляем символ в текущее слово
            else:  # если символ является пробелом:
                if (currWord != ""):  # и текущее слово не пустое
                    words.append(currWord)  # то добавляем текущее слово в список слов
                    currWord = ""  # затем обнуляем текущее слово

    # Добавляем финальное слово, если оно не пустое
    # Если не сделать, то потеряем финальное слово, потому что текст чаще всего заканчивается не на пробел
    if (currWord != ""):  # если слово не пустое
        words.append(currWord)  # то добавляем финальное слово в список слов

    return words  # фунция возвращает набор слов




def preprocess_text(document, stop_words):
    stemmer = WordNetLemmatizer()
    # Удаление специальных символов
    document = re.sub(r'\W', ' ', str(document))
    # Удаление всех одиночных символов
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Удаление символов в начале слова
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Замена нескольких пробелов одинарным пробелом
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Удаление бинарного символа 'b'
    document = re.sub(r'^b\s+', '', document)
    # приобразование в нижний регистр
    document = document.lower()

    # Лематизация
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 5]

    return tokens



def words2Indexes(words, vocabulary, maxWordsCount):  # Преобразования листа слов в лист индексов
    wordsIndexes = []

    for word in words:
        wordIndex = 0
        wordInVocabulary = word in vocabulary

        if (wordInVocabulary):
            index = vocabulary[word]
            if (index < maxWordsCount):
                wordIndex = index

        wordsIndexes.append(wordIndex)

    return wordsIndexes

def changeXTo01(trainVector, conceptsCount):          # Преобразование одного короткого вектора в вектор из 0 и 1 # По принципу words bag
  out = np.zeros(conceptsCount)
  for x in trainVector:
    out[x] = 1
  return out

def changeSetTo01(trainSet, conceptsCount):           # Преобразование выборки (обучающей или проверочной) к виду 0 и 1 # По принципу words bag
  out = []
  for x in trainSet:
    out.append(changeXTo01(x, conceptsCount))
  return np.array(out)
