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

################################
# Загрузка файлов из директории
################################
def open_text(path):
    texts = []
    classes = []
    codecs_list = ['UTF-8', 'Windows-1251']
    for filename in os.listdir(path):
        for codec_s in codecs_list:
            try:
                texts.append(readText(path + '/' + filename, codec_s)) # Считываем файл
                classes.append(filename.replace(".txt", ""))
                break
            except UnicodeDecodeError:
                next
                print('Не прочитался файл: ', path + '/' + filename, 'Кодировка: ', codec_s)
            else:
                next

    return texts, classes

def open_numpy(path):
    classes = []
    allLoadData = []
    for filename in os.listdir(path):
        loaded_data = np.load(path + '/' + filename, allow_pickle=True)
        classes.append(filename.replace(".npy", ""))
        allLoadData.append(loaded_data)

    return allLoadData, classes

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


def createVocabulary(allWords):  # Создание словаря - все слова, упорядоченные по частоте появления

    wCount = dict.fromkeys(allWords, 0)

    for word in allWords:
        wCount[word] += 1

    wordsList = list(wCount.items())
    wordsList.sort(key=lambda i: i[1], reverse=1)

    sortedWords = []

    for word in wordsList:
        sortedWords.append(word[0])

    wordIndexes = dict.fromkeys(allWords, 0)
    for word in wordIndexes.keys():
        wordIndexes[word] = sortedWords.index(word) + 1

    return wordIndexes

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


def compute_tfidf(corpus):
    def compute_tf(text):
        tf_text = Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i] / float(len(text))

        return tf_text

    def compute_idf(word, corpus):
        return math.log10(len(corpus) / sum([1.0 for i in corpus if word in i]))

    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
            documents_list.append(tf_idf_dictionary)

    return documents_list


def getSetFromIndexes(conceptIndexes, xLen,
                      step):  # Формирование обучающей выборки по листу индексов концептов (разделение на короткие векторы)
    xTrain = []
    conceptLen = len(conceptIndexes)
    index = 0
    while (index + xLen <= conceptLen):
        xTrain.append(conceptIndexes[index:index + xLen])
        index += step
    return xTrain


def createSetsMultiClasses(conceptIndexes, xLen,
                           step):  # Формирование обучающей и проверочной выборки выборки из 10 листов индексов от 10 классов
    nClasses = len(conceptIndexes)
    classesXTrain = []
    for cI in conceptIndexes:  # Для каждого из 10 классов
        classesXTrain.append(getSetFromIndexes(cI, xLen, step))  # Создаём обучающую выборку из индексов

    xTrain = []  # Формируем один общий xTrain
    yTrain = []

    for t in range(nClasses):
        xT = classesXTrain[t]
        for i in range(len(xT)):
            xTrain.append(xT[i])

        currY = to_categorical(t, nClasses)  # Формируем yTrain по номеру класса
        for i in range(len(xT)):
            yTrain.append(currY)

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    return (xTrain, yTrain)


def createTestsClasses(allIndexes, i, train_size):
    # Формируем общий xTrain и общий xTest
    X_train, X_test, y_train, y_test = np.array(
        train_test_split(allIndexes, np.ones(len(allIndexes), 'int') * (i + 1), train_size=train_size))

    return (X_train, y_train, X_test, y_test)

def translateDis(dictClass, item):
    for k, v in dictClass.items():
        if k == item:
           return v


def print_border(info):
    print('*' * 150)
    print('*' * 75 + info + '*' * 75)
    print('*' * 150)


