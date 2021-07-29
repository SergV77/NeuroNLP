from testFunction import *
from testSettings import *



model_BOW = keras.models.load_model('models/model_BOW_best202107152')
model_Emb = keras.models.load_model('models/model_Embd202107151')

with open('models/model_best20210715.pickle', 'rb') as f:
    libs = pickle.load(f)

# загружаем тестовый текст
text = input("Введите симптомы: ")
loaded_test = text2Words(text)
print("\n", loaded_test)

dictClass = {
    "Acute appendicitis": "Острый аппендицит",
    "Duodenitis": "Дуоденит",
    "Acute pancreatitis": "Острый панкреатит",
    "Peptic ulcer and gastritis": "Язвенная болезнь желудка и гастрит",
    "Peritonitis": "Перитонит",
    "Pyloroduodenal stenosis": "Стеноз пилородуоденальный",
    "Ulcer perforation": "Перфорация язвы",
    "Ulcerative bleeding": "Язвенное кровотечение",
    "Acute cholecystitis": "Острый холецистит",
}

vocabulary = libs['vocabulary']
classes = libs['classes']
nClasses = len(classes)
xLen = libs['xLen']
maxConceptsCount = len(vocabulary) + 1
# print(classes)
# преобразуем полученный массив концептов в массив индексов согласно словаря
conceptIndexes = []
conceptIndexes.append(words2Indexes(loaded_test, vocabulary, maxConceptsCount))

print(conceptIndexes)
xTest = changeSetTo01(conceptIndexes, maxConceptsCount)
print(xTest)
out_BOW = model_BOW.predict([xTest])
# out_Emb = model_Emb.predict(xTest)
#print(out_BOW)
#print(np.argmax(out_BOW))

data = list(out_BOW[0])

dic = {}
for num in out_BOW[0]:
    if ((num * 100) > 1):
        dic[classes[data.index(num)]] = num * 100


for key, value in dic.items():

    print(f'Вероятность заболевания {translateDis(dictClass, key)} - ', round(value, 2), '%')


# diagnosis1 = classes[np.argmax(out)]


