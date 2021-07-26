from testFunction import *
from modelFunction import *
from testSettings import *


path = 'baseW/db_concepts'
path_save_model = "models"
path_save_dataset = "dataset"
path_save_train_info = "trainInfo"



loaded_data = []
allLoadData = []
classes = []
nClasses = 0

for filename in os.listdir(path):
  loaded_data = np.load(path + '/' + filename, allow_pickle=True)
  classes.append(filename.replace(".npy", ""))
  print('Файл прочитался: ', path + '/' + filename)
  allLoadData.append(loaded_data)

nClasses = len(classes)
#
# for doc in allLoadData:
#   print(doc)

list_all_words = [el for doc in allLoadData for item in doc for el in item]
print(list_all_words)
print(len(list_all_words))
vocabulary = createVocabulary(list_all_words)
print(vocabulary)
print(len(vocabulary))

maxConceptsCount = len(vocabulary)
xLen = 50
step = 2

conceptIndexes = []
for i in range(len(allLoadData)):
  temp = [item for doc in allLoadData[i] for item in doc]
  print(f'{i} - ', len(temp))
#   conceptIndexes.append(words2Indexes(allLoadData[i], vocabulary, maxConceptsCount))
#   print(i, classes[i], len(conceptIndexes[i]))
#
# normolized_concept = []
# for element in conceptIndexes:
#     x_array = np.array(element)
#     normolized_concept.append(preprocessing.normalize([x_array]))
#
#
# #
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