from testSettings import *

#Рабочая модель
def build_model_bow(maxConceptsCount):

    model = Sequential()
    model.add(Dense(800, input_dim=maxConceptsCount, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(9, activation='softmax'))

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

#Доработать модель, пока не применяется
def build_model_bow_sigm(maxConceptsCount):

    model = Sequential()
    model.add(Dense(800, input_dim=maxConceptsCount, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(9, activation='sigmoid'))

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model