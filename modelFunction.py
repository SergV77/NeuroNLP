from testSettings import *


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