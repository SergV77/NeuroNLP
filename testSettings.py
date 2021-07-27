####################***GENERAL LIB***####################
import os
import re
import pickle
import numpy as np
import requests
import json
import pandas
from tqdm import tqdm
from collections import Counter
import math

####################***TENSORFLOW & KERAS LIB***####################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, Conv2DTranspose, SpatialDropout1D, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization, Flatten, Embedding, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical, plot_model


####################***SKLEARN LIB***####################
from sklearn.preprocessing import LabelEncoder, StandardScaler # Функции для нормализации данных
from sklearn import preprocessing # Пакет предварительной обработки данных
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer


####################***NLP LIB***####################
import spacy
from spacy.lang.ru.examples import sentences
from spacy.symbols import ORTH, LEMMA
from spacy.lang.ru import Russian
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab

# import textacy
# from textacy import extract, preprocessing

####################***NLTK LIB***####################
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

####################***GENSIM LIB***####################
# import gensim
# from gensim import corpora

####################***MATPLOTLIB LIB***####################
import matplotlib.pyplot as plt # Отрисовка изображений