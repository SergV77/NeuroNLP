####################***GENERAL LIB***####################
import os
import re
import pickle
import numpy as np
from tqdm import tqdm


####################***TENSORFLOW & KERAS***####################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model


####################***NLP***####################
import spacy
from spacy.lang.ru.examples import sentences
from spacy.symbols import ORTH, LEMMA
from spacy.lang.ru import Russian
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab

# import textacy
# from textacy import extract, preprocessing

####################***NLTK***####################
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer