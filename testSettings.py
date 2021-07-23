####################***GENERAL LIB***####################
import os
import pickle
import numpy as np


####################***TENSORFLOW & KERAS***####################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model


####################***NLP***####################
import spacy
from spacy.lang.ru.examples import sentences
from spacy.symbols import ORTH, LEMMA