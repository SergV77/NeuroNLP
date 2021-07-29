#!/usr/bin/venv python*
# -*- coding: utf-8 -*-


from testFunction import *
from postFunction import *
from testSettings import *


path_download_dataset = 'baseW/dataframe/dataset_umkb'

df = pd.read_pickle(path_download_dataset)

print(df)
