#!/usr/bin/venv python
# -*- coding: utf-8 -*-

from testFunction import *
from postFunction import *
from testSettings import *


path_download_data_id = 'baseW/db_concepts'
path_download_data_names = 'baseW/db_names_concepts'


texts_id, classes = open_numpy(path_download_data_id)
texts_names, classes = open_numpy(path_download_data_names)

print_border('ID КОНЦЕПТОВ')
print(texts_id)

print_border('НАЗВАНИЕ КОНЦЕПТОВ')
print(texts_names)