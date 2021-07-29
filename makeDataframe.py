#!/usr/bin/venv python*
# -*- coding: utf-8 -*-


from testFunction import *
from postFunction import *
from testSettings import *


path_download_data_id = 'baseW/db_concepts'
path_download_data_names = 'baseW/db_names_concepts'
path_download_dict_id_name = 'baseW/db_dict_id_name/'
path_save_dataset = 'baseW/dataframe/'


texts_id, classes_id = open_numpy(path_download_data_id)
texts_names, classes_name = open_numpy(path_download_data_names)
dict_id_name, classes_dict = open_numpy(path_download_dict_id_name)

all_set_name = []
for el in texts_names:
    set_name = set(el)
    all_set_name += set_name


new_dict_id_name = {}
for el in dict_id_name:
    for k, v in el.tolist().items():
        if k in new_dict_id_name:
            new_dict_id_name[k].extend(v)
        else:
            new_dict_id_name[k] = v


dataSet = pd.DataFrame(0, index=[k for k, _ in new_dict_id_name.items()],
                       columns=list(set(all_set_name)))



for key, value in tqdm(new_dict_id_name.items()):
    for el2 in value:
        for k, v in el2.items():
            for el3 in dataSet.columns:
                dataSet.loc[str(key), str(v)] = 1



dataSet.to_pickle(path_save_dataset + 'datasetUmkb')
dataSet.to_csv(path_save_dataset + 'datasetUmkb.csv')

with open(path_save_dataset + 'datasetUmkb.npy', 'wb') as np_file:
    np.save(np_file, dataSet)





# print_border('\nID КОНЦЕПТОВ')

# all_set_id = []
# for el in texts_id:
#     set_id = set(el)
#     all_set_id.append(set_id)
#     # print(len(set_id))
# # print(texts_id)




# new_dict_id_name = {}
# for el in dict_id_name[0]:
#     for k, v in el.items():
#         new_dict_id_name[k] = v



# print(dataSet)


# df = pd.DataFrame([v for _, val in new_dict_id_name.items() for k, v in val.items()],
#                   index=[k for k, _ in new_dict_id_name.items()],
#                   columns=all_set_name)
# print(df)
#
# # for i in range(len(dataSet.index)):
# #     if dataSet.index ==
#
# print(dataSet)

# print(pd.DataFrame.from_dict(new_dict_id_name[0]))


# dataSet = pd.DataFrame(0, index=[k for el in dict_id_name for k, _ in el.items()],
#                        columns=[v for _, v in dict_id_name)



# print(temp_dict)
