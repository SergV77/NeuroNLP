from testSettings import *
from testFunction import *


def create_dictionary(data, classes):
    temp_data = {}
    for i, el in enumerate(data):
        if classes[i] in temp_data:
            temp_data[classes[i]].extend(el)
        else:
            temp_data[classes[i]] = el
    return temp_data

def create_count_dictionary(data, classes):
    temp_data_count = {}
    temp_data = create_dictionary(data, classes)
    for key, value in temp_data.items():
        if key in temp_data_count:
            temp_data_count[key].extend(Counter(value))
        else:
            temp_data_count[key] = Counter(value)

    return temp_data_count

def make_datset_zero(data, dictionary):
    dataSet = pd.DataFrame(0, index=[k for k, _ in data.items()],
                           columns=[el[0] for el in dictionary.items()])

    return dataSet

def make_dataset_full(data, dictionary):
    dataSet = make_datset_zero(data, dictionary)
    dataset_tok_one = make_datset_zero(data, dictionary)
    for key, value in tqdm(data.items()):
        for el1 in value.items():
            for el2 in dataSet.columns:
                dataSet.loc[str(key), str(el1[0])] = int(el1[1])
                dataset_tok_one.loc[str(key), str(el1[0])] = 1

    return dataset_tok_one, dataSet

def save_dataset(dataSet, path, name):
    dataSet.to_pickle(path + name)
    dataSet.to_csv(path + name + '.csv')
    with open(path + name + '.npy', 'wb') as np_file:
        np.save(np_file, dataSet)

    print("[+] ФАЙЛЫ СОХРАНЕНЫ")

def normolize_concept(conceptIndexes):
    normolized_concept = []
    for element in conceptIndexes:
        x_array = np.array(element)
        normolized_concept.append(preprocessing.normalize([x_array]))

    return normolized_concept




