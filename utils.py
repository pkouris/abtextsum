import pickle


def read_pickle_file(path_to_pickle_file):
        with open(path_to_pickle_file, "rb") as f:
            b = pickle.load(f)
        return b


def sort_by_second(input_list_of_tuples, descending=True):
    if descending:
        return sorted(input_list_of_tuples, key=lambda tup: -tup[1])
    else:
        return sorted(input_list_of_tuples, key=lambda tup: tup[1])


def sort_by_third(input_list_of_tuples, descending=True):
    if descending:
        return sorted(input_list_of_tuples, key=lambda tup: -tup[2])
    else:
        return sorted(input_list_of_tuples, key=lambda tup: tup[2])




