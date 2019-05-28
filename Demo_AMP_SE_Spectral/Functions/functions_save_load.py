import pickle


def save_object(obj, filename_object):
    """
    Save object with pickle as 'filename_object'
    """
    with open(filename_object, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print('Object saved:', filename_object)


def load_object(filename_object):
    """
    Load 'filename_object' with pickle
    """
    with open(filename_object, 'rb') as input:
        obj = pickle.load(input)
    print("Object loaded:", filename_object)
    return obj
