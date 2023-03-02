import os
import pickle
import torch



def save_variable(var_list, filename, folder=None):
    """
    Takes a list of variables and save them in a .pkl file.

    :param var_list: a list of variables to save
    :param filename: name of the file the variables should be save in
    :param folder: name of the subdirectory folder. If given, the .pkl file will be saved in the subdirectory.
    :return: None
    """

    subdir = "./variables"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    if folder is not None:
        path = os.path.join(subdir, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, filename + ".pkl"), 'wb') as f:
            pickle.dump(var_list, f)
    else:
        with open(os.path.join(subdir, filename + ".pkl"), 'wb') as f:
            pickle.dump(var_list, f)


def load_variable(filename, folder=None):
    """
    Load variables from a .pkl file.

    :param filename: name of the file to load the variables from
    :param folder: name of the subdirectory folder. If given, the .pkl file will be load from the subdirectory.
    :return: list of variables loaded from .pkl file
    """
    if folder is not None:
        path = os.path.join("./variables", folder, filename + ".pkl")
    else:
        path = os.path.join("./variables", filename + ".pkl")

    with open(path, 'rb') as f:
        var_list = pickle.load(f)

    return var_list


def save_model(model, filename, folder=None):
    """
    Saves a model in a file.

    :param model: The model that should be saved.
    :param filename: name of the file the model should be save in
    :param folder: name of the subdirectory folder. If given, the model will be saved in the subdirectory.
    :return: None
    """

    subdir = "./models"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    if folder is not None:
        path = os.path.join(subdir, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model, os.path.join(path, filename))
    else:
        torch.save(model, os.path.join(subdir, filename))


def load_model(filename, folder=None):
    """
    Load a model from a file.

    :param filename: name of the file to load the model from
    :param folder: name of the subdirectory folder. If given, the model will be loaded from the subdirectory.
    :return: model from the file
    """

    if folder is not None:
        path = os.path.join("./models", folder, filename)
    else:
        path = os.path.join("./models", filename)

    model = torch.load(path)

    return model


def save_weight(model, filename, folder=None):
    """
    Save weights of a model in a file.

    :param model: The model from which the weights should be saved.
    :param filename: name of the file the weights should be save in
    :param folder: name of the subdirectory folder. If given, the weights will be saved in the subdirectory.
    :return: None
    """

    subdir = "./weights"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    if folder is not None:
        path = os.path.join(subdir, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, filename))
    else:
        torch.save(model.state_dict(), os.path.join(subdir, filename))


def load_weight(model, filename, folder=None):
    """
    Load model weights from a file into the model argument.

    :param model: model to which the weights should be loaded to
    :param filename: name of the file to load the model weights from
    :param folder: name of the subdirectory folder. If given, the weights will be loaded from the subdirectory.
    :return: model with weights from the file
    """

    if folder is not None:
        path = os.path.join("./weights", folder, filename)
    else:
        path = os.path.join("./weights", filename)

    model.load_state_dict(torch.load(path))

    return model


def delete_file(subdir, filename, folder=None):
    """
    Delete the file corresponding to the given path.

    :param subdir: subdirectory in which the to deleted file is located
    :param filename: name of the file which should be deleted
    :param folder: name of the subdirectory folder. If given, the file in the subdirectory will be deleted.
    :return: None
    """

    if folder is not None:
        path = os.path.join(subdir, folder, filename)
    else:
        path = os.path.join(subdir, filename)

    if os.path.isfile(path):
        os.remove(path)
