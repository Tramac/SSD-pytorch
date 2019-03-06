import os


def create_dir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
