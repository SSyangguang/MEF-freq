import os


def read_MEF_simple(root_path):
    # read the only two image pairs MEF dataset version
    over_root_path = os.path.join(root_path, 'HR_over')
    under_root_path = os.path.join(root_path, 'HR_under')
    over_path = []
    under_path = []
    for root, dirs, files in os.walk(over_root_path, topdown=False):
        for name in files:
            over_path.append(os.path.join(over_root_path, name))

    for root, dirs, files in os.walk(under_root_path, topdown=False):
        for name in files:
            under_path.append(os.path.join(under_root_path, name))

    return over_path, under_path