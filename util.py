import os


def get_project_path():
    """
    get the absolute path to project's root directory
    """

    path= os.getcwd().split('Holy-Miner')[0] + 'Holy-Miner/'

    return path

root_path = get_project_path()


