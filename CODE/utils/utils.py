import os
from glob import glob


def fix_path():
    # os.chdir('/Users/imad/PycharmProjects/')
    current = os.getcwd()
    # print(os.getcwd())
    if 'metrics_evaluation' in current:
        fixed_path = os.path.join(current.split('metrics_evaluation')[0], 'metrics_evaluation')
        # print(fixed_path)
        os.chdir(fixed_path)
    elif os.path.isdir(os.path.join(current, 'metrics_evaluation')):
        fixed_path = os.path.join(current, 'metrics_evaluation')
        # print(fixed_path)
        os.chdir(fixed_path)
    else:
        search = current + "/*/"
        if '//' in search:
            search = str(search).replace('//', '/')
        potentials = glob(search)
        potentials = [glob(search_i + "/*/") for search_i in potentials]
        # print(potentials)
        fixed_path = current
        for p in potentials:
            for path in p:
                if 'metrics_evaluation' in path:
                    fixed_path = os.path.join(path.split('metrics_evaluation')[0], 'metrics_evaluation')
                    # print(fixed_path)
                    os.chdir(fixed_path)
                    return fixed_path
    return fixed_path
