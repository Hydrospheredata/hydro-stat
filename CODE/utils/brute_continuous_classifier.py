from subprocess import call

import sys

from utils.utils import fix_path

path = fix_path()
root = path.replace('metrics_evaluation', '')
print(root)
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([root, root + 'metrics_evaluation/CODE',
                 root + 'metrics_evaluation/CODE/libsvm/python',
                 root])

if __name__ == '__main__':
    datasets = [
    ]

    generate = ['cluster_removal', 'test_split']
    times = [3, 3]
    classifiers = ['lg', 'svm', 'nb', 'knn', 'dt', 'xgboost']
    profiles = {'one_sample_t_test': ['mean', 'same'], 'sign_test': ['median', 'same'], 'min_max': ['min_max', 'same'],
                'hull': ['delaunay', 'same']}
    for dataset in datasets:
        for i in range(len(generate)):
            method = generate[i]
            repeat = times[i]
            if not (dataset == 'golf_ball' and i == 0):
                for _ in range(repeat):
                    for classifier in classifiers:
                        call(["python3", "training_on_all.py", "--file1", dataset, "--type", method, "--classifier",
                              classifier, '--log_name', dataset + 'classification'])
