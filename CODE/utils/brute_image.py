from subprocess import call

import sys
import threading

from utils.utils import fix_path
from time import sleep

path = fix_path()
root = path.replace('metrics_evaluation', '')
print(root)
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([root, root + 'metrics_evaluation/CODE',
                 root + 'metrics_evaluation/CODE/libsvm/python',
                 root])

if __name__ == '__main__':

    nb = 3
    nb_cu = 0
    threadLimiter = threading.BoundedSemaphore(nb)

    def execute(dataset, method):
        global threadLimiter
        threadLimiter.acquire()
        global nb_cu
        for _ in range(repeat):
            for test in tests:
                # profiler = profiles.get(test, ['same', 'same'])
                call([sys.executable, "CODE/main.py", "--file1", dataset, "--type", method, "--test_type", test,
                      '--log_name', dataset, "--data_type", "Image", "--read", "False"])
                nb_cu -= 1

        print("finished")
        threadLimiter.release()


    datasets = [
        'mnist', 'usps', 'svhn'
    ]

    generate = ['class', 'cluster_removal', 'test_split']
    times = [6, 34, 40]
    tests = ['two_sample_t_test', 'one_sample_t_test', 'anova', 'mann', 'kruskal',
             'levene_mean', 'levene_median', 'levene_trimmed',
             'sign_test', 'median_test',
             'min_max', 'ks', 'kupier', 'a_dist']
    profiles = {'one_sample_t_test': ['mean', 'same'], 'sign_test': ['median', 'same'], 'min_max': ['min_max', 'same']}
    for dataset in datasets:
        for i in range(len(generate)):
            method = generate[i]
            repeat = times[i]
            if not (dataset == 'golf_ball' and i == 0):
                # for _ in range(repeat):
                    nb_cu += 1
                    # executor = ThreadPoolExecutor(max_workers=nb)
                    # executor.submit(execute, dataset, method)
                    # threading.Thread(target=f)
                    # executor.

                    thread1 = threading.Thread(target=execute, args=(dataset, method))
                    thread1.start()
                    while nb_cu >= nb:
                        print(nb_cu)
                        sleep(50)