import threading
from subprocess import call

import sys
from time import sleep

from utils.utils import fix_path

# from sklearn.model_selection import train_test_split
path = fix_path()
root = path.replace('metrics_evaluation', '')
# print(root)
# print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([root, root + 'metrics_evaluation/CODE',
                 root + 'metrics_evaluation/CODE/libsvm/python',
                 root])

if __name__ == '__main__':
    nb = 20
    nb_cu = 0
    threadLimiter = threading.BoundedSemaphore(nb)

    datasets = [
        'bike_weekend',
        'lsun', 'Hepta', 'golf_ball',
        'engy_time', 'bike_season',
        'chainlink', 'atom', 'bike_weather',
        'bike_holiday',
        'abalone_rings', 'abalone_sex',
        'auto_cylinders',
        'auto_origin', 'absenteeism_reason',
        'absenteeism_month',
        'absenteeism_day', 'absenteeism_season',
        'absenteeism_kids',
        'absenteeism_alcohol', 'absenteeism_smoking',
        'absenteeism_pet', 'backnote'
    ]

    generate = ['class', 'cluster_removal', 'test_split']
    times = [6, 6, 6]
    tests = ['min_max']
    profiles = {'one_sample_t_test': ['mean', 'same'], 'sign_test': ['median', 'same'], 'min_max': ['min_max', 'same']}


    def execute(dataset, method):
        global threadLimiter
        threadLimiter.acquire()
        global nb_cu
        for _ in range(repeat):
            for test in tests:
                # profiler = profiles.get(test, ['same', 'same'])
                call([sys.executable, "CODE/main.py", "--file1", dataset, "--type", method, "--test_type", test,
                      "--data_type", "Continuous", '--log_name', dataset + test + str(_) + 'read' + "True"])
        nb_cu -= 1

        print("finished")
        threadLimiter.release()


    for dataset in datasets:
        for i in range(len(generate)):
            method = generate[i]
            repeat = times[i]
            if not (dataset == 'golf_ball' and i == 0):
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
