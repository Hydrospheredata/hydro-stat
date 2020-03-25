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
    times = [2, 2, 2]
    tests = ['two_sample_t_test', 'one_sample_t_test', 'anova', 'mann', 'kruskal', 'brunner',
             'levene_mean', 'levene_median', 'levene_trimmed', 'sign_test', 'median_test',
             'min_max', 'ks', 'kupier', 'a_dist']
    profiles = {'one_sample_t_test': ['mean', 'same'], 'sign_test': ['median', 'same'], 'min_max': ['min_max', 'same']}
    percentage = [20, 40, 60, 80]


    def execute(dataset, method, repeat):
        global threadLimiter
        threadLimiter.acquire()
        global nb_cu
        for _ in range(repeat):
            for test in tests:
                # profiler = profiles.get(test, ['same', 'same'])
                for p_t in percentage:
                    for p_d in percentage:
                        call(
                            [sys.executable, "CODE/best_length.py", "--file1", dataset, "--type", method, "--test_type",
                             test,
                             "--data_type", "Continuous", '--log_name', dataset + test + method, '--read', "True",
                             '--reduce_training_ratio', str(p_t), '--reduce_deployment_ratio', str(p_d)])
        nb_cu -= 1

        print("finished")
        threadLimiter.release()


    for dataset in datasets:
        for i in range(len(generate)):
            method = generate[i]
            repeat = times[i]
            if not (dataset == 'golf_ball' and i == 0):
                nb_cu += 1
                thread1 = threading.Thread(target=execute, args=(dataset, method, repeat))
                thread1.start()
                while nb_cu >= nb:
                    print(nb_cu)
                    sleep(50)
