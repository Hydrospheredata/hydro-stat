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
    datasets = ['iris', 'wine',
                'wing_nut', 'two_diamonds', 'tetra',
                'target', 'bike_weather', 'bike_weekend',
                'lsun', 'Hepta', 'golf_ball',
                'engy_time', 'bike_season',
                'chainlink', 'atom', 'bike_weather',
                'bike_holiday',
                'bike_weekend', 'bike_season',
                'abalone_rings', 'abalone_sex',
                'auto_cylinders',
                'auto_origin', 'absenteeism_reason',
                'absenteeism_month',
                'absenteeism_day', 'absenteeism_season',
                'absenteeism_kids',
                'absenteeism_alcohol', 'absenteeism_smoking',
                'absenteeism_pet',
                'backnote']

    generate = ['class', 'cluster_removal', 'test_split']
    times = [1, 1, 2]

    profiles = {'one_sample_t_test': ['mean', 'same'], 'sign_test': ['median', 'same'], 'min_max': ['min_max', 'same'],
                'hull': ['delaunay', 'same']}
    for dataset in datasets:
        for i in range(len(generate)):
            method = generate[i]
            repeat = times[i]
            if not (dataset == 'golf_ball' and i == 0):
                for _ in range(repeat):
                    call(["python3", "reduce_dim.py", "--file1", dataset, "--type", method])
