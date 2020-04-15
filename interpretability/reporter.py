import logging
import json
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


root = path.replace('metrics_evaluation', '')
sys.path.extend([root, root + 'metrics_evaluation/CODE',
                 root + 'metrics_evaluation/CODE/libsvm/python',
                 root])


def save_report(config, report):
    logger = logging.getLogger('report-logger')
    message = 'data_type:{}, stats_type1:{}, test_type:{}, read:{}, type:{}, file1:{}, file2:{}, report:{}' \
        .format(config.data_type, config.stats_type1, config.test_type, config.read, config.type, config.file1,
                config.file2, report)
    logger.info(message)


def parse_logs(file):
    # match_list = []
    # regex = ''
    all_logs = []
    with open(file, "r") as file:
        line = file.readline()
        while line:
            splits = line.split(' -- ')
            logger = splits[0]
            one_log = {'status': 'Success', 'data': {
                'data': None,
                'type': 'cluster',
                'removed_label': 0.,
                'classes': [],
                'random_state': None,
                'len_d1': 0,
                'len_d2': 0,
            }, 'graph': {}, 'results': {
                'test': None,
                'p_value': [],
                'metric': [],
                'decision': [],
                'other': {}
            }}
            if logger == 'fake_data_logger':
                time = splits[1]
                message = splits[2]
                if 'len_d2' not in message:
                    message += file.readline()
                message = message.split(', ')
                if message[0].split(':')[0] == 'cluster':
                    cluster = int(message[0].split(':')[1])
                    random_state = int(message[1].split(':')[1])
                    len_d1 = int(message[2].split(':')[1])
                    len_d2 = int(message[3].split(':')[1])
                    one_log['data']['type'] = 'cluster'
                    one_log['data']['removed_label'] = cluster
                    one_log['data']['random_state'] = random_state
                    one_log['data']['len_d1'] = len_d1
                    one_log['data']['len_d2'] = len_d2

                elif message[0].split(':')[0] == 'removed_class':
                    removed_class = int(message[0].split(':')[1])
                    classes = message[1].split(':')[1].strip('][').replace("'", '').split(' ')
                    len_d1 = int(message[2].split(':')[1])
                    len_d2 = int(message[3].split(':')[1])
                    one_log['data']['type'] = 'class'
                    one_log['data']['removed_label'] = removed_class
                    one_log['data']['classes'] = classes
                    one_log['data']['len_d1'] = len_d1
                    one_log['data']['len_d2'] = len_d2
                else:
                    test_size = float(message[0].split(':')[1])
                    random_state = int(message[1].split(':')[1])
                    len_d1 = int(message[2].split(':')[1])
                    len_d2 = int(message[3].split(':')[1])
                    one_log['data']['type'] = 'test_split'
                    one_log['data']['removed_label'] = test_size
                    one_log['data']['random_state'] = random_state
                    one_log['data']['len_d1'] = len_d1
                    one_log['data']['len_d2'] = len_d2

                line = file.readline().replace('\n', ' ')
                splits = line.split(' -- ')
                logger = splits[0]
                if logger == 'report-logger':
                    try:
                        while line[-2] != '}':
                            line += file.readline()
                        line = line.replace('\n', ' ').replace('), array(', ', ').replace('array(', '').replace('])]',
                                                                                                                ']]')
                        splits = line.split(' -- ')
                        time = splits[1]
                        message = splits[2]
                        message = message.split(', ', 7)
                        test = message[2].split(':')[1]
                        data = message[5].split(':')[1]
                        p_value = message[-1].split(':')[1]
                        # logger.info(message[-1].replace('report:', '').replace("'", "\""))
                        message = json.loads(message[-1].replace('report:', '').replace("'", "\""))
                        one_log['data']['data'] = data
                        one_log['results']['test'] = test
                        # logger.info(message.get('metric', []), message.get('p_value', []))
                        one_log['results']['metric'] = message.get('metric', [])
                        one_log['results']['metric'] = message.get('metric', [])
                        one_log['results']['decision'] = message.get('decision', [])
                    except:
                        one_log['status'] = 'failed'
                else:
                    one_log['status'] = 'failed'

            else:
                one_log['status'] = 'failed'
            all_logs.append(one_log)
            line = file.readline()
        return all_logs


def filter(all_logs, test):
    filtered_logs = []
    for log in all_logs:
        # plogger.info(log)
        if log['results']['test'] in test:
            filtered_logs.append(log)

    return filtered_logs


def accuracy_bar_plot(logs, file):
    labels = []
    labels_maj = []
    labels_one = []
    labels_all = []
    for log in logs:
        # plogger.info(log)
        if log['data']['type'] in ('class', 'cluster'):
            labels.append(1)
        else:
            labels.append(0)
        decisions = log['results']['decision']
        counts = decisions.count('there is a change')
        if counts > 0:
            labels_one.append(1)
        else:
            labels_one.append(0)

        if counts == len(decisions):
            labels_all.append(1)
        else:
            labels_all.append(0)

        if counts >= len(decisions) // 2:
            labels_maj.append(1)
        else:
            labels_maj.append(0)
    # logger.info(len(labels))
    # logger.info(len(logs))
    logger.info(accuracy_score(labels, labels_all))
    logger.info(accuracy_score(labels, labels_maj))
    logger.info(accuracy_score(labels, labels_one))

    eval_funcs = [accuracy_score, precision_score, recall_score, f1_score]

    all_res = [ev(labels, labels_all) for ev in eval_funcs]
    maj_res = [ev(labels, labels_maj) for ev in eval_funcs]
    one_res = [ev(labels, labels_one) for ev in eval_funcs]
    plot(all_res, maj_res, one_res)


def successful_logs(logs):
    result = []
    for log in logs:
        if log['status'] == 'Success':
            # logger.info(log['results']['metric'])
            for met in log['results']['metric']:
                if np.isnan(met):
                    logger.info(met)
                if np.float(met) == 'nan':
                    float(met)
                    continue
                # logger.info(str(met))
            result.append(log)
    return result


def plot(all_res, maj_res, one_res):
    labels = ['accuracy', 'precision', 'recall', 'f1-score']

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1.5 * width, all_res, width, label='all')
    rects2 = ax.bar(x, maj_res, width, label='majority')
    rects3 = ax.bar(x + 1.5 * width, one_res, width, label='at-least one')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('results')
    ax.set_title('Results by combination ' + file.replace('.png', ''))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 4, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    # plt.savefig(file)

    plt.show()


def plot_density(test, clas, cluster, train=False):
    logger.info(test,clas)
    sns.distplot(test, hist=True, kde=True,
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4}, label='no DD')

    sns.distplot(clas, hist=True, kde=True,
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4}, label='DD')

    # sns.distplot(cluster, hist=True, kde=True,
    #              hist_kws={'edgecolor': 'black'},
    #              kde_kws={'linewidth': 4}, label='cluster_removal')
    if train:
        plt.title('Train dist of losses of inliers and outliers')
    else:
        plt.title('distribution of mean metric values for '+ file)
    plt.xlabel('metric value')
    plt.ylabel('Probability density')
    plt.legend()
    # experiment.log_figure(figure=plt)
    plt.show()


def split(logs):
    test = []
    clas = []
    cluster = []
    for log in logs:
        # plogger.info(log)
        if log['data']['type'] =='test_split':
            if not np.isnan(np.mean(log['results']['metric']) ):
                test.append(np.mean(log['results']['metric']))
            # for metric in log['results']['metric']:
            #     test.append(metric)
        # elif log['data']['type'] =='cluster':
        #     cluster.append(np.mean(log['results']['metric']))
        #     # for metric in log['results']['metric']:
        #     #     cluster.append(metric)
        elif log['data']['type'] =='class':
            if not np.isnan(np.mean(log['results']['metric'])):
                clas.append(np.mean(log['results']['metric']))
            # for metric in log['results']['metric']:
            #     clas.append(metric)

    return np.array(test), np.array(clas), np.array(cluster)


if __name__ == '__main__':
    logs = parse_logs('/media/6TB/public/imad/Research/hydro-research/metrics_evaluation/outputs/Continuous/merge.log')

    file = 'a_dist'
    logs = successful_logs(logs)
    # plogger.info(logs)
    logs = filter(logs, file)
    # plogger.info(logs)
    test, clas, cluster = split(logs)
    plot_density(test, clas, None)
    # plogger.info(logs)
    # accuracy_bar_plot(logs, file + '.png')
