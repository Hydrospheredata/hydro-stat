from reporter import *

def filter_dataset(all_logs, dataset):
    filtered_logs = []
    for log in all_logs:
        # pprint(log)
        if log['data']['data'] in dataset:
            filtered_logs.append(log)

    return filtered_logs


def filter_type(all_logs, file):
    filtered_logs = []
    for log in all_logs:
        # pprint(log)
        if log['data']['type'] in file:
            filtered_logs.append(log)

    return filtered_logs


if __name__ == '__main__':
    logs = parse_logs('../outputs/Continuous/merge.log')

    file = 'iris'
    logs = successful_logs(logs)
    # pprint(logs[0])

    logs = filter_dataset(logs, file)
    file = 'class'
    logs = filter_type(logs, file)
    pprint(len(logs))

    # accuracy_bar_plot(logs, file + '.png')
