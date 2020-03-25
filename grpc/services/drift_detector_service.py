"""The Python implementation of the GRPC helloworld.Greeter server."""
import os
import sys
from concurrent import futures
import logging

import grpc
from multiprocessing.pool import ThreadPool
import dataloader
import profiler
from metric_tests import continuous_stats
import drift_pb2
import drift_pb2_grpc
from loguru import logger

from utils.utils import fix_path

os.chdir('../../')

with open("version") as version_file:
    VERSION = version_file.read().strip()

tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                     'min_max': ('min_max', 'same'),
                     'hull': ('delaunay', 'same')}


class DriftDetector(drift_pb2_grpc.DriftDetectorServiceServicer):

    def hello(self, request, context):
        return drift_pb2.HelloResponse(message='Hi! I am Domain Drift Service.')

    def deployInfo(self, request, context):
        branch_name = "metric_evaluation"  # check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf8").strip()
        py_version = sys.version
        return drift_pb2.DeployInfoResponse(available_routes=["/buildinfo", "/", "/metrics"],
                                            version=VERSION,
                                            pythonVersion=py_version,
                                            name="domain-drift",
                                            gitCurrentBranch=branch_name)

    def metrics(self, request, context):
        tests = ['two_sample_t_test', 'one_sample_t_test', 'anova', 'mann', 'kruskal',
                 'levene_mean', 'levene_median', 'levene_trimmed',
                 'sign_test', 'median_test',
                 'min_max', 'ks']
        full_report = {}
        d1, d2 = dataloader.read_datasets(request.training, request.deployment)
        logger.info(d1.shape)
        logger.info(d2.shape)
        pool = ThreadPool(processes=1)
        async_results = {}
        for test in tests:
            async_results[test] = pool.apply_async(self.one_test, (d1, d2, test))
        for test in tests:
            full_report[test] = async_results[test].get()
        # print(type(full_report))
        full_report_ = self.final_decision(full_report)
        final_decision = self.decision_from_report(full_report_)
        del full_report_['final_decision']
        tests = [self.test_from_report(name, results) for name, results in full_report_.items()]
        return drift_pb2.MetricsResponse(
            tests=tests,
            final_decision=final_decision)

    def one_test(self, d1, d2, name):
        logger.info(name)
        stats_type1, stats_type2 = tests_to_profiles.get(name, ['same', 'same'])
        s1 = profiler.get_statistic(d1, stats_type1, None, 1)
        logger.info(name)
        s2 = profiler.get_statistic(d2, stats_type2, None, 2)
        logger.info(name)
        report = continuous_stats.test(s1, s2, name, None)
        logger.info(name)

        # pprint(report)
        return report

    def final_decision(self, full_report):
        count_pos = 0
        count_neg = 0
        # pprint(full_report)
        for key, log in full_report.items():
            # pprint(log)
            if log['status'] == 'succeeded':
                if list(log['decision']).count("there is no change") > len(log['decision']) // 2:
                    log['final_decision'] = 'there is no change'
                    count_pos += 1
                else:
                    log['final_decision'] = 'there is a change'
                    count_neg += 1
        if count_pos > count_neg:
            full_report['final_decision'] = 'there is a change'
        else:
            full_report['final_decision'] = 'there is no change'
        return full_report

    def decision_from_report(self, full_report_):
        if full_report_['final_decision'] == 'there is no change':
            return drift_pb2.decision.THERE_IS_NO_CHANGE
        else:
            return drift_pb2.decision.THERE_IS_A_CHANGE

    def test_from_report(self, name, results):
        print(results)
        p_values = results.get('p_value', [-1])
        metrics = results['metric']
        status = drift_pb2.status.SUCCEEDED if results['status'] == 'succeeded' else drift_pb2.status.FAILED
        final_decision = drift_pb2.decision.THERE_IS_NO_CHANGE if results[
                                                                      'final_decision'] == 'there is no change' else drift_pb2.decision.THERE_IS_A_CHANGE
        decisions = [self.decision_from_string(decision) for decision in results['decision']]
        return drift_pb2.Test(name=name,
                              p_values=p_values,
                              metrics=metrics,
                              decisions=decisions,
                              test_status=status,
                              final_decision=final_decision)

    def decision_from_string(self, decision):
        if decision == 'there is no change':
            return drift_pb2.decision.THERE_IS_NO_CHANGE
        else:
            return drift_pb2.decision.THERE_IS_A_CHANGE


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    drift_pb2_grpc.add_DriftDetectorServiceServicer_to_server(DriftDetector(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    fix_path()
    logging.basicConfig()
    serve()
