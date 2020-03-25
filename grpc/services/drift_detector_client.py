"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function
import logging

import grpc

import drift_pb2
import drift_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = drift_pb2_grpc.DriftDetectorServiceStub(channel)
        response = stub.hello(drift_pb2.HelloRequest())
        print("Domain Drift Detector client received: " + response.message)
        response = stub.deployInfo(drift_pb2.DeployInfoRequest())
        print("Domain Drift Detector client received: ", len(response.available_routes))
        print("Domain Drift Detector client received: " + response.available_routes[0])
        print("Domain Drift Detector client received: " + response.available_routes[1])
        print("Domain Drift Detector client received: " + response.available_routes[2])
        print("Domain Drift Detector client received: " + response.gitCurrentBranch)
        print("Domain Drift Detector client received: " + response.name)
        print("Domain Drift Detector client received: " + response.pythonVersion)
        print("Domain Drift Detector client received: " + response.version)
        response = stub.metrics(drift_pb2.MetricsRequest(model_name='mnist-classifier',
                                                         model_version=1,
                                                         training='iris',
                                                         deployment='iris'))
        print("Domain Drift Detector client received: ", len(response.tests))
        print("Domain Drift Detector client received: ", response.final_decision)


if __name__ == '__main__':
    logging.basicConfig()
    for i in range(3):
        run()
