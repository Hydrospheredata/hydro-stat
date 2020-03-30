# Metric Evaluation for Domain Drift Detection 
Python Implementation of Metric Evaluation for Domain Drift Detection 
## Contributors 

* Imad Eddine Ibrahim Bekkouch ibekkouch@provectus.com

Provectus

Innopolis 

## Getting Started
Please follow the instructions to get an up and running version of our code running on your local machine.
### Prerequisites
Please make sure you have the following installed.

1. Python 3.6+
2. Scipy 1.3.0
3. scikit-learn 0.21.1
4. ctypes 1.1.0
5. opencv 4.1+


## Problem 
Detecting Domain Drifts between training and deployment data.



## Continuous Tests
### Parametric Statistical Hypothesis Tests (Testing on means)
**1. two_sample_t_test** 

**2. one_sample_t_test** 

**3. ANOVA** 
### Nonparametric Statistical Hypothesis Tests
#### Mean Tests
**1. Mann-Whitney U Test** 

**2. Kruskal-Wallis H Test** 

**3. Kruskal-Wallis H Test:** 

#### Levene Tests for Equality of Variances
**1. Mean-based Levene Test:** 

**2. Median-based Levene Test:** 

**3. Trimmed Levene Test:** 

#### Median Tests
**1. Sign Test:** 

**2. Mood's median test:** 

#### Coverage Tests
**1. Min-Max Test:** 

**1. hull Test:** 

## APIs

Here you will find the way to deploy and use both the rest api and grpc api.

### Rest api
The rest api uses http requests and reponds in json format.

To deploy the rest api, please follow the next steps:

1. Download and the run the metric_eval container which will use the port 5000 for the service.

    ```bash
    docker run -p 5000:5000 imadeddinebek/metric_eval:0.4.1
    ```
2. Send the following http request:
    ```http
    http://0.0.0.0:5000/metrics?model_name=mnist-classifier&model_version=1&training=iris&deployment=iris
    ```
please specify the following values in your http request:

* model_name (ignored for now)
* model_version (ignored for now)
* training
* deployment

(ignored for now): will add an implementation which will take only the model's name and version
 and extract on its own the training and deployment data.


3. The JSON response will contain the following:

* Final Decision
* Test results, every test result contains the following:
    * final_decision: the majority decision from every feature.
    * metric: a list of the metric values for each feature.
    * decision: a list of decision for each feature.
    * p_value: a list of the p-value values for each feature.
    
    
**Example**:
Here is an example of the resulting json string for the iris dataset:
```json
{
  "anova": {
    "decision": [
      "there is no change",
      "there is no change",
      "there is no change",
      "there is no change"
    ],
    "final_decision": "there is no change",
    "metric": [
      0.0,
      7.562131837039091e-14,
      0.0,
      0.0
    ],
    "p_value": [
      1.0,
      0.9999997956196081,
      1.0,
      1.0
    ],
    "status": "succeeded"
  },
  "final_decision": "there is a change",
  .,
  .,
  .,
  .,
  "two_sample_t_test": {
    "decision": [
      "there is no change",
      "there is no change",
      "there is no change",
      "there is no change"
    ],
    "final_decision": "there is no change",
    "metric": [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "p_value": [
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "status": "succeeded"
  }
}
```

The rest api responds to 3 types of requests:

1. Hello request: basic test to verify that the service is up and running.
    ```http
        http://0.0.0.0:5000/
    ```
    the response will be the following: 
    ```text
    Hi! I am Domain Drift Service
    ```

2. build_info request: returns a set of information about the service.
    ```http
        http://0.0.0.0:5000/buildinfo
    ```
    the response will be the similar to the following json structure: 
    ```json
        {
      "available_routes": [
        "/buildinfo",
        "/",
        "/metrics"
      ],
      "gitCurrentBranch": "metric_evaluation",
      "name": "domain-drift",
      "pythonVersion": "3.7.4 (default, Oct 17 2019, 06:26:55) \n[GCC 6.3.0 20170516]",
      "version": "0.0.1"
         }
    ```
    
3. drift_detection request: This is the main request provided by the service.
    ```http
    http://0.0.0.0:5000/metrics?model_name=<MODEL_NAME>&model_version=<MODEL_VERSION>&training=<S3_FILE>&deployment=<S3_FILE>
    ```
    Please check the example response above.
### gRPC api

The grpc server can be launched by running the `drift_detector_service.py` file.
You will find a detailed client for how to use the grpc sever and access all 
it's information in the `drift_detector_client.py`.

The grpd server responds to 3 types of requests similarly to the REST API:
1. Hello request: basic test to verify that the service is up and running.
    * We defined the following proto messages as the request and response:
        ```proto
              syntax = "proto3";

              message HelloRequest {
              }
                
              message HelloResponse {
                    string message = 1;
              }
         ```
2. build_info request: returns a set of information about the service.
    * We defined the following proto messages as the request and response:
        ```proto
              syntax = "proto3";

              message DeployInfoRequest {
              }
                
              message DeployInfoResponse {
                  repeated string available_routes = 1;
                  string gitCurrentBranch = 2;
                  string name = 3;
                  string pythonVersion = 4;
                  string version = 5;
              }
         ```
         
3. drift_detection request: This is the main request provided by the service.
    * We defined the following proto enums:
        ````proto
            enum decision {
                THERE_IS_NO_CHANGE = 0;
                THERE_IS_A_CHANGE = 1;
            }
            
            enum status {
                FAILED = 0;
                SUCCEEDED = 1;
            }
        ````
    * We defined the following proto messages as the request and response:
        ```proto
              syntax = "proto3";

              message MetricsRequest {
                  string model_name = 1;
                  int32 model_version = 2;
                  string training = 3;
                  string deployment = 4;
              }

                
              message Test {
                  decision final_decision = 1;
                  status test_status = 2;
                  repeated decision decisions = 3;
                  repeated float metrics = 4;
                  repeated float p_values = 5;
                  string name = 6;
              }
                
              message MetricsResponse {
                  repeated Test tests = 1;
                  decision final_decision = 2;
              }
         ```
         
The final service looks like the following:
```proto
syntax = "proto3";

service DriftDetectorService {
    rpc hello (HelloRequest) returns (HelloResponse) {
    };

    rpc deployInfo (DeployInfoRequest) returns (DeployInfoResponse) {
    };

    rpc metrics (MetricsRequest) returns (MetricsResponse) {
    };
}

```

#### Python Interface with the GRPC service

The grpc service provides several methods and here is an example
on how to communicate with the service and extract the values provided.
```python
import logging

import grpc

import drift_pb2
import drift_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        # Creating a link 
        stub = drift_pb2_grpc.DriftDetectorServiceStub(channel)
        
        # Sending a hello message request
        response = stub.hello(drift_pb2.HelloRequest())
        # Retrieving the response message
        print("Domain Drift Detector client received: " + response.message)
        
        # Sending a deployInfo request
        response = stub.deployInfo(drift_pb2.DeployInfoRequest())
        # Retrieving the response message
        print("Domain Drift Detector client received: ", len(response.available_routes))
        print("Domain Drift Detector client received: " + response.available_routes[0])
        print("Domain Drift Detector client received: " + response.available_routes[1])
        print("Domain Drift Detector client received: " + response.available_routes[2])
        print("Domain Drift Detector client received: " + response.gitCurrentBranch)
        print("Domain Drift Detector client received: " + response.name)
        print("Domain Drift Detector client received: " + response.pythonVersion)
        print("Domain Drift Detector client received: " + response.version)
        
        # Sending a domain drift detection request
        response = stub.metrics(drift_pb2.MetricsRequest(model_name='mnist-classifier',
                                                         model_version=1,
                                                         training='iris',
                                                         deployment='iris'))
        # Retrieving the response message
        print("Domain Drift Detector client received: ", len(response.tests))
        print("Domain Drift Detector client received: ", response.final_decision)


if __name__ == '__main__':
    logging.basicConfig()
    for i in range(3):
        run()
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details