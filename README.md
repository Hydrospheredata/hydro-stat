# Metric Evaluation for Domain Drift Detection 
Python Implementation of Metric Evaluation for Domain Drift Detection 
## Contributors 

* Imad Eddine Ibrahim Bekkouch ibekkouch@provectus.com

Provectus

Innopolis 

## Getting Started
Please follow the instructions to get an up and running version of our code running on your local machine.

## Problem 
Detecting Domain Drifts between training and deployment data.


## APIs

Here you will find the way to deploy and use both the rest api.

### Rest api
The rest api uses http requests and reponds in json format.

To deploy the rest api, please follow the next steps:

1. Download and the run the metric_eval container which will use the port 5000 for the service.

    ```bash
    docker build -t provectus/hydro_stat:0.0.1 .
    docker run -p 5000:5000 provectus/hydro_stat:0.0.1
    ```
2. Send the following http request:
    ```http
    http://0.0.0.0:5000/metrics?model_name=adult_classification&model_version=1
    ```
please specify the following values in your http request:

* model_name
* model_version 

(ignored for now): will add an implementation which will take only the model's name and version
 and extract on its own the training and deployment data.


3. The JSON response will contain the following:

* overall_probability_drift
* per_feature_report:
    * drift-probability.
    * histogram:
        * bins
        * training
        * deployment
    * statistics: a list of statistics for each feature.
        * change_probability
        * deployment
        * training
*warnings: 
    * final_decision
    * report: 
        * drift_probability_per_feature
        * message
    
    
**Example**:
Here is an example of the resulting json string to show the difference between continuous and discrete feature responses:
```json
{
  "overall_probability_drift": 1.0,
  "per_feature_report": {
    "Class": {
      "drift-probability": 1.0,
      "histogram": {
        "bins": [
          10101500.0,
          14353520.0,
          18605540.0,
          22857560.0,
          27109580.0,
          31361600.0,
          35613620.0,
          39865640.0,
          44117660.0,
          48369680.0,
          52621700.0,
          56873720.0,
          61125740.0,
          65377760.0,
          69629780.0,
          73881800.0,
          78133820.0,
          82385840.0,
          86637860.0,
          90889880.0,
          95141900.0
        ],
        "deployment": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.968305977835799e-07,
          4.617996712812843e-09,
          8.545267421572484e-09,
          6.973043469489479e-10,
          0.0,
          9.551753884621436e-09,
          1.7366825244766249e-09,
          4.848238714163912e-09,
          3.058929446521328e-09,
          5.196890887638385e-10,
          4.775876942310718e-09
        ],
        "training": [
          5.787626079676267e-08,
          6.052075464085208e-10,
          5.940243634857547e-09,
          1.60379999798258e-08,
          2.5063486432787653e-08,
          4.696936827561781e-09,
          3.9272707087596405e-09,
          4.237768493438795e-08,
          7.808493017249066e-09,
          7.084875298717141e-08,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      },
      "statistics": {
        "mean": {
          "change_probability": 1.0,
          "deployment": 54260379.42714889,
          "training": 33558627.974042684
        },
        "median": {
          "change_probability": 1.0,
          "deployment": 50591900.0,
          "training": 40171900.0
        },
        "std": {
          "change_probability": 1.0,
          "deployment": 9453600.881636694,
          "training": 15655558.710062902
        }
      }
    },
    "Class Name": {
      "drift-probability": 0.0,
      "histogram": {
        "bins": [
          "Table grape purees",
          "Live animals",
          "Fresh fruit purees",
          "Food Beverage and Tobacco Products",
          "Livestock",
          "Regina grape purees",
          "Mink",
          "Live Plant and Animal Material and Accessories and Supplies"
        ],
        "deployment": [
          1,
          0,
          1,
          1,
          0,
          1,
          0,
          0
        ],
        "training": [
          0,
          1,
          0,
          0,
          1,
          0,
          1,
          1
        ]
      },
      "statistics": {
        "entropy": {
          "change_probability": 0.0,
          "deployment": 2.0,
          "training": 2.0
        },
        "unique values": {
          "change_probability": 0.0,
          "deployment": 4,
          "training": 4
        }
      }
    },
  },
  "warnings": {
    "final_decision": "there is a change",
    "report": [
      {
        "drift_probability_per_feature": 1.0,
        "message": "the feature \"Segment\" has changed."
      },
      {
        "drift_probability_per_feature": 1.0,
        "message": "the feature \"Family\" has changed."
      },
      {
        "drift_probability_per_feature": 1.0,
        "message": "the feature \"Class\" has changed."
      },
      {
        "drift_probability_per_feature": 1.0,
        "message": "the feature \"Commodity\" has changed."
      }
    ]
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details