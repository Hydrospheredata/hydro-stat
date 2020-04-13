# Metric Evaluation for Domain Drift Detection 
Python Implementation of Metric Evaluation for Domain Drift Detection 
## Contributors 

* Imad Eddine Ibrahim Bekkouch ibekkouch@provectus.com

Provectus

Innopolis 

## Environment variables to configure service while deploying
Addresses to other services:
* `HTTP_UI_ADDRESS` - http address of hydro-serving cluster, used to create `hydrosdk.Cluster(HS_CLUSTER_ADDRESS)`


AWS/Minio parameters:
* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `S3_ENDOPOINT`, if no s3 endpoint is present, data will be downloaded with boto3

Flask server parameters:
* `APPLICATION_ROOT` - prefix of all routes specified in [hydro_auto_od_openapi.yaml](hydro-auto-od-openapi.yaml), not used right now
* `DEBUG`