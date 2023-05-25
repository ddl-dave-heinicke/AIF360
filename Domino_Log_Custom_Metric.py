# Import necessary packages

import sys
import os
sys.path.insert(1, "../")  

import numpy as np
import pandas as pd
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

import domino

project = "{}/{}".format(os.environ['DOMINO_PROJECT_OWNER'],
                         os.environ['DOMINO_PROJECT_NAME'])

print("Project name: {}".format(project))

# Registered model for monitoring 
dmm_model_id = "644ffac7a9872366aac7065d"

# Initiate custom metric client
d = domino.Domino(project)
metrics_client = d.custom_metrics_client()

# Load dataset from AIF360
dataset_orig = GermanDataset(
    protected_attribute_names=['age'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
)


dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

# Define privileged groups
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

original_mean_difference = abs(metric_orig_train.mean_difference())

RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)

metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" %metric_transf_train.mean_difference())

corrected_mean_difference = abs(metric_orig_train.mean_difference())

metrics_client.log_metrics([
    { "modelMonitoringId" : dmm_model_id, "metric" : "Age_Mean_Difference", "value" : corrected_mean_difference,
    "timestamp" : "2023-05-08T00:00:00Z",
    "tags" : { "example_tag1" : "value1", "example_tag2" : "value2" }
    },
    { "modelMonitoringId" : dmm_model_id, "metric" : "Age_Mean_Difference", "value" : original_mean_difference,
    "timestamp" : "2023-05-08T00:00:10Z" }
    ])

metrics_client.trigger_alert(dmm_model_id, "Age_Mean_Difference", 3.14,
                            condition = metrics_client.GREATER_THAN,
                            lower_limit=0.1, 
                            upper_limit=999,
                            description = "AIF 360 has detected Age Difference Factor Greater than 0.1" )

res = metrics_client.read_metrics(dmm_model_id, "Age_Mean_Difference",
"2023-05-08T00:00:00Z", "2023-05-08T00:00:10Z")

res_df = pd.DataFrame.from_dict(res['metricValues'])

print(res_df.head())