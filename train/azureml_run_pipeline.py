import argparse

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute

parser = argparse.ArgumentParser("train")
parser.add_argument("--subscription_id", type=str)
parser.add_argument("--resource_group_name", type=str)
parser.add_argument("--workspace_name", type=str)
parser.add_argument("--location", type=str)

args = parser.parse_args()
subscription_id = args.subscription_id
resource_group_name = args.resource_group_name
workspace_name = args.workspace_name
location = args.location

URI_FOLDER = "uri_folder"

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print(ex)
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

# Get a handle to workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
)

# Retrieve an already attached Azure Machine Learning Compute.
cluster_name = "simple-cpu-low"

cluster_basic = AmlCompute(
    name=cluster_name,
    type="amlcompute",
    size="Standard_D4s_v3",
    location=location, #az account list-locations -o table
    min_instances=0,
    max_instances=1,
    idle_time_before_scale_down=60,
)
ml_client.begin_create_or_update(cluster_basic).result()


@pipeline(default_compute=cluster_name)
def azureml_pipeline(pdfs_input_data: Input(type=URI_FOLDER)):
    extraction_step = load_component(source="extraction/command.yaml")
    extraction = extraction_step(
        pdfs_input=pdfs_input_data
    )

    return {}


pipeline_job = azureml_pipeline(
    pdfs_input_data=Input(
        path="azureml:cats_dogs_others:1", type=URI_FOLDER
    )
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="cats_dos_others_pipeline"
)

ml_client.jobs.stream(pipeline_job.name)