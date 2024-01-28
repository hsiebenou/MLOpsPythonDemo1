import argparse
import json
import uuid

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute, Environment, BuildContext
from azure.ai.ml.constants import AssetTypes

from extraction import register_extracted_dataset

parser = argparse.ArgumentParser("train")
parser.add_argument("--subscription_id", type=str)
parser.add_argument("--resource_group_name", type=str)
parser.add_argument("--workspace_name", type=str)
parser.add_argument("--location", type=str)
parser.add_argument("--tags", type=str, default="{}")

args = parser.parse_args()
subscription_id = args.subscription_id
resource_group_name = args.resource_group_name
workspace_name = args.workspace_name
location = args.location
tags = json.loads(args.tags)


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
def azureml_pipeline(pdfs_input_data: Input(type=AssetTypes.URI_FOLDER)):
    extraction_step = load_component(source="extraction/command.yaml")
    extraction = extraction_step(
        pdfs_input=pdfs_input_data
    )

    output_step = load_component(source="output/command.yaml")
    output = output_step(
        extraction_hash_input=extraction.outputs.hash_output,
        extraction_images_input=extraction.outputs.images_output,
    )

    return {
        "output": output.outputs.main_output,
    }


pipeline_job = azureml_pipeline(
    pdfs_input_data=Input(
        path="azureml:cats_dogs_others:1", type=AssetTypes.URI_FOLDER
    )
)

azure_blob = "azureml://datastores/workspaceblobstore/paths/"
experiment_id = str(uuid.uuid4())
custom_output_path = (
    azure_blob + "extraction/cats-dogs-others/" + experiment_id + "/"
)
pipeline_job.outputs.output = Output(
    type=AssetTypes.URI_FOLDER, mode="rw_mount", path=custom_output_path
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="cats_dos_others_pipeline",
    tags=tags
)

ml_client.jobs.stream(pipeline_job.name)

register_extracted_dataset(
    ml_client, custom_output_path, tags
)