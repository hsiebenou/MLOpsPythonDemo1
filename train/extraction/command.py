import argparse
from extraction import DataManager, ExtractImages
import mlflow

parser = argparse.ArgumentParser("extraction")
parser.add_argument("--pdfs_input", type=str)
parser.add_argument("--images_output", type=str)

args = parser.parse_args()
pdfs_input = args.pdfs_input
images_output = args.images_output

extract_images = ExtractImages(DataManager())
result = extract_images.extract_images(pdfs_input, images_output)

mlflow.log_metric("number_files_input", result.number_files_input)
mlflow.log_metric("number_images_output", result.number_images_output)