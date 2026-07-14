import boto3
import zipfile

s3 = boto3.client("s3")

bucket = "wellbore-geology-models"
key = "models.zip"

s3.download_file(
    bucket,
    key,
    "models.zip"
)

with zipfile.ZipFile("models.zip") as z:
    z.extractall(".")