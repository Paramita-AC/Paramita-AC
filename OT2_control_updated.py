import boto3
import os

def download_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                relative_path = os.path.relpath(s3_key, s3_folder)
                local_file_path = os.path.join(local_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file_path)
                print(f"Downloaded: {s3_key} to {local_file_path}")

if __name__ == "__main__":
    bucket_name = "acceleration-research-s3-bucket"
    s3_folder = "SDL1/Input/20250326_014/"
    local_directory = "./local_data"
    
    download_s3_folder(bucket_name, s3_folder, local_directory)
