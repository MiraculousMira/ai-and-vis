import os
import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")

#print("Path to dataset files:", path)

shutil.copytree(path, "data", dirs_exist_ok=True)

if os.path.exists(path):
    shutil.rmtree(path)