from huggingface_hub import HfApi, create_repo
from huggingface_hub import RepositoryNotFoundError, HfHubHTTPError
import os

r_id="Enoch1359/Tourism_data"
r_type="dataset"

api=HfApi(token=os.getenv("HF_TOKEN"))
try:
  api.repo_info(repo_id=r_id, repo_type=r_type)
  print("Repository already exists, So using it")
except RepositoryNotFoundError:
  print("Repository does not exist, So creating it")
  api.create_repo(repo_id=r_id, repo_type=r_type, private= False)
  print("Repo created")
api.upload_folder(folder_path="/tourism_project/data", repo_id=r_id, repo_type=r_type)
