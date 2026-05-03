
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Define the absolute path for the project directory for script independence
project_dir = "/content/tourismprjtt" # Assuming this is consistent with the notebook setup

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") # please use your token

repo_id = "verma89/tourismprjtt" # Please create your space and repository

repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path=os.path.join(project_dir, "data"), # Use absolute path
    repo_id=repo_id,
    repo_type=repo_type,
)
