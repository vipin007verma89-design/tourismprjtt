from huggingface_hub import HfApi
import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") # It will grab the token from your environment
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourismprjtt/deployment",     # the local folder containing your files
    repo_id="verma89/tourismprjtt",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
