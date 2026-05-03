# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Set your token first

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Define the absolute path for the project directory for script independence
project_dir = "/content/tourismprjtt" # Assuming this is consistent with the notebook setup

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/verma89/tourismprjtt/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the directory to save the processed data
output_dir = os.path.join(project_dir, "model_building")
os.makedirs(output_dir, exist_ok=True)

Xtrain.to_csv(os.path.join(output_dir, "Xtrain.csv"),index=False)
Xtest.to_csv(os.path.join(output_dir, "Xtest.csv"),index=False)
ytrain.to_csv(os.path.join(output_dir, "ytrain.csv"),index=False)
ytest.to_csv(os.path.join(output_dir, "ytest.csv"),index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_name in files:
    local_file_path = os.path.join(output_dir, file_name)
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=file_name,  # just the filename in the repo
        repo_id="verma89/tourismprjtt",
        repo_type="dataset",
    )
