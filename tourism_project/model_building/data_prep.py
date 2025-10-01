import pandas as pd
import sklearn
from huggingface_hub import HfApi, login
from sklearn.model_selection import train_test_split
import os
api=HfApi(token=os.getenv("HF_TOKEN"))
Dataset_path="hf://datasets/Enoch1359/Tourism_data/tourism.csv"
data=pd.read_csv(Dataset_path)
print("Dataset loaded success")
target='ProdTaken'
ncols=[ 'Age', 'CityTier',
       'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
       'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
       'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
       'MonthlyIncome']
cat_cols=['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
       'MaritalStatus', 'Designation']
x=data[ncols+cat_cols]
y=data[target]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
x_train.to_csv("x_train.csv", index=False)
x_test.to_csv("x_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
files=["x_train.csv","x_test.csv","y_train.csv","y_test.csv"]
for f in files:
  api.upload_file(path_or_fileobj=f,
                  path_in_repo=f.split('/')[-1],
                  repo_id="Enoch1359/Tourism_data",
                  repo_type="dataset")
