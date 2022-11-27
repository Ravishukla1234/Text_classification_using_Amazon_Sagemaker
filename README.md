
# StumbleUpon Text Classification

In this project, I have tried to solve [StumbleUpon]("https://www.kaggle.com/competitions/stumbleupon") Kaggle Problem.
 In `notebooks/01-stumbleupon-art-of-eda.ipynb` which is very similar to [this kaggle notebook]("https://www.kaggle.com/code/raviista/stumbleupon-art-of-eda")
  I have done some cool EDA followed by model training and ensembling to achieve a test AUC of 88.44% on kaggle private leaderboard.
  Then in `notebooks/02-DataPrep-and-training.ipynb`, I have used Amazon Sagemaker to train and deploy a `bert-base-uncased` model on stumbleupon dataset from kaggle. Link to the dataset:- https://www.kaggle.com/competitions/stumbleupon/data.
   Then in the notebook `03-Create-Lambda.pynb` I have showed how to create AWS Lambda required to host the Sagemaker Endpoint via API Gateway .

Pytorch and hugging face is used for the modeling purpose.

## Follow the steps as below.
1. Create a Sagemaker notebook instance with the instance type as `ml.t2.medium`
2. Once the Notebook instance is `In Service`, clone this git repo in the Jupyter environment
3. Run `notebooks/02-DataPrep-and-training.ipynb` notebook to train and deploy the model with Amazon Sagemaker followed by Inference
4. Refer to `src/train.py` script used for training the model
5. Run notebook `notebooks/03-Create-Lambda` to create AWS Lambda required to host the Sagemaker Endpoint via API Gateway
6. Follow this detailed [AWS tutorial]("https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/") to invoke lambda function via Amazon API gateway
7. Download `flask-api` folder in your local and run `flask-api/app.py` and change the variable `url = "<<Amazon API Gateway url link>>"` with your Amazon API Gateway url link  to create a flask API.
8. (Optional) You can follow this [medium article]("https://medium.com/techfront/step-by-step-visual-guide-on-deploying-a-flask-application-on-aws-ec2-8e3e8b82c4f7") to run your Flask API on AWS EC2 instance.


## Demo

<img src="demo_stumbleUpon.gif" width="900" height="400" />

## Authors

- [@Ravishukla1234](https://www.github.com/Ravishukla1234)

