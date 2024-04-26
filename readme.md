# Create Amazon Product Review Sentiment Analysis / Create Flask Application


## 1) Create Amazon Product Review Sentiment Analysis


### NOTE:   
#### Download the Glove Embedding from Kaggle  here : https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt


## Project Overview

This repository contains a machine learning project focused on performing sentiment analysis on Amazon product reviews. The aim is to classify the sentiments expressed in the reviews as either positive or negative. This can assist businesses in automating the analysis of customer feedback and enhance user experience through improved product recommendations.

## Problem Setup

The task is framed as a binary classification problem:
- **Input**: Text of a product review.
- **Output**: Sentiment classification (Positive or Negative).

## Data Used

The dataset comprises reviews collected from Amazon, each labeled as either positive or negative based on the sentiment expressed by the reviewer. The dataset includes:
- **Review Text**: The text content of the review.
- **Sentiment Label**: Binary labels (Positive or Negative).

### Sample Data Structure
```python
{
  "reviewText": "Great product, loved it!",
  "sentiment": "Positive"
}
 ```

 ## Technologies and Techniques Used
- **Python**: For all backend operations including data handling and model training.
- **TensorFlow/Keras**: To build and train the neural network.
- **Natural Language Processing**: Techniques such as tokenization and vectorization to convert text data into a form that can be fed into the model.


## Installation and Setup
To get started with this project, you need to install the required Python packages. Here is a step-by-step guide to set up your environment:

```bash
# Clone the repository
git clone https://github.com/your-repository/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis

```
To run this project, you need to install the required libraries. Here is how you can install the necessary packages:
```bash
pip install numpy pandas tensorflow sklearn pickle matplotlib

```

## How to Run the Notebook
```bash
# Activate your virtual environment (if applicable)
source your-env/bin/activate

# Start Jupyter Notebook
jupyter notebook

# Open the notebook named 'Amazon_Sentiment_Review.ipynb' and run the cells
```

## Model Architecture
The neural network model used in this project includes the following layers:
- **Embedding Layer**: Converts the input data into dense vectors of fixed size.
- **LSTM Layer**: A type of recurrent neural network layer that helps the model understand the context in the text.
- **Dense Layer**: Outputs the prediction probability of the review being positive.

## Training and Evaluation
The model is trained using a set of reviews with known sentiments. The training process involves adjusting the model weights to minimize the loss over several epochs. The model's performance is then evaluated using a separate test set to ensure it generalizes well to new data.

## Results
The results section includes a discussion on the model's accuracy and any metrics used to evaluate its performance. It also includes visualizations of the training process, such as loss and accuracy curves.

## Future Work
Future enhancements could include:
- Expanding the dataset to include more diverse reviews.
- Experimenting with different model architectures.
- Implementing additional features such as sentiment intensity scoring.

## Contributor
 - Hasan Sattar 



# 2)  Create Flask Application/Deploy 

### Overview
This Flask application performs sentiment analysis on user-submitted reviews. It predicts whether the sentiment of the review is positive or negative using a pre-trained machine learning model and provides real-time feedback to the user.


## Technologies and Techniques
- **Flask**: A web framework used to create the web interface for submitting reviews and displaying sentiment analysis.
- **TensorFlow/Keras**: For building, training, and loading the sentiment analysis model.
- **Pickle**: For loading pre-processed tokenizer data necessary for text data preparation.
- **HTML/CSS**: For creating and styling the web interface.

## Installation and Setup
Ensure you have Python installed, then follow these steps to set up the project:

```bash
# Clone the repository
git clone https://github.com/your-repository/flask-sentiment-analysis.git


# Install required packages
pip install -r requirements.txt

```


## How to Run the Application

```bash
python app.py

```

Navigate to http://127.0.0.1:5000/ in your web browser to use the application.

## Application Structure
- **app.py**: Contains the Flask application setup, routes, and model loading logic.
- **sentiment_model.h5**: The pre-trained TensorFlow/Keras model for sentiment analysis.
- **tokenizer.pkl**: The tokenizer used for preprocessing the review text.
- **requirements.txt**: List of Python packages required to run the application.
- **templates/index.html**: The HTML template for the web interface.


# Optional work (Deploy application on AWS Cloud Server )

Deploying your Flask application on Amazon EC2.Here’s a detailed step-by-step guide on how to deploy your Flask app on Amazon EC2:

- Log in to AWS Management Console and navigate to the EC2 Dashboard.
- Click Launch Instance to create a new instance.
- Select an Amazon Machine Image (AMI), such as Amazon Linux 2, which comes with essential tools pre-installed.
- Choose an instance type (e.g., t2.micro for - testing purposes, which is eligible for the free tier).
- Configure instance details such as network, subnet, and enable public IP.
- Add storage if the default storage size is not sufficient for your application.
Configure security groups:
- Add a rule to allow SSH (port 22) from your IP address for secure shell access.
- Add a rule to allow HTTP (port 80) and HTTPS (port 443) from anywhere (or specific IPs if you prefer more security).
- Review and launch the instance by selecting a key pair for SSH access. If you don’t have a key pair, create a new one and download it to your machine.