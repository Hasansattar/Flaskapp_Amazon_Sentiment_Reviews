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




# Run EC2 Instance

Install Python Virtualenv
```bash
sudo apt-get update
sudo apt-get upgrade

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt-get install python3.10-venv
```
Activate the new virtual environment in a new directory

Create directory
```bash
mkdir helloworld
cd helloworld
```
Create the virtual environment
```bash
python3.10 -m venv venv
```
Activate the virtual environment
```bash
source venv/bin/activate
```
Install Flask
```bash
pip install Flask 
pip install tensorflow==2.12.0 --no-cache-dir 
```
Create a Simple Flask API
```bash
sudo nano app.py
```
```bash
from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model and tokenizer
model = load_model('sentiment_model.h5')
# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        new_review = request.form['review']    
        sentiment = predict_sentiment(new_review, model, tokenizer)
            
        return render_template('index.html', sentiment=sentiment , new_review=new_review )
    return render_template('index.html')



def preprocess_text(text, tokenizer, max_length=250):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

def predict_sentiment(text, model, tokenizer):
    preprocessed_text = preprocess_text(text, tokenizer)
    prediction = model.predict(preprocessed_text)[0]
    print("Prediction:", prediction)
    sentiment_label = 'Negative' if prediction < 0.5 else 'Positive'
    return sentiment_label


if __name__ == "__main__":
    app.run(debug=True)

```

Create directory
```bash
mkdir templates
cd templates
```
Create a HTML file inside the template directory

```bash
sudo nano index.html
```

```HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>  Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f0f0;
        }
        h1 {
            text-align: center;
            margin-top: 50px;
            color: #333;
        }
        form {
            text-align: center;
            margin-top: 30px;
        }
        input[type="text"] {
            width: 300px;
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
            outline: none;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        h2 {
            text-align: center;
            margin-top: 20px;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>Sentiment Analysis of User Reviews</h1>
    <form action="/" method="post">
        <input type="text" name="review" placeholder="Enter a review" required>
        <button type="submit">Analyze Sentiment</button>
    </form>
    {% if sentiment %}
        <h2>Review: {{ new_review }}</h2>
        <h2>Sentiment: {{ sentiment }}</h2>
    {% endif %}
</body>
</html>


```

Create a requirements.txt file

```TxT
Flask==2.0.1
tensorflow==2.12.0  # Updated to a version available in the repositories
numpy>=1.19.5       # Ensure compatibility with the TensorFlow version
scikit-learn>=0.24.2  # If you used any functionalities from scikit-learn
pickle-mixin==1.0.2
gunicorn==20.1.0  # If deploying using Gunicorn, especially useful in production

```


Transfer the "sentiment_model.h5" and tokenizer.pkl file to the EC2 instance using SCP.Transfer the file from your local PC to the EC2 instance

```bash
scp -i ~/keys/my-ec2-key.pem ~/example.txt ubuntu@ec2-xx-xxx-xxx-xxx.compute-1.amazonaws.com:/home/ubuntu/

scp -i flaskapp.pem sentiment_model.h5 ubuntu@ec2-54-226-147-136.compute-1.amazonaws.com:/home/ubuntu/helloworld


scp -i flaskapp.pem tokenizer.pkl ubuntu@ec2-54-226-147-136.compute-1.amazonaws.com:/home/ubuntu/helloworld

```




Verify if it works by running 
```bash
python app.py
```
Run Gunicorn WSGI server to serve the Flask Application
When you “run” flask, you are actually running Werkzeug’s development WSGI server, which forward requests from a web server.
Since Werkzeug is only for development, we have to use Gunicorn, which is a production-ready WSGI server, to serve our application.

Install Gunicorn using the below command:
```bash
pip install gunicorn
```
Run Gunicorn:
```bash
gunicorn -b 0.0.0.0:8000 app:app 
```
Gunicorn is running (Ctrl + C to exit gunicorn)!

Use systemd to manage Gunicorn
Systemd is a boot manager for Linux. We are using it to restart gunicorn if the EC2 restarts or reboots for some reason.
We create a <projectname>.service file in the /etc/systemd/system folder, and specify what would happen to gunicorn when the system reboots.
We will be adding 3 parts to systemd Unit file — Unit, Service, Install

Unit — This section is for description about the project and some dependencies
Service — To specify user/group we want to run this service after. Also some information about the executables and the commands.
Install — tells systemd at which moment during boot process this service should start.
With that said, create an unit file in the /etc/systemd/system directory
	
```bash
sudo nano /etc/systemd/system/helloworld.service
```
Then add this into the file.
```bash
[Unit]
Description=Gunicorn instance for a simple hello world app
After=network.target
[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/helloworld
ExecStart=/home/ubuntu/helloworld/venv/bin/gunicorn -b localhost:8000 app:app
Restart=always
[Install]
WantedBy=multi-user.target
```
Then enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl start helloworld
sudo systemctl enable helloworld
```
Check if the app is running with 
```bash
curl localhost:8000
```
Run Nginx Webserver to accept and route request to Gunicorn
Finally, we set up Nginx as a reverse-proxy to accept the requests from the user and route it to gunicorn.

Install Nginx 
```bash
sudo apt-get install nginx
```
Start the Nginx service and go to the Public IP address of your EC2 on the browser to see the default nginx landing page
```bash
sudo systemctl start nginx
sudo systemctl enable nginx
```
Edit the default file in the sites-available folder.
```bash
sudo nano /etc/nginx/sites-available/default
```
Add the following code at the top of the file (below the default comments)
```bash
upstream flaskhelloworld {
    server 127.0.0.1:8000;
}
```
Add a proxy_pass to flaskhelloworld atlocation /
```bash
location / {
    proxy_pass http://flaskhelloworld;
}
```
Restart Nginx 
```bash
sudo systemctl restart nginx
```
 Our application is up! 
 54.226.147.136



