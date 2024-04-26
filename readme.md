# Create Amazon Product Review Sentiment Analysis / Flask Application


### NOTE:   
#### Download the Glove Embedding from Kaggle from here : https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt


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
 - Hassn Sattar 
