# Time-Series-Forecasting

## I. Problem Statement:
 In this project, I am aiming to build a model to predict stock price, based on the stock price of past
7 days. Here I am practicing with the time series data to predict stock price.

## II. Methodology:
- Data Pre-processing and Data Cleaning to gain better accuracy.
- Encode the data into numeric range – using MinMax Scaler .
- Apply various models to predict the stock price of the 8th day close, depending up on past
7 days data.
- Using a full-connected neural network model, LSTM and CNN to predict [Close] of a day
based on the last 7 days’ data
- Parameter tuning to compare model with different parameters.


### Step 0: Dataset
- We can see from the below image that there are no categorical features and I have separated the existing features into discrete, continuous and dependent variables.

![Dataset](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.08.08%20AM.png)

### Step 1: Importing the dataset
- Loaded the dataset to analyze the features present.

![import the dataset](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.11.54%20AM.png)

### Step 2: Replacing the null values with NAN values 
- As by doing analysis of the dataset I found that there are no Null values. Finding duplicate values and removing the duplicates.

![Data Preprocessing](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.12.14%20AM.png)

### Step 3: Datatype analysis and dropping less significant features like date and adj_close.

![Dropping columns](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.12.29%20AM.png)

### Step 4: EDA -Exploratory Data Analysis on all features
- I have performed sns plot analysis on the target feature and hist plot analysis on the discrete and continuous features.

![Data Analysis](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.12.54%20AM.png)

### Step 5: Feature Normalization
- I have performed feature normalization using MinMax() Scaler from scikit-learn module. That normalizes all the input features from discrete and continuous to (0,1) scale.

![Data Normalization](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.13.10%20AM.png)

### Step 6: Dataset sequencing and reshaping
- I have then sequence the dataset such that each record has 7 * 5 = 35 input features and 1 output feature(close). Because our goal is to predict [Close] of a day based on the last 7 days’ data [Open, High, Low, Volume, Close].

![Data Sequencing](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.13.27%20AM.png)

### Step 7: Split the data into train and test set
- I am using the first 70% of the available records for training and the remaining 30% of the available records for test.

![Data Splitting](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.13.37%20AM.png)

### Step 8: Model Training
#### Model 1 : Fully Connected Neural Network Model

![Model Training Fully CNN](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.14.00%20AM.png)

#### Model 2 : LSTM Model

![Model Training LSTM](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.14.20%20AM.png)

#### Model 3 : CNN

![Model Training CNN](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.14.45%20AM.png)

![Model Training CNN](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.15.00%20AM.png)

## III. Experimental Results and Analysis
- RMSE of the model and The Regression lift chart
1. Full-Connected Neural Network Model

![Model Training Fully CNN](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.15.13%20AM.png)

2. LSTM

![Model Training LSTM](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.15.26%20AM.png)

3. CNN

![Model Training CNN](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.15.35%20AM.png)

## Comparison of all algorithms
- Here I have compared our results obtained from training our models. I have also compared our results with parameter tuning.
For parameter tuning I have used,
-  L1 and L2 regularization
-  Different Optimizer and Activation Function
-  Different combinations of Neurons.

#### Fully Connected Neural Network

![Comparison Fully CNN](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.15.59%20AM.png)

#### LSTM

![Comparison LSTM](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.16.07%20AM.png)

#### CNN

![Comparison LSTM](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.16.25%20AM.png)

## IV. Challenges and Learning
- How to do feature normalization using Sklearn.
- Applying the models and comparing their performance.
- How to implement neural network using TensorFlow and Keras.
- How to use Early stopping and Save and Use saved best weights of Neural Networks.
- How to implement Convolution Neural Networks.
- How to implement RNN-LSTM and Bidirectional LSTM.
- Parameter Tuning for Neural Network, CNN (Kernel size, Kernel no. and Strides), RNN
- (LSTM cells).


## V. Additional Feature implementation
1) Stock price prediction on GOOGLE and APPLE – Bidirectional LSTM
- I have selected two companies Google and Apple. As an additional feature I have selected two companies Google and Tesla for analyzing stock prices. I have planned on analyzing stock prices from 2010 to 2020. I have performed all the steps explained above in the project and I have performed Model Training using Bidirectional LSTM model.

![Additional Feature](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.16.46%20AM.png)

![Additional Feature](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.17.56%20AM.png)

2) Used LSTM to predict the stock price of next 5 days

![Additional Feature](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.18.47%20AM.png)

![Additional Feature](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.19.01%20AM.png)

![Additional Feature](https://github.com/shmoksh/Time-Series-Forecasting/blob/main/Images/Screen%20Shot%202020-12-26%20at%207.19.14%20AM.png)
