# Predict Housing Prices

### Project Overview

#### Predicting Housing Prices

The primary goal of this project is to predict housing prices based on various factors, such as the number of rooms, bathrooms, and available amenities. By leveraging machine learning techniques, we aim to provide accurate and reliable price estimates to help potential homebuyers and real estate professionals make informed decisions.

### Table of Contents

- [Project Overview](#project-overview)
- [Objective](#objective)
- [Front End Development](#front-end-development)
- [Machine Learning Approach](#machine-learning-approach)
- [Technologies and Tools](#technologies-and-tools)
- [Dataset](#dataset)
- [Team Responsibilities](#team-responsibilities)
- [Repository](#repository)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Status and Future Improvements](#project-status-and-future-improvements)

### Objective

The main objective of this project is to develop a predictive model that accurately estimates housing prices based on various attributes, such as the number of rooms, bathrooms, and other available feautures. By utilizing machine learning algorithms, we aim to provide potential homebuyers and real estate professionals with reliable price predictions, enabling them to make more informed decisions in the housing market.

### Exploratroy Data Analysis (EDA)

We performed extensive Exploratory Data Analysis (EDA) to understand the underlying patterns and relationships in the dataset. The EDA was divided into three main parts:

1. **Univariate Analysis**: Examined individual features in the dataset to understand their distributions and detect any anomalies.
   
2. **Bivariate Analysis**: Analyzed relationships between pairs of features to identify correlations and potential predictors for housing prices.
   
3. **Multivariate Analysis**: Explored interactions between multiple features simultaneously to uncover complex relationships.

### Machine Learning Approach

For the predictive model, we will employ sklearn Random Forest Regressor to understand and model the relationship between the input features (e.g., number of rooms, bathrooms, amenities) and housing prices. Random Forest Regressor is chosen for its simplicity and effectiveness in handling this type of regression problem. The model will be saved into a pickle file.

### Front End Development

We will develop an intuitive and user-friendly website where users can input details about a property, including the number of rooms, bathrooms, and other relevant amenities. The website will then utilize our predictive model to estimate the property's market price, offering users immediate insights based on their inputs.

### Back End Development

We will develop a Flask application to deploy our predictive model and manage the different routes for the website. The backend will be responsible for processing user inputs, querying the machine learning model, and returning the predicted housing prices.


### Technologies and Tools

- **Programming Language**: The primary language for data preprocessing, exploratory data analysis (EDA), and model development will be Python. 
- **Frameworks and Libraries**:
  - **Front End**: HTML, CSS, JavaScript
  - **Back End**: Flask, Python
  - **Machine Learning**: scikit-learn, pandas, numpy
  - **Model Serialization**: pickle
 
### Dataset

The dataset for training and testing our model is sourced from Kaggle, specifically the [Housing Price Dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset/data). This dataset includes various attributes relevant to housing prices, allowing for comprehensive analysis and model training.

#### Dataset Features

- **ID**
- **Date**
- **Bedrooms**
- **Bathrooms**
- **Floors**
- **Sqft of Living Space**
- **Sqft of Lot**
- **Waterfront**
- **View**
- **Condition**
- **Grade**
- **Sqft Above**
- **Sqft Basement**
- **Year Built**
- **Year Renovated**
- **Zipcode**
- **Latitude and Longitude**
- **Sqft Living15**
- **Sqft Lot15**
- **Price** (target variable)

### Team Responsibilities

- **Data Cleaning**
  - **Team Member**: Rodney
  - **Tasks**: Cleaning and preparing the dataset for analysis using Python, and updating the dataset post-cleaning.

- **Exploratory Data Analysis (EDA)**
  - **Team Members**: Jess and Rossie
  - **Tasks**: Performing EDA using the updated dataset to uncover patterns and relationships within the data, particularly focusing on how features like the number of bedrooms, bathrooms, and square footage correlate with housing prices.

- **Machine Learning Model Development**
  - **Team Members**: Jess and Rossie
  - **Tasks**: Developing the Random Forest Regressor model using scikit-learn, using visualization plot for feature importance for optimal performance, and saving trained model with pickle.

- **Website Development (Front End and Back End Development)**
  - **Team Member**: Francesca
  - **Tasks**: Responsible for designing and implementing the user interface, setting up the server, and integrating the model. 

### Repository

For more detailed information and access to our codebase, please visit our project repository: [Link to Repository](https://github.com/RossieJimenez/House-Price-Prediction-Project.git) <!--  -->

### Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone [repository link]

2. **Install required dependencies**:
   ```bash
    pip install -r requirements.txt

3. **Run the Flask app**:
    ```bash
    python app.py

### Usage

1. **Open the web browser and navigate**:
   ```bash
    http://localhost:5000

2. *Enter the property details in the input form*


3. *Click **"Predict"** to see the estimated housing price*


### Project  Future Improvements
The project is currently in the initial stages. Future improvements include:

* Adding more features to the model for better predictions.
* Implementing a more advanced machine learning algorithm.
* Enhancing the user interface of the website.
* Providing deployment instructions for cloud platforms.
