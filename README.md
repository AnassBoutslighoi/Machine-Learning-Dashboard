# Machine Learning Dashboard

This repository contains two machine learning dashboards developed for distinct use cases.

## Dashboard 1: Machine Learning Dashboard

This dashboard serves as a machine learning model training and evaluation tool. It facilitates the following functionalities:

- **Dataset Handling**: Users can upload a CSV dataset and preprocess it by handling missing values, encoding categorical variables, and scaling numeric variables.
- **Data Splitting**: Allows users to divide the dataset into train and test sets for model evaluation purposes.
- **Classifier Selection**: Provides options for three classifiers: Random Forest, Support Vector Machine (SVM),Logistic Regression,..ect. Users can fine-tune the model's hyperparameters via a sidebar UI.
- **Model Training**: Trains the selected model on the dataset and displays crucial evaluation metrics like accuracy score and confusion matrix.
- **Downloadable Outputs**: Enables users to download preprocessed data, the trained model, and predictions on the test set.

Explore the Machine Learning Dashboard [here](https://machine-learning-dashboard-um6.streamlit.app/).

## Dashboard 2: Diabetes Use Case Dashboard

This dashboard focuses on a specific use case related to diabetes classification. Its features include:

- **Specific Use Case**: Designed for the classification of diabetes-related data.
- **Similar Functionality**: Offers dataset handling, data splitting, classifier selection (Random Forest, SVM, Logistic Regression, etc...), hyperparameter tuning, model training, and downloadable outputs.
- **Diabetes-Specific Model**: Tailored towards diabetes-related dataset and classification tasks.

Access the Diabetes Use Case Dashboard [here](https://ml-dashboard-diabetes-usecase.streamlit.app/).

### Setup

To run this project locally, the following steps can be followed:

1. **Installation**: Ensure Python and necessary libraries are installed. If you haven't installed **Python**, consider directly installing the **Anaconda distribution.**
Anaconda is a Python distribution designed for Data Science.

    This allows the installation of Python and its Data Science libraries like Pandas, Matplotlib, Seaborn, Scipy, Numpy, etc...
    It also includes Jupyter notebook, which is essential and highly recommended!
    Get it here: [Anaconda](https://anaconda.org/).

2. **Dependencies Installation**:

    ```bash
    pip install streamlit
    pip install pandas
    pip install matplotlib
    pip install numpy
    pip install scipy
    pip seaborn==0.11.0
    pip plotly>=4.4
    pip requests==2.25.1
    pip Pillow==8.1.0
    pip scikit_learn>=0.24.1
    pip lightgbm==3.2.1
    pip shap>=0.39

    ```

3. **Running the Dashboard**:

    ```bash
    streamlit run app.py
    ```

## Project Details

### Data Sources

The project uses data from the Kaggle dataset [Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).

### Dependencies

- **Streamlit**
- **Scipy**
- **plotly**
- **xgboost**
- **Statsmodels**
- **Pandas**: Version >=1.1.3
- **Matplotlib**: Version >=1.19.2
- **Numpy**: Version >=1.19.2
- **seaborn**: Version==0.11.0
- **matplotlib**: Version>=3.2.2
- **plotly**: Version>=4.4
- **requests**: Version==2.25.1
- **Pillow**: Version==8.1.0
- **scikit_learn**: Version>=0.24.1
- **lightgbm**: Version==3.2.1
- **shap**: Version>=0.39

## Additional Information

- **Author**: Anass Boutslighoi
- **Machine Learning Dashboard Link**: [Machine Learning Dashboard Link](https://machine-learning-dashboard-um6.streamlit.app/)
- **Machine Learning Dashboard : Classification diabetes Use Case Link**: [Classification diabetes Use Case Link](https://ml-dashboard-diabetes-usecase.streamlit.app/)
- **Github Link**: [Machine Learning Dashboard Link - Eqdom](https://github.com/AnassBoutslighoi/Machine-Learning-Dashboard/)

