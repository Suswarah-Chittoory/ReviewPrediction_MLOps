import numpy as np
import pandas as pd
import regex as re



import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm, tqdm_notebook


nltk.download('stopwords')
# Downloading wordnet before applying Lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from prefect import task, flow



@task
def import_data(file_path):
    return pd.read_csv(file_path)


@task
def handle_nulls(df):
    df.replace("", pd.NA, inplace = True)
    df.replace(" ", pd.NA, inplace = True)
    df.dropna(subset=['Review text', 'Review Title'], how='all', inplace=True)
    df['Review Title'].replace(pd.NA, "None", inplace = True)
    df.isnull().sum()
    return df
    


@task
def feature_extraction(df):
    df.drop(['Reviewer Name', 'Place of Review', 'Up Votes', 'Down Votes', 'Month'], axis=1, inplace = True)
    return df


@task
def feature_engineering(df):
    df['Review text'] = df['Review text'].str.replace(r'READ MORE', '', regex=True)
    # Use replace() with if condition to create the target variable 'Sentiment'
    df['Sentiment'] = df['Ratings'].replace({rating: 1 if rating >= 3 else 0 for rating in df['Ratings']})
    df["Review"] = df['Review Title'] + " " + df['Review text']
    df.drop(['Review Title', 'Review text', 'Ratings'], axis = 1, inplace = True)
    df.head()
    return df


@task
def split_inputs_output(data, inputs, output):
    
    x = data[inputs]
    y = data[output]
    return x, y


@task
def split_train_test(x, y, test_size=0.25, random_state=0):
    
    return train_test_split(x, y, test_size=test_size, random_state=random_state)




@task
def preprocess_data(X_train, X_test, y_train, y_test):
    def preprocessor(text):
        # Removing special characters and digits
        letters_only = re.sub("[^a-zA-Z]", " ", text)
        # change sentence to lower case
        letters_only = letters_only.lower()
        # tokenize into words
        words = letters_only.split()
        # remove stop words
        words = [word for word in words if word not in stopwords.words("english")]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    # Define TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessor)

    # Fit TF-IDF vectorizer on X_train and transform X_train and X_test
    x_train_transformed = tfidf_vectorizer.fit_transform(X_train)
    x_test_transformed = tfidf_vectorizer.transform(X_test)
    return x_train_transformed, x_test_transformed, y_train, y_test


@task
def train_model(x_train_transformed, y_train, hyperparameters):

    clf = LogisticRegression(**hyperparameters)
    clf.fit(x_train_transformed, y_train)
    return clf



@task
def evaluate_model(model, x_train_transformed, y_train, x_test_transformed, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(x_train_transformed)
    y_test_pred = model.predict(x_test_transformed)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score




@flow(name="Logistic Regression final Flow")
def workflow():
    DATA_PATH = "C:\\Users\\Suswarah\\Downloads\\MLOps\\MLFlow\\badminton_review_data.csv"
    INPUTS = 'Review'
    OUTPUT = 'Sentiment'
    HYPERPARAMETERS = {'C': 10, 'max_iter': 5000, 'penalty': 'l2'}


    # Load data
    df = import_data(DATA_PATH)
    
    # Handle nulls
    df = handle_nulls(df)
    
    # Extract Features
    df = feature_extraction(df)
    
    # Feature Engineering
    df = feature_engineering(df)

    
    # Identify Inputs and Output
    x, y = split_inputs_output(df, INPUTS, OUTPUT)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    
    # Preprocess the data
    x_train_transformed, x_test_transformed, y_train, y_test = preprocess_data(x_train, x_test, y_train, y_test)

    # Build a model
    model = train_model(x_train_transformed, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, x_train_transformed, y_train, x_test_transformed, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)



if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="3 * * * *"
    )






