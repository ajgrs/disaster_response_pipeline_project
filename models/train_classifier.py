import sys

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

import pickle

def load_data(database_filepath):
    """
        Purpose:
        loads data from a sqlite database into a pandas dataframe

        Arguments:
        database_filepath -- the filepath of the sqlite database to connect to

        Returns:
        the training values, training labels and an array of category names for the labels
    """

    # creates a new sqlalchemy engine
    engine = create_engine('sqlite:///' + database_filepath)

    # gets all messages and classifications from the database
    df = pd.read_sql("SELECT * FROM messages", engine)

    # fixes a small bug in data
    df.related.replace(2, 1, inplace=True)

    # puts messages in the X variable
    X = df["message"]

    # puts classification into the y variable
    y = np.asarray(df[df.columns[4:]])

    # gets the necessary category names
    category_names = df.columns[4:].tolist()

    return X, y, category_names


def tokenize(text):
    """
        Purpose:
        normalises, tokenises and lemmatises a text string.

        Arguments:
        text -- the string to normalise, tokenise and lemmatise.

        Returns:
        the text duly normalised, tokenised and lemmatised.
    """

    # normalizes received text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenizes text
    words = word_tokenize(text)

    # removes stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # lemmatizes verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos = 'v') for w in words]

    return lemmed


def build_model():
    """
        Purpose:
        builds a multi-label pipeline supervised training model that tokenizes a sentence, creates a matrix of term counts,
        creates features for term frequency and inverse term frequency and classifies new messages using multiple labels.

        Returns:
        a pipeline model ready to be trained
    """

    # defines a pipeline using tf-idf
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])

    #parameters = {'clf__estimator__n_neighbors': [5, 5]}
    parameters = {'clf__n_neighbors': [5, 5]}

    model = GridSearchCV(estimator = pipeline, param_grid = parameters, scoring = 'f1_micro', cv = 2, verbose = 3, n_jobs = -1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Purpose:
        evaluates a model using its precision, recall and f1 scores for each category.
        it also computes the model's accuracy for each category.

        Arguments:
        model -- the pipeline model to evaluate.
        X_test -- the test set on which to evaluate the model.
        Y_test -- the labels on which to evaluate the model.
        category_names -- the category names of each label present in the model.
    """

    # gets model predictions on test set
    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):

    """
        Purpose:
        saves a pipeline model to disk.

        Arguments:
        model -- a pipeline model duly trained.
        model_filepath -- the filepath where to save the model.
    """

    pickle.dump(model, open(model_filepath,'wb'))


def main():

    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
