import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Purpose:
        loads all messages and categories into a single dataframe

        Arguments:
        messages_filepath -- the csv messages file to use
        categories_filepath -- the csv categories file to use

        Returns:
        a dataframe with all messages and their categories
    """

    # loads messages dataset
    messages = pd.read_csv(messages_filepath)

    # loads categories dataset
    categories = pd.read_csv(categories_filepath)

    # merges the two datasets using the common "id" column
    df = messages.merge(categories, how = 'outer', on = ['id'])

    return df


def clean_data(df):
    """
        Purpose:
        merges the messages and categories datasets, splits the
        categories column into separate, clearly named columns,
        converts values to binary, and drops duplicates.

        Arguments:
        df -- a dataframe containing all messages and their categories.

        Returns:
        a cleaned dataframe with all messages and their corresponding categories flagged as '1'
        ready to be used in a multi-label classifier.
    """

    # gets the individual categories in a new dataframe by splitting using the ";" separator yielding one category per column -> 36 categories in total
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # applies a lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:len(x) - 2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:

        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, join = 'inner', sort = False)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    """
        Purpose:
        saves a cleaned messages and categories dataframe to a sqlite database on disk

        Arguments:
        df -- the dataframe to save
        database_filename -- the sqlite database name

    """

    # creates a new SQLAlchemy engine
    engine = create_engine('sqlite:///' + database_filename)

    # stores the cleaned df to the database
    df.to_sql('messages', engine, index = False)


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!') 

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
