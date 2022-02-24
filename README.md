# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

### Files

- data/
  - DisasterResponse.db
    a sqlite database containing one table 'messages' with all messages and their categories.
  - disaster_categories.csv
    a .csv file containing a list of categories corresponding to each disaster message.
  - disaster_messages.csv
    a .csv file containing all disaster messages.
  - process_data.py
    loads messages and their categories from the two .csv files, cleans the messages, removes duplicates, splits the different categories and saves the cleaned dataset to a sqlite database.
- models/
  - train_classifier.py
    reads the table of cleaned messages and their categories from a sqlite database, trains, evaluates and saves a multi-label supervised learning classifier allowing the classification of new disaster messages into multiple labels.
- app
  - run.py
    launches a flask web app that allows a user to see several dashboards containing statistics about the different message genres and classify new new disaster messages into 36 different categories.
  - templates
    - go.html
      a jinja2 html template showing the categories of a given message.
    - master.html
      the main jinja2 html template (frontend) of the flask application allowing a user to see statistics about existing disaster messages and classify new messages.

### Dependencies

- Flask==0.12.5
- nltk==3.2.5
- pandas==0.23.3
- plotly==2.0.15
- scikit_learn==1.0.2
- SQLAlchemy==1.2.19
