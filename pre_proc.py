import pickle
import numpy as np
import pandas as pd

# Activities we want to consider for scoring
activities_to_score = [ "expert opinions",
                       "skill-building", "Industry news"]
candidate_labels = ['Career advice','Industry news', 'skill-building' , 'Networking' ,'Company updates' ,'expert opinions' ,'startup culture','Marketing' ,'Diversity', 'Work-life balance', 'Economic news', 'success stories']


def active_users():
    # Read the CSV file
    df = pd.read_csv('D:\Study Docs\Degree Material\Sem 7 proj\lab1\proj\j_lm67wmhz252pzdjs60.1693912734086.csv')

    # filter only the rows with the column 'post' not empty lists as in [] and select the 'url' column
    df = df[df['posts'].apply(lambda x: x != '[]')]

    df.to_csv('D:\Study Docs\Degree Material\Sem 7 proj\lab1\proj\j_lm67wmhz252pzdjs60.1693912734086_100k.csv',
              index=False)


def get_industry():
    # Load the model
    model = pickle.load(open('industry_classifier.pkl', 'rb'))

    # load profiles_activities.csv and apply the model on column 'position' and save the result in a new column 'industry'
    # where position is not np.nan
    df = pd.read_csv('merged_data.csv')
    df['industry'] = np.nan
    df['industry'] = df['industry'].astype('object')
    df['industry'] = df['position'].apply(lambda x: model.predict([x])[0] if not pd.isnull(x) else np.nan)

    # save the result in a new csv file
    df.to_csv('profiles_industries.csv', index=False)


def apply_industry(row):
    # Load the model
    model = pickle.load(open('industry_classifier.pkl', 'rb'))

    # apply the model on column 'position' and save the result in a new column 'industry'
    # where position is not np.nan
    row['industry'] = model.predict([row['position']])[0] if not pd.isnull(row['position']) else np.nan

    return row



def merge_scrapes():
    # Read the CSV file
    profiles = pd.read_csv('D:\Study Docs\Degree Material\Sem 7 proj\lab1\proj\post_pre process.csv')
    posts = pd.read_csv('D:\Study Docs\Degree Material\Sem 7 proj\lab1\proj\scraped_posts.csv')

    # merge the two dataframes based on the id column
    df = pd.merge(profiles, posts, on='id')

    # Save the merged dataframe to a new CSV file
    df.to_csv('D:\Study Docs\Degree Material\Sem 7 proj\lab1\proj\merged_data.csv', index=False)



def filter_columns(df):
    # Read the CSV file
    columns=['id', 'current_company:name', 'industry', 'followers','recommendations_count','activities','activities_scores']
    dfc = df[columns]
    #if recommend_count is null then replace it with 0
    dfc['recommendations_count'] = dfc['recommendations_count'].fillna(0)
    #index the industry column where Education = 0 IT = 1 Accountancy = 2 Marketing = 3 and replace the null values with 4
    dfc['industry'] = dfc['industry'].map({'Education': 0, 'IT': 1, 'Accountancy': 2, 'Marketing': 3,'else': 4,
                                           '0': 0, '1': 1, '2': 2, '3': 3})

    dfc['industry'] = dfc['industry'].fillna(4)
    #Change column name from 'output' to 'activities_scores'
    # dfc = dfc.rename(columns={'output': 'activities_scores'})
    #activities_scores is list of dictionaries so convert it to a dictionary by calculating the mean of the values for each same key in every cell

    def mean_dict(x):
        d = {}
        for i in x:
            for k, v in i.items():
                d[k] = d.get(k, []) + [v]
        return {k: sum(v) / len(v) for k, v in d.items()}

    # dfc['activities_scores'] = dfc['activities_scores'].apply(eval).apply(mean_dict)



    return dfc


def calculate_grade(activity_scores):
  """Calculates a total grade based on the provided activity scores.

  Args:
      activity_scores: A dictionary containing activity scores.

  Returns:
      The total grade calculated by summing scores for defined activities (handling missing values).
  """
  # Get sum of values for the desired activities (handling missing values with default 0)
  total_score = sum(activity_scores.get(key, 0) for key in activities_to_score)
  # Scale the score by 20 and upper bound it at 10
  return min(total_score * 20, 10)


# Function to convert string representation of dictionary to a dictionary
def parse_activity_scores(scores_string):
  """Parses a string representation of a dictionary into a dictionary object.

  Args:
      scores_string: The string containing the dictionary data.

  Returns:
      A dictionary object representing the parsed data.
  """
  return eval(scores_string)  # Assuming the string is a valid dictionary format

# Function to sort the dictionary by the desired keys
def sort_activities_by_keys(scores):
  """Sorts an activity score dictionary based on the desired activities list.

  Args:
      scores: A dictionary containing activity scores.

  Returns:
      A new dictionary with scores sorted based on the original dict
  """
  return {key: scores[key] for key in candidate_labels }

def normalize_scores(row):
  scores = row['activities_scores']
  total_score = sum(scores.values())
  normalized_scores = {label: score / total_score for label, score in scores.items()}
  return normalized_scores


