import pandas as pd
from pre_proc import apply_industry
import pickle


def create_df_row(request_info):
    """Creates a DataFrame row from a request dictionary.

    Args:
        request_info (dict): Dictionary containing request information.

    Returns:
        pd.DataFrame: A single-row DataFrame with the request data.
    """
    row = {
        "id": request_info["id"],
        "position": request_info["position"],
        "followers": request_info["followers"],
        "recommendations_count": request_info["recommendations_count"],
        "current_company:name": request_info["current_company:name"],
    }
    row = apply_industry(row)
    row = pd.DataFrame([row])
    row['industry'] = row['industry'].map({'Education': 0, 'IT': 1, 'Accountancy': 2, 'Marketing': 3,'else': 4,
                                           '0': 0, '1': 1, '2': 2, '3': 3})

    row['industry'] = row['industry'].fillna(4)
    return row


def predict_grade(request_info, user_info, model_path="regression_model.pkl"):
    """Predicts a grade using a loaded model and combines it with industry and position similarity.

    Args:
        request_info (dict): Dictionary containing request information.
        user_info (dict): Dictionary containing user information.
        model_path (str, optional): Path to the saved regression model. Defaults to "regression_model.pkl".

    Returns:
        float: The predicted grade.
    """
    request_data = create_df_row(request_info)
    user_data = create_df_row(user_info)

    # Load the model
    model = pickle.load(open(model_path, "rb"))

    # Select relevant features and predict
    grade = model.predict(request_data[["industry", "followers", "recommendations_count"]])[0]

    # Add similarity bonuses based on position and industry
    grade += 1.2 * (request_data["current_company:name"] == user_data["current_company:name"])
    grade += 1.1*(request_data["position"] == user_data["position"])
    grade += 1 * (request_data["industry"] == user_data["industry"])
    grade=min(grade[0],10)


    return grade
