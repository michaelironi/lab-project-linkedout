import pprint

from Decision_func import predict_grade


if __name__ == "__main__":
    # Example usage
    request_info = {"id": "1", "position": "devops", "followers": 100, "recommendations_count": 3,
                    "current_company:name": "Google"}
    user_info = {"id": "2", "position": "data engineer", "followers": 100, "recommendations_count": 5,
                 "current_company:name": "Google"}

    print(
        f"User Info:\n  Position: {user_info['position']}\n  Followers: {user_info['followers']}\n  Recommendations: {user_info['recommendations_count']}\n  Company: {user_info['current_company:name']}")

    print(
        f"Request Info:\n  Position: {request_info['position']}\n  Followers: {request_info['followers']}\n  Recommendations: {request_info['recommendations_count']}\n  Company: {request_info['current_company:name']}")

    grade = predict_grade(request_info, user_info)
    print(f"request grade: {grade}")
