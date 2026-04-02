import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df['Available_Hours'] = (
        40
        - df['OverTime'].map({'Yes': 8, 'No': 0})
        - df['WorkLifeBalance'] * 2
    )

    df['Assigned_Hours'] = (
        30
        + df['JobLevel'] * 3
        + df['OverTime'].map({'Yes': 10, 'No': 0})
    )

    df['Task_Complexity'] = df['JobLevel'] * 2

    df['Deadline_Days'] = 30 - df['JobInvolvement'] * 3

    df['Skill'] = df['JobRole']

    df['Risk'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    df['Future_Workload'] = (
        df['MonthlyIncome'] / 1000 + df['YearsAtCompany']
    )

    df['Month'] = df['YearsAtCompany'] % 12

    return df