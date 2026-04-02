from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
from preprocess import load_and_preprocess

df = load_and_preprocess("dataset.csv")

X_reg = df[['Available_Hours', 'Assigned_Hours', 'Task_Complexity']]
y_reg = df['Future_Workload']

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

print("MAE:", mean_absolute_error(y_test, reg_model.predict(X_test)))

X_clf = df[['Available_Hours', 'Assigned_Hours', 'Deadline_Days']]
y_clf = df['Risk']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2)

clf_model = DecisionTreeClassifier()
clf_model.fit(X_train_c, y_train_c)

print("Accuracy:", accuracy_score(y_test_c, clf_model.predict(X_test_c)))

prob_model = LogisticRegression(max_iter=1000)
prob_model.fit(X_train_c, y_train_c)

joblib.dump(reg_model, "reg_model.pkl")
joblib.dump(clf_model, "clf_model.pkl")
joblib.dump(prob_model, "prob_model.pkl")