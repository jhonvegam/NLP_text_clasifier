import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline


# Files loading
data = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/eval.csv")

data_X = data["text"]
data_y = data["decade"]

X_train, X_val, y_train, y_val = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)

modelo_texto = make_pipeline(
    TfidfVectorizer(), 
    SVC(random_state=42)
)

modelo_texto.fit(X_train, y_train)

predictions_train = modelo_texto.predict(X_train)
accuracy_train = accuracy_score(y_train, predictions_train)

predictions_val = modelo_texto.predict(X_val)
accuracy = accuracy_score(y_val, predictions_val)

predictions_test = modelo_texto.predict(data_test["text"])

print(accuracy_train)
print(accuracy)

df_envio = pd.DataFrame({
    'id': data_test['id'],
    'answer': predictions_test
})

df_envio.to_csv("prediction_test.csv", index=False)