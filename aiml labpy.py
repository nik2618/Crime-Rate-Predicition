import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

file_path = 'crime_rate_prediction_dataset_sorted.csv'
df = pd.read_csv(file_path)

label_encoder_type = LabelEncoder()
label_encoder_city = LabelEncoder()

df['Type_encoded'] = label_encoder_type.fit_transform(df['Type'])
df['City_encoded'] = label_encoder_city.fit_transform(df['City'])

X = df[['Year', 'Population (in Lakhs)', 'Type_encoded', 'Crime Rate']]
y = df['City_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print(f"Model Accuracy: {accuracy * 100:.2f}%")
