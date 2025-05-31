import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Step 2: Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Step 4: Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Step 5: Train the SVM model
model = SVC()
model.fit(X_train_vectors, y_train)

# Step 6: Predict and check accuracy
predictions = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Test on custom input
while True:
    custom_message = input("Enter an email (or 'q' to quit): ")
    if custom_message.lower() == 'q':
        break
    vector = vectorizer.transform([custom_message])
    result = model.predict(vector)
    print("Prediction:", "Spam" if result[0] == 1 else "Not Spam")
