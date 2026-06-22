import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string

np.random.seed(42)

data = [
    ("Congratulations! You won a free iPhone. Click here to claim now.", "spam"),
    ("Hey, are we still meeting for lunch tomorrow?", "ham"),
    ("You have been selected for a $1000 gift card. Claim now!", "spam"),
    ("Please find the attached report for the project.", "ham"),
    ("Win a luxury vacation to Bali. Limited time offer!", "spam"),
    ("Can you send me the documents by Friday?", "ham"),
    ("URGENT: Your account has been compromised. Reset your password now.", "spam"),
    ("Thanks for your email. I will get back to you soon.", "ham"),
    ("Free money! Click here to get your cash reward.", "spam"),
    ("Don't forget about the meeting at 3 PM.", "ham"),
    ("Your credit card has been charged $500. Call us immediately.", "spam"),
    ("Let's catch up over coffee this weekend.", "ham"),
    ("Limited time offer! Buy one get one free.", "spam"),
    ("The project deadline has been extended to next week.", "ham"),
    ("You have won a brand new car! Claim your prize now.", "spam"),
    ("I have sent you the updated file. Please check.", "ham"),
    ("Click here to get free access to premium content.", "spam"),
    ("Are you coming to the office tomorrow?", "ham"),
    ("Your account has been flagged for suspicious activity.", "spam"),
    ("The team meeting has been rescheduled to 4 PM.", "ham"),
    ("Exclusive deal for our customers. Limited stock available.", "spam"),
    ("I appreciate your help with this matter.", "ham"),
    ("You have been selected as the winner of our contest.", "spam"),
    ("Would you like to review the new proposal?", "ham"),
    ("Save up to 50% on your next purchase. Hurry!", "spam"),
    ("Can you please confirm your availability for the interview?", "ham"),
    ("You have a new message from your bank.", "spam"),
    ("The report looks great. Well done!", "ham"),
    ("Get paid instantly for taking surveys online.", "spam"),
    ("I will call you later to discuss the details.", "ham")
]

df = pd.DataFrame(data, columns=['text', 'label'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("SPAM EMAIL DETECTOR")
print("=" * 60)
print()
print("Total Emails:", len(df))
print("Spam:", len(df[df['label'] == 'spam']))
print("Ham:", len(df[df['label'] == 'ham']))
print()
print("Model Training Complete")
print("Accuracy:", round(accuracy * 100, 2), "%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))
print()
print("Sample Predictions:")
for i in range(5):
    print("  Email:", X_test.iloc[i][:40] + "...")
    print("  Actual:", y_test.iloc[i])
    print("  Predicted:", y_pred[i])
    print()
print("=" * 60)
print("SYSTEM READY")
print("=" * 60)