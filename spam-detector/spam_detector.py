# Module 6 - Final AI & ML Project
# Spam Message Detector using Machine Learning (Text Classification)
# Libraries: pandas, scikit-learn, matplotlib
# Install if needed: pip install pandas scikit-learn matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# 1. DATASET
# A small labeled dataset is included so the project runs anywhere.
# For better accuracy, replace it with the larger "SMS Spam Collection"
# dataset (available on Kaggle / UCI) -- see README for instructions.
# ---------------------------------------------------------
spam_messages = [
    "Congratulations! You have won a free iPhone. Click the link to claim now",
    "URGENT: Your bank account is blocked. Verify your details immediately",
    "Win cash prizes now! Reply WIN to 5000 to enter the lucky draw",
    "Free entry in our weekly contest. Text WIN to claim your reward",
    "You have been selected for a loan of 5 lakh. Apply now with no documents",
    "Claim your free gift card today. Limited time offer, click here",
    "Your lottery ticket has won 10 million. Send your bank details to receive the prize",
    "Get cheap medicines online without prescription. Order now and save 80 percent",
    "Earn 50000 per month working from home. No experience needed. Join now",
    "Final notice: your electricity will be cut tonight. Pay immediately at this link",
    "Hot singles in your area are waiting. Click to chat now for free",
    "Exclusive offer! Buy one get three free. Hurry, offer ends today",
    "Your KYC is expired. Update now or your account will be suspended",
    "You are our lucky winner. Claim your free vacation to Goa today",
    "Double your money in 7 days guaranteed. Invest now with zero risk",
    "Act now and get 90 percent discount on luxury watches. Click the link",
    "Your parcel is held at customs. Pay the fee online to release it",
    "Free recharge of 500 rupees. Just click and fill in your details",
    "Dear customer, you have won a car. Call this number to claim your prize",
    "Limited seats left! Get a guaranteed job offer by paying a small registration fee",
    "Unlock your free credit card with 0 percent interest. Apply instantly",
    "Congratulations, you are pre-approved for a personal loan. Reply YES to accept",
    "Click here to watch exclusive videos free. Adults only. Subscribe today",
    "Your account password has expired. Login at this link to avoid deactivation",
    "Make money fast with this secret trick. Millionaires hate it. Learn more now",
    "Special deal just for you: free mobile phone with every purchase. Order now",
    "Warning: your device has a virus. Download this security app immediately",
    "You won a free trip to Dubai. Send your passport details to confirm",
    "Last chance to claim your cash reward. Offer expires in 2 hours",
    "Get rich quick with crypto. Guaranteed 10x returns. Join our group now",
]

ham_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Please send me the notes from today's class when you get a chance",
    "Happy birthday! Hope you have a wonderful day with your family",
    "I will be late to the lab, please start without me",
    "Can you call me when you are free? I need to ask something about the project",
    "Don't forget to submit the assignment before 5 pm today",
    "Mom said dinner will be ready at 8, come home on time",
    "The internship report is due next week, let's finish it together",
    "Thanks for helping me with the Python code yesterday",
    "Are you coming to college tomorrow? There is a test in the first period",
    "Let's study together this evening for the exam",
    "I reached home safely, see you in class tomorrow",
    "Can you share the GitHub link of your project?",
    "The meeting has been moved to 3 pm, please confirm if you can attend",
    "Good morning! Have a great day ahead",
    "I am going to the library, do you need any book?",
    "Please bring your ID card tomorrow for the lab session",
    "Did you watch the match last night? It was amazing",
    "Let me know when you are done with the weekly report",
    "Happy to help, just tell me which topic you are stuck on",
    "Our project demo is on Friday, let's practice once more",
    "I will send you the photos from the trip tonight",
    "Can we reschedule our call to tomorrow morning?",
    "Remember to carry your laptop charger for the workshop",
    "Thanks a lot for your help, I really appreciate it",
    "I finished the Colab notebook, can you review it?",
    "Dinner tonight at my place? Bring your cousin along too",
    "The bus is running late today, I will reach in 15 minutes",
    "Please check your email, I have sent the updated schedule",
    "Great work on the presentation today, well done",
]

df = pd.DataFrame({
    "message": spam_messages + ham_messages,
    "label": ["spam"] * len(spam_messages) + ["ham"] * len(ham_messages),
})

print("Dataset size:", df.shape)
print(df["label"].value_counts())
print("\nSample rows:")
print(df.sample(5, random_state=1))

# ---------------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

# ---------------------------------------------------------
# 3. BUILD AND TRAIN THE MODEL
# TF-IDF converts text into numbers; Naive Bayes classifies the message
# ---------------------------------------------------------
model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english"),
    MultinomialNB(),
)
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# ---------------------------------------------------------
# 4. TEST AND EVALUATE THE MODEL
# ---------------------------------------------------------
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# The test set is small, so a single split can be misleading.
# Cross-validation tests the model on 5 different splits for a more reliable score.
cv_scores = cross_val_score(model, df["message"], df["label"], cv=StratifiedKFold(5, shuffle=True, random_state=0))
print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# Confusion matrix chart
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(cmap="Blues")
plt.title("Spam Detector - Confusion Matrix")
plt.show()

# ---------------------------------------------------------
# 5. PREDICT NEW MESSAGES
# ---------------------------------------------------------
def predict_message(text):
    label = model.predict([text])[0]
    spam_probability = model.predict_proba([text])[0][list(model.classes_).index("spam")]
    return label, spam_probability

test_messages = [
    "Congratulations! You won a free prize, click here to claim",
    "Hey, can you send me the project report tonight?",
    "URGENT: verify your bank account now to avoid suspension",
    "Let's meet at the library after class",
]

print("\nPredictions on new messages:")
for msg in test_messages:
    label, prob = predict_message(msg)
    print(f"[{label.upper()}] ({prob * 100:.1f}% spam) -> {msg}")

# ---------------------------------------------------------
# 6. INTERACTIVE MODE
# ---------------------------------------------------------
print("\n===== Spam Message Detector =====")
print("Type a message to check if it is spam. Type 'exit' to quit.\n")

while True:
    user_msg = input("Enter message: ")
    if user_msg.lower() == "exit":
        print("Thank you for using the Spam Message Detector!")
        break
    label, prob = predict_message(user_msg)
    print(f"Result: {label.upper()} (spam probability: {prob * 100:.1f}%)\n")