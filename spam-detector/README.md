# Spam Message Detector

A Machine Learning project that classifies text messages as **spam** or **not spam (ham)**, built during my AI & ML Internship at Codomax Digital Solutions.

## About the Project
Spam and scam messages (fake prizes, fake bank alerts, loan offers) are a common problem. This project trains a text classification model that learns the patterns in spam messages and flags new messages automatically.

## How It Works
1. **Dataset:** a labeled set of spam and normal messages
2. **Text to numbers:** TF-IDF converts each message into numerical features based on how important each word is
3. **Model:** a Multinomial Naive Bayes classifier learns which words are common in spam vs. normal messages
4. **Evaluation:** the model is tested on unseen messages using accuracy, a classification report, a confusion matrix, and 5-fold cross-validation
5. **Prediction:** you can type any message and get a spam/ham result with a spam probability

## Tech Stack
- Python
- pandas
- scikit-learn
- matplotlib

## Results
- 5-fold cross-validation accuracy: about 97% on the included dataset
- The dataset is small (60 messages), so results on a single train/test split can vary. Cross-validation gives a more reliable estimate.

## How to Run
1. Install the libraries:
   ```
   pip install pandas scikit-learn matplotlib
   ```
2. Run the project:
   ```
   python spam_detector.py
   ```
3. The program prints the evaluation results, shows a confusion matrix chart, then lets you type your own messages. Type `exit` to quit.

## Example
```
Enter message: You have won a free prize, click here now
Result: SPAM (spam probability: 85.2%)

Enter message: Can you send me the notes from class?
Result: HAM (spam probability: 20.4%)
```

## Improving the Model
For higher accuracy on real-world messages, replace the small built-in dataset with the larger public **SMS Spam Collection** dataset (available on Kaggle and the UCI Machine Learning Repository) and load it with `pandas.read_csv()`.

## What I Learned
- The complete ML workflow: data preparation, training, testing, and prediction
- Converting text into numerical features with TF-IDF
- Why a single test split can be misleading on small datasets, and how cross-validation helps
- Documenting and publishing a project on GitHub

## Author
Ahmed Tasnim | Diploma in Computer Science
Internship: AI & ML, Codomax Digital Solutions