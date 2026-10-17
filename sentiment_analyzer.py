# Module 5 - AI Tools & Mini Project
# Simple AI-Powered Sentiment Analyzer
# Install first (run once in terminal or Colab cell): pip install textblob

from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # ranges from -1 (negative) to +1 (positive)

    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return sentiment, polarity

def sentiment_app():
    print("===== AI-Powered Sentiment Analyzer =====")
    print("Type a sentence and the AI will tell you its sentiment.")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("Enter a sentence: ")

        if user_text.lower() == "exit":
            print("Thank you for using the Sentiment Analyzer!")
            break

        sentiment, polarity = analyze_sentiment(user_text)
        print(f"Sentiment: {sentiment} (Polarity Score: {polarity:.2f})\n")

# Run the app
sentiment_app()