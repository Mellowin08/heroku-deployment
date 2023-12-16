from flask import Flask, jsonify, render_template, request, redirect, url_for, session 
# Import function
from satisfaction_analysis import predict_sentiment
import pandas as pd
import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Set a secret key for sessions
app.secret_key = 'your_secret_key_here'

@app.route('/individual_review.html')
def individual_reviews():
   return render_template('individual_review.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['review']

        # Get Prediction using imported function
        predicted_sentiment, (negative_confidence, neutral_confidence, positive_confidence) = predict_sentiment(user_input)

        return f"{predicted_sentiment},{negative_confidence},{neutral_confidence},{positive_confidence}"

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

def extract_phrases(text, min_length=4, max_length=6):
    doc = nlp(text)
    phrases = set()  # Use a set to store phrases

    current_phrase = []

    for token in doc:
        if token.is_alpha:
            current_phrase.append(token.text)
        else:
            if min_length <= len(current_phrase) <= max_length:
                phrases.add(" ".join(current_phrase))  # Add phrases to the set
            current_phrase = []

    if min_length <= len(current_phrase) <= max_length:
        phrases.add(" ".join(current_phrase))  # Add the last phrase to the set

    return phrases

def extract_phrases_nopunc(text):
    highlighted_phrases = []

    # Define the parts of speech that are typically part of phrases
    phrase_pos_tags = {"ADJ", "NOUN", "ADV"}

    doc = nlp(text)

    for i, token in enumerate(doc):
        if token.pos_ in phrase_pos_tags:
            phrase = token.text
            for j in range(1, 6):
                if i + j < len(doc):
                    next_token = doc[i + j]
                    if next_token.pos_ in phrase_pos_tags:
                        phrase += " " + next_token.text
                    else:
                        break
                else:
                    break
            if 3 <= len(phrase.split()) <= 6:
                highlighted_phrases.append(phrase)

    return highlighted_phrases

def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

@app.route('/consolidated_reviews.html')
def consolidated_reviews():
    return render_template('consolidated_reviews.html')

@app.route('/consolidate_csv', methods=['POST'])
def consolidate_csv():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(uploaded_file)

            # Convert column names to lowercase
            df.columns = df.columns.str.lower()

            # Ensure the CSV file has a 'reviews' column in lowercase
            if 'reviews' in df.columns:
                # Limit the DataFrame to the first 10 rows
                df = df.head(10)

                reviews = df['reviews'].str.lower().tolist()  # Convert reviews to lowercase

                # Initialize counters for sentiments
                positive_count = 0
                negative_count = 0
                neutral_count = 0

                # Lists to store highlighted phrases and their sentiment
                positive_phrases = set()
                negative_phrases = set()
                neutral_phrases = set()

                # Perform sentiment analysis on each review and extract phrases
                for review in reviews:
                    predicted_sentiment, sentiment_confidence = predict_sentiment(review)

                    # Extract and analyze phrases
                    review_phrases = list(extract_phrases(review)) + extract_phrases_nopunc(review)

                    for phrase in review_phrases:
                        phrase_sentiment = analyze_sentiment(phrase)

                        # Add phrases to the respective sets
                        if phrase_sentiment == 'Positive':
                            positive_phrases.add(phrase)
                        elif phrase_sentiment == 'Negative':
                            negative_phrases.add(phrase)
                        elif phrase_sentiment == 'Neutral':
                            neutral_phrases.add(phrase)

                    # Update sentiment counts
                    if predicted_sentiment == 'Positive':
                        positive_count += 1
                    elif predicted_sentiment == 'Negative':
                        negative_count += 1
                    else:
                        neutral_count += 1

                total_count = len(reviews)

                # Convert sets to lists before storing in the session
                session['positive_phrases'] = list(positive_phrases)
                session['negative_phrases'] = list(negative_phrases)
                session['neutral_phrases'] = list(neutral_phrases)

                return redirect(url_for('result_page', positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count, total_count=total_count))

    return redirect(url_for('error_page'))

@app.route('/error_page')
def error_page():
    return render_template('error_page.html')


@app.route('/result_page', methods=['GET'])
def result_page():
    positive_count = int(request.args.get('positive_count'))
    negative_count = int(request.args.get('negative_count'))
    neutral_count = int(request.args.get('neutral_count'))
    total_count = int(request.args.get('total_count'))

    # Retrieve the sets of phrases from the session and convert them back to sets
    positive_phrases = set(session.get('positive_phrases', []))
    negative_phrases = set(session.get('negative_phrases', []))
    neutral_phrases = set(session.get('neutral_phrases', []))

    # Create a dictionary of sentiments and their counts
    sentiments = {
        'Positive': positive_count,
        'Negative': negative_count,
        'Neutral': neutral_count,
    }

    # Sort the sentiments by count in descending order
    sorted_sentiments = sorted(sentiments, key=lambda x: sentiments[x], reverse=True)

    # Create a ranking dictionary
    ranking = {sentiment: i + 1 for i, sentiment in enumerate(sorted_sentiments)}

    # Determine the higher value and high label
    hvalue = sentiments[sorted_sentiments[0]]
    hlabel = sorted_sentiments[0]

    # Determine the second value and label
    svalue = sentiments[sorted_sentiments[1]]
    slabel = sorted_sentiments[1]

    # Determine the lowest value and label
    lvalue = sentiments[sorted_sentiments[2]]
    llabel = sorted_sentiments[2]

    return render_template('result_page.html', positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count, total_count=total_count, ranking=ranking, hvalue=hvalue, hlabel=hlabel, svalue=svalue, slabel=slabel, lvalue=lvalue, llabel=llabel, positive_phrases=positive_phrases, negative_phrases=negative_phrases, neutral_phrases=neutral_phrases)

if __name__ == '__main__':
    app.run(debug=True)
