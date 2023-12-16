from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from data_preprocess import text_cleaner
import joblib
import numpy as np
import gdown

# Google Drive links for the files
google_drive_link_vectorizer = "https://drive.google.com/uc?id=1EzHFwNxd1FXUvZ8sArzX0moWrhA0fdHP"
google_drive_link_svm_classifier = "https://drive.google.com/uc?id=1ABWUGve7-HnrITGXyiQ2GpGj0yEXaOxm"

# Destination paths for saving the downloaded files
destination_path_vectorizer = "tfidf_vectorizer.joblib"
destination_path_svm_classifier = "SVM_classifier.joblib"

# Download the files from Google Drive
gdown.download(google_drive_link_vectorizer, destination_path_vectorizer, quiet=False)
gdown.download(google_drive_link_svm_classifier, destination_path_svm_classifier, quiet=False)

# Load the vectorizer and SVM classifier using joblib
vectorizer = joblib.load(destination_path_vectorizer)
svm_classifier = joblib.load(destination_path_svm_classifier)



def predict_sentiment(user_input):
    clean_input = text_cleaner(user_input)

    input_vectorized = vectorizer.transform([clean_input])
    main_prediction = svm_classifier.predict(input_vectorized)

    # Calculate decision scores for each class
    decision_scores = svm_classifier.decision_function(input_vectorized)

    # Calculate confidence levels using softmax
    confidence_levels = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)

    # Convert confidence levels to percentages and round to whole numbers
    confidence_percentages = (confidence_levels * 100).round().astype(int).tolist()

    return main_prediction[0], confidence_percentages[0]


'''' Example usage
user_input = "I didn't like the movie but I enjoyed the popcorn"
predicted_sentiment, (negative_confidence, neutral_confidence, positive_confidence) = predict_sentiment(user_input)

print("Predicted Sentiment:", predicted_sentiment)
print("Negative Confidence (%):", negative_confidence)
print("Neutral Confidence (%):", neutral_confidence)
print("Positive Confidence (%):", positive_confidence)
'''
