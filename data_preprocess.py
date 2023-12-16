import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)


stop_words = set(stopwords.words('english'))
negative_words = {'no', 'not', "never", "neither", "nor", "none", "nobody", "nowhere", "nothing", "hardly", "scarcely"}
stop_words -= negative_words # Remove the Negative words


import contractions
import re

def text_cleaner(text):
    text = str(text).lower()                   # Convert to string and lowercase
    text = contractions.fix(text)              # Expand Contractions
    text = re.sub(r'<.*?>', ' ', text)         # Remove HTML tags
    text = re.sub(r'http\S+', ' ', text)       # Remove URLs using regular expression not covered by tags
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text) # Remove non-alphanumeric characters using regular expression
    words = text.split()                       # Tokenize for stop word removal
    cleaned_text = " ".join(word for word in words if word not in stop_words) # Remove the stopwords from the text.
    print(cleaned_text)
    return cleaned_text

