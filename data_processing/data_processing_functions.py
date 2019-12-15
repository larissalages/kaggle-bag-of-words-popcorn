from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import nltk
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords
import re

# A collection of usefull functions to Data Cleaning and Text Preprocessing

def remove_HTML_Markup(raw_review):
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    return review_text

def remove_non_letters(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    return letters_only

def remove_stop_words(review_text):
    # 3. Convert to lower case, split into individual words
    words = review_text.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))     

def bag_of_words(clean_train_reviews, max_features = 5000):
    print "Creating the bag of words...\n"

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                tokenizer = None,    \
                                preprocessor = None, \
                                stop_words = None,   \
                                max_features = max_features) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()
    return train_data_features