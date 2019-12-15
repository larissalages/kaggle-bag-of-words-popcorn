from bs4 import BeautifulSoup
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