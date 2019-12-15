import pandas as pd
import os
import yaml
import sys
sys.path.append(os.getcwd().replace('src', 'data_processing'))
from data_processing_functions import *


def main():
    #Setup paths
    path_raw = os.getcwd().replace('src', 'data/raw/')
    path_config = os.getcwd().replace('src', 'configs/config_1.yaml')

    # Read files
    #Reading config
    with open(path_config) as file:
        functions = yaml.load(file, Loader=yaml.FullLoader)

    #Reading data files
    #  quoting=3 tells Python to ignore doubled quote
    train = pd.read_csv(path_raw + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # Initialize an empty list to hold the clean reviews
    clean_data_train = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list 
    for i in xrange( 0, train["review"].size ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        review = train["review"][i]
        for function in functions['data_processing']:
            review = eval(str(function)(review))
        clean_data_train.append(review)
    
    # Transform the text into a vector
    data_train_vector = eval(functions['transform_2_vector'])(clean_data_train, max_features = 8000)


if __name__ == '__main__':
    main()