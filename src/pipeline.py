"""
Usage:
    pipeline.py --sdc string1 --sdt string2

Options:
    -sc string1 --sdc=string1           Print all the things.
    -st string2 --sdt=string2           Get more bees into the path.
"""

import pandas as pd
import os
import yaml
import sys
sys.path.append(os.getcwd().replace('src', 'data_processing'))
from data_processing_functions import *
import logging
from docopt import docopt
from pprint import pprint


def main():
    saved_data_clean = False
    saved_data_transformed = False

    logging.basicConfig(level=logging.INFO)

    #Setup paths
    path_raw = os.getcwd().replace('src', 'data/raw/')
    path_config = os.getcwd().replace('src', 'configs/config_1.yaml')
    path_modified_data = os.getcwd().replace('src', 'data/processed_data')

    # Read files
    #Reading config
    with open(path_config) as file:
        functions = yaml.load(file, Loader=yaml.FullLoader)

    #Reading data files
    #  quoting=3 tells Python to ignore doubled quote
    train = pd.read_csv(path_raw + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    logging.info("Cleaning the data")
    if saved_data_clean == False:
        # Initialize an empty list to hold the clean reviews
        clean_data_train = []
        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list 
        for i in xrange( 0, train["review"].size ):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            review = train["review"][i]
            for function in functions['data_processing']:
                review = eval(str(function))(review)
            clean_data_train.append(review)
        pd.DataFrame(clean_data_train).to_csv('data_train_cleaned.csv', index = None, header=False)
    else:
        clean_data_train = list(pd.read_csv('data_train_cleaned.csv'))
    # Transform the text into a vector
    logging.info("Transforming text into a vector")
    data_train_vector = eval(functions['transform_2_vector'])(clean_data_train, max_features = 8000)


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0.0')
    args = {k.replace('-', ''): v for k, v in args.items()}
    pprint(args)

    main()