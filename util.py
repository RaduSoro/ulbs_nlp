import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import xml.etree.ElementTree as ET
from nltk.tokenize import WhitespaceTokenizer
import json
import glob
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
from nltk.tokenize import TweetTokenizer

Tokenizer = TweetTokenizer()
punctuation_tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()

def save_json_data(filename, data):
    with open(filename, 'w', ) as f:
        json.dump(data, f, indent=4)


def extract_topics(filepath):
    topics = []
    tree = ET.parse(filepath)
    for element in tree.find("//codes[@class='bip:topics:1.0']"):
        topics.append(element.attrib['code'])
    return topics


def extract_title(filepath):
    tree = ET.parse(filepath)
    return tree.find("title").text


def extract_body(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    found_text = ""
    for child in root.findall("text"):
        for element in child:
            found_text += element.text + "\n"
    return found_text


def get_small_files():
    return glob.glob("./data/Reuters_34/Training/*")

def get_big_files():
    return glob.glob("./data/Reuters_7083/Training/*")


def process_sentece(data):
    # make all text lowerCase
    data = data.lower()
    data = Tokenizer.tokenize(data)
    stop = stopwords.words('english')
    # remove stopwords
    data = [item for item in data if item not in stop]
    # do the lemmatization
    data = [lemmatizer.lemmatize(y) for y in data]
    # do the stemming
    data = [stemmer.stem(y) for y in data]
    # remove punctuation:
    data = punctuation_tokenizer.tokenize(' '.join(data))
    # remove numbers
    data_w_o_num = []
    for item in data:
        if not re.match("\d+",item):
            data_w_o_num.append(item)
    data_w_o_num = ' '.join(data_w_o_num)
    return data_w_o_num