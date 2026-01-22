import collections
from pathlib import Path

import nltk

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem import *

DATA = Path.home() / "data"
CONTRACTIONS = CONTRACTION_MAP = {
    "he's": "he is",
    "He's": "He is",
    "there's": "there is",
    "There's": "There is",
    "We're": "We are",
    "we're": "we are",
    "That's": "That is",
    "that's": "that is",
    "That\x89Ûªs": "That is",
    "won't": "will not",
    "they're": "they are",
    "They're": "They are",
    "Can't": "Cannot",
    "can't": "cannot",
    "Can\x89Ûªt": "Cannot",
    "can\x89Ûªt": "cannot",
    "wasn't": "was not",
    "don\x89Ûªt": "do not",
    "don't": "do not",
    "Don't": "do not",
    "DON'T": "DO NOT",
    "Don\x89Ûªt": "Do not",
    "donå«t": "do not",
    "aren't": "are not",
    "isn't": "is not",
    "Isn't": "is not",
    "What's": "What is",
    "what's": "what is",
    "haven't": "have not",
    "Haven't": "Have not",
    "hasn't": "has not",
    "It's": "It is",
    "it's": "it is",
    "it\x89Ûªs": "it is",
    "It\x89Ûªs": "It is",
    "You're": "You are",
    "you're": "you are",
    "You\x89Ûªre": "You are",
    "I'M": "I am",
    "I'm": "I am",
    "i'm": "I am",
    "I\x89Ûªm": "I am",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "wouldn\x89Ûªt": "would not",
    "Here's": "Here is",
    "Here\x89Ûªs": "Here is",
    "you've": "you have",
    "you\x89Ûªve": "you have",
    "youve": "you have",
    "We've": "We have",
    "we've": "we have",
    "I've": "I have",
    "i've": "I have",
    "I\x89Ûªve": "I have",
    "couldn't": "could not",
    "who's": "who is",
    "y'all": "you all",
    "Y'all": "You all",
    "would've": "would have",
    "it'll": "it will",
    "we'll": "we will",
    "he'll": "he will",
    "they'll": "they will",
    "you'll": "you will",
    "you\x89Ûªll": "you will",
    "I'll": "I will",
    "i'll": "I will",
    "Weren't": "Were not",
    "weren't": "were not",
    "Didn't": "Did not",
    "didn't": "did not",
    "they'd": "they would",
    "i'd": "I would",
    "I'd": "I would",
    "I\x89Ûªd": "I would",
    "we'd": "we would",
    "you'd": "You would",
    "should've": "should have",
    "where's": "where is",
    "let's": "let us",
    "Let's": "Let us",
    "doesn't": "does not",
    "doesn\x89Ûªt": "does not",
    "ain't": "am not",
    "Ain't": "am not",
    "Could've": "Could have",
}


def setup_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
    ]

    for path, name in resources:
        try:
            # Check if the resource exists
            nltk.data.find(path)
        except LookupError:
            # If not found, download it
            print(f"Downloading {name}...")
            nltk.download(name)


def load_from_files():
    subfolder = DATA / "aclImdb"
    train_df = pd.read_csv(
        subfolder / "imdbEr.txt", header=None, names=["sentiment_score"]
    )
    train_df.head()
    reviews = collections.defaultdict(list)
    for split in ["train", "test"]:
        for sentiment in ["pos", "neg"]:
            for file in (subfolder / split / sentiment).iterdir():
                file.read_text()
                id, rating = file.stem.split("_")
                reviews[split].append(
                    {
                        "text": file.read_text(),
                        "id": id,
                        "rating": rating,
                        "sentiment": sentiment,
                    }
                )

    train, test = reviews["train"], reviews["test"]
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    # needs downloading
    # you should not remove stop words for sentiment analysis
    # 'not' should not be removed, changes the meaning
    stop_words = stopwords.words("english")

    # Count of good and bad reviews
    count = train["sentiment"].value_counts()
    print("Total Counts of both sets".format(), count)
    print("==============")
    return train, test


# setup_nltk_resources()
