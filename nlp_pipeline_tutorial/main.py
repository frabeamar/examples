import os
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir= /usr/lib/nvidia-cuda-toolkit"
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf
import transformers
from gensim.models import Word2Vec
from keras.models import Model

# from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import *
from pytorch_pretrained_bert import BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import (
    BartModel,
    BertTokenizer,
    FlaubertModel,
    TFAlbertModel,
    TFAutoModel,
    TFBertModel,
    TFElectraModel,
    TFLongformerModel,
    TFOpenAIGPTModel,
    TFRobertaModel,
    TFXLNetModel,
    pipeline,
)

from visualization import DimRed, dimen_reduc_plot, display_cloud

# fixes problems with my gpu


def initial_analysis(df: pd.DataFrame):
    df = df.assign(word_count=df.text.apply(lambda x: len(x.split(" "))))
    sns.displot(df, hue="sentiment", x="word_count", kind="hist").savefig("word_count")

    df = df.assign(
        punctuation_count=df.text.apply(
            lambda x: sum(1 for char in x if char in string.punctuation)
        )
    )
    # punctuations
    sns.displot(df, hue="sentiment", x="punctuation_count", kind="hist").savefig(
        "punctuation_count"
    )

    stop_words = stopwords.words("english")
    df = df.assign(
        stop_words=df.text.apply(
            lambda x: sum(1 for word in x.lower().split() if word in stop_words)
        )
    )
    sns.displot(df, hue="sentiment", x="stop_words", kind="hist").savefig("stop_words")

    display_cloud(train, stop_words, Path("pos_cloud.png"))

    # most common
    corpus = " ".join(df.text).split()
    cnt = Counter(corpus)
    y, x = zip(*cnt.most_common()[:10])
    sns.barplot(x=x, y=y)
    plt.savefig("most_common")
    plt.close()

    skip_stops = [c for c in corpus if c not in stop_words]
    cnt = Counter(skip_stops)
    y, x = zip(*cnt.most_common()[:10])
    sns.barplot(x=x, y=y)
    plt.savefig("most_common_no_skips")
    plt.close()


def gram_analysis(df: pd.DataFrame, gram: int):
    corpus = " ".join(df.text).split()
    stop_words = stopwords.words("english") + [""]

    tokens = [c for c in corpus if c not in stop_words]

    ngrams = zip(*[tokens[i:] for i in range(gram)])
    final_tokens = [" ".join(z) for z in ngrams]
    cnt = Counter(final_tokens)
    y, x = zip(*cnt.most_common()[:10])
    sns.barplot(x=x, y=y)
    plt.savefig("grams")
    plt.close()


def clean(text: str):
    # punctuation
    punct_tag = re.compile(r"[^\w\s]")
    data = punct_tag.sub(r"", text)

    # remove html
    html_tag = re.compile(r"<.*?>")
    data = html_tag.sub(r"", data)

    url_clean = re.compile(r"https://\S+|www\.\S+")
    data = url_clean.sub(r"", data)

    # Removes Emojis
    emoji_clean = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    data = emoji_clean.sub(r"", data)
    return data


def tf_idf_vectorizer(df: pd.DataFrame):
    """
    term frequence = how often a word is present in a specific document
    idf = inverse document frequency, how often a word is there in the entire corpus
    metric = tf x idf
    (this penalizes words which happen often)
    """
    tfidf_vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    train_tfidf = tfidf_vect.fit_transform(df.text.values)


def vectorize(data):
    cv = CountVectorizer()
    fit_data_cv = cv.fit_transform(data)
    return fit_data_cv, cv


def visualize_in_low_dim(df: pd.DataFrame):
    cv = CountVectorizer()
    compressed = cv.fit_transform(df.text)
    dimen_reduc_plot(compressed, df.sentiment, DimRed.PCA)


def word2vec(df: pd.DataFrame):
    # shallow two layer network; trained by predicting the next word;
    # or predicting the context

    model = Word2Vec(" ".join(df.text).split(), min_count=5)
    # slow af
    word_li = list(model.wv.vocab)


def bert_encode(data, maximum_length):
    input_ids = []
    attention_masks = []
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-large-uncased", do_lower_case=True
    )
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        # id in the vocabulary

        input_ids.append(encoded["input_ids"])
        # whether its a padded token or a real one
        attention_masks.append(encoded["attention_mask"])
    return np.array(input_ids), np.array(attention_masks)


def model_summary():
    bert_model = transformers.TFBertModel.from_pretrained("bert-large-uncased")

    # Build a miniature model for extracting the embeddings
    input_ids = tf.keras.layers.Input(shape=(128,), name="input_token", dtype="int32")
    input_masks_ids = tf.keras.layers.Input(
        shape=(128,), name="masked_token", dtype="int32"
    )
    bert_output = bert_model([input_ids, input_masks_ids])[0]
    bert_output.shape
    bert_output[:, 0, :]
    model = Model(inputs=[input_ids, input_masks_ids], outputs=[bert_output])
    model.summary()


def get_embeddings(model: TFBertModel, tokenizer, model_name: str, inp):
    from transformers import AutoTokenizer
    tokenizer = tokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(
        model_name, use_safetensors=False, force_download=True
    )
    # m = TFBertModel.from_pretrained(model_name, use_safetensors=False)
    # safetensors introduced to replaces pickle (safer option)
    input_ids = tf.constant(tokenizer.encode(inp))[None, :]  # Batch size 1
    outputs = m(input_ids)
    last_hidden_states = outputs[0]
    cls_token = last_hidden_states[0]
    return cls_token


MODELS = {
    "bert-base-uncased": TFBertModel,
    "roberta-base": TFRobertaModel,
    "xlnet-base-cased": TFXLNetModel,
    "facebook/bart-base": BartModel,
    "albert-base-v1": TFAlbertModel,
    "flaubert/flaubert_base_cased": FlaubertModel,
    "google/electra-small-discriminator": TFElectraModel,
    "allenai/longformer-base-4096": TFLongformerModel,
}


GPT_MODELS = {"openai-gpt": TFOpenAIGPTModel}


def transformer_embedding(name, inp):
    m = AutoModel.from_pretrained(name, use_safetensors=False)
    tokenizer = AutoTokenizer.from_pretrained(name)
    pipe = pipeline("feature-extraction", model=m, tokenizer=tokenizer)
    features = pipe(inp)
    features = np.squeeze(features)
    return features


def transformer_gpt_embedding(name, inp, model_name):
    model = model_name.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = "[PAD]"
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    features = pipe(inp)
    features = np.squeeze(features)
    return features


class SentimentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(768, 256) # Intermediate layer
        self.l2 = nn.Linear(256, 3)   # 3 output classes
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, features):
        # features is the [0] token embedding from your pipeline
        x = torch.relu(self.l1(features))
        return self.softmax(self.l2(x))

def sentiment_analysis():
    # m = AutoModel.from_pretrained(name, use_safetensors=False)
    # tokenizer = AutoTokenizer.from_pretrained(name)

    pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    train = pd.read_csv("train.csv")
    outputs = []
    for t in train.text:
        outputs.extend(pipe(T))

    preds = pd.DataFrame.from_records(outputs)
    y_pred =  preds["label"] == "positive"
    y_gt = train["sentiment"] == "pos"
    print(accuracy_score(y_gt, y_pred))
    print(classification_report(y_gt, y_pred))


# cls_token = transformer_embedding("roberta-base", z)
# "bert-base-uncased": TFBertModel,
# "roberta-base": TFRobertaModel,
# "xlnet-base-cased": TFXLNetModel,
# "facebook/bart-base": BartModel,
# "albert-base-v1": TFAlbertModel,
# "flaubert/flaubert_base_cased": FlaubertModel,
# "google/electra-small-discriminator": TFElectraModel,
# "allenai/longformer-base-4096": TFLongformerModel,
