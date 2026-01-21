import re
import string
from collections import Counter
from pathlib import Path
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import tqdm
import transformers
from datasets import Dataset
from gensim.models import Word2Vec
from keras.models import Model
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
# from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    BartModel,
    DataCollatorWithPadding,
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
        self.l1 = nn.Linear(768, 256)  # Intermediate layer
        self.l2 = nn.Linear(256, 2)  # 3 output classes
        self.softmax = nn.LogSoftmax(dim=1)
        #vocab of roberta
        self.embedding = nn.Embedding(50265, 768)        


    def forward(self, data):
        # features is the [0] token embedding from your pipeline
        # features = self.feature_extractor.model(
        #     input_ids=data["input_ids"].cuda(),
        #     attention_mask=data["attention_mask"].cuda(),
        # )
        embedded = self.embedding(data["input_ids"])

        
        # Use the attention_mask to ignore padding during averaging
        mask = data["attention_mask"].unsqueeze(-1).float() 
        masked_embedded = embedded * mask
        
        # Mean Pooling: Average the word vectors to get one sentence vector
        # [batch, embed_dim]
        sentence_vector = masked_embedded.sum(1) / mask.sum(1).clamp(min=1e-9)
        
        # Pass through MLP
        x = torch.relu(self.l1(sentence_vector))
        return self.softmax(self.l2(x))


def sentiment_analysis():
    # m = AutoModel.from_pretrained(name, use_safetensors=False)
    # tokenizer = AutoTokenizer.from_pretrained(name)

    pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        # in case there a longer text
        truncation=True,
        max_length=512,
    )

    train = pd.read_csv("train.csv")
    train.text = train.text.apply(clean)
    outputs = []
    for t in tqdm.tqdm(train.text):
        outputs.extend(pipe(t))

    preds = pd.DataFrame.from_records(outputs)
    y_pred = preds["label"] == "positive"
    y_gt = train["sentiment"] == "pos"
    print(accuracy_score(y_gt, y_pred))
    print(classification_report(y_gt, y_pred))
    """
                  precision    recall  f1-score   support

       False       0.72      0.90      0.80     12500
        True       0.86      0.65      0.74     12500

    accuracy                           0.77     25000
   macro avg       0.79      0.77      0.77     25000
weighted avg       0.79      0.77      0.77     25000

    """


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


def retrain_sentiment_analysis():
    name = "roberta-base"
    m = AutoModel.from_pretrained(name, use_safetensors=False)
    tokenizer = AutoTokenizer.from_pretrained(name)
    # pipe = pipeline("feature-extraction", model=m, tokenizer=tokenizer)
    df = pd.read_csv("train.csv")
    df.text.apply(clean)
    df.sentiment = df.sentiment.apply(lambda x: x == "pos")
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")
    # Create the DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    # still needs a tokenizer to substitute ids into words
    # pipeline("feature-extraction", model=m, tokenizer=tokenizer)
    pipe = SentimentHead().train()
    # Unfreeze all parameters
    for p in pipe.parameters():
        p.requires_grad = True
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn  = nn.NLLLoss()
    pbar = tqdm.tqdm(enumerate(train_dataloader))
    optimizer = torch.optim.AdamW(pipe.parameters(), lr=2e-3)
    acc = torchmetrics.Accuracy("multiclass", num_classes=10).cuda()
    metrics = MetricCollection([
        Accuracy(task="binary"),
        Precision(task="binary"),
        Recall(task="binary"),
        F1Score(task="binary")
    ])
    for epoch in range(1):
        for batch_idx,  data in pbar:

            pred = pipe(data)
            gt = torch.where(data.sentiment, 1, 0)
            # .to( torch.int32)
            optimizer.zero_grad()
            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss:.4f}", "status": "Training"})
            metrics( pred[:, 0], gt)
            if batch_idx % 100 == 0:
                print(metrics.compute())

        
        torch.save(pipe.state_dict(), f"sentiment_{epoch}.pth")




# sentiment_analysis()
retrain_sentiment_analysis()
# cls_token = transformer_embedding("roberta-base", z)
# "bert-base-uncased": TFBertModel,
# "roberta-base": TFRobertaModel,
# "xlnet-base-cased": TFXLNetModel,
# "facebook/bart-base": BartModel,
# "albert-base-v1": TFAlbertModel,
# "flaubert/flaubert_base_cased": FlaubertModel,
# "google/electra-small-discriminator": TFElectraModel,
# "allenai/longformer-base-4096": TFLongformerModel,
