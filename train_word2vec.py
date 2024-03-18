import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def tokenize_emails(emails):
    return [word_tokenize(email.lower()) for email in emails]


def train_word2vec(tokenized_emails, vector_size=1000, window=5, min_count=1, workers=4):
    w2v_model = Word2Vec(sentences=tokenized_emails, vector_size=vector_size, window=window, min_count=min_count,
                         workers=workers)
    w2v_model.save("email_word2vec.model")
    return w2v_model
