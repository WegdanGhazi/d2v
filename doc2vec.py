import multiprocessing

import gensim
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class Doc2vecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, dm=0, dbow_words=1, vector_size=1000, window=3, min_count=1, sample=0, hs=0, negative=5,
                                                train_epochs=300, train_start_alpha=0.025, train_end_alpha=0.0001,
                                                                re_infer=True, infer_steps=20, infer_alpha=0.025):
        # Multi-processing assertion for multi-core usage
        cores = multiprocessing.cpu_count()
        assert gensim.models.doc2vec.FAST_VERSION > -1
        # Model parameters
        self.dm = dm
        self.dbow_words = dbow_words
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.hs = hs
        self.negative = negative
        # Training parameters
        self.train_epochs = train_epochs
        self.train_start_alpha = train_start_alpha
        self.train_end_alpha = train_end_alpha
        # Re-infer trained data
        self.re_infer = re_infer
        # Inference parameters
        self.infer_steps = infer_steps
        self.infer_alpha = infer_alpha

        self.model = Doc2Vec(dm=dm, dbow_words=dbow_words, vector_size=vector_size,  window=window,
                             min_count=min_count, sample=sample, workers=cores, hs=hs, negative=negative)

    def fit(self, X_train, y=None):
        taggedDocs = []
        for i, doc in enumerate(X_train):
            taggedDocs.append(TaggedDocument(doc, [str(i)]))
        self.model.build_vocab(taggedDocs)
        epochs = self.train_epochs
        start_alpha = self.train_start_alpha
        end_alpha = self.train_end_alpha
        self.model.train(taggedDocs, total_examples=len(
            taggedDocs), epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)

    def fit_transform(self, X_train, y=None):
        taggedDocs = []
        for i, doc in enumerate(X_train):
            taggedDocs.append(TaggedDocument(word_tokenize(doc), [str(i), doc]))
        self.model.build_vocab(taggedDocs)
        epochs = self.train_epochs
        start_alpha = self.train_start_alpha
        end_alpha = self.train_end_alpha
        self.model.train(taggedDocs, total_examples=len(
            taggedDocs), epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)
        
        # Re-infer training data or return ready-trained vocab
        if self.re_infer:
            return self.transform(X_train)
        else:
            return self.retrive_pretrained_vecs(self.model.docvecs, self.vector_size)

    def transform(self, X_test, y=None):
        vecs = []
        steps = self.infer_steps
        alpha = self.infer_alpha
        for words in X_test:
            vec = self.model.infer_vector(
                doc_words=word_tokenize(words), steps=steps, alpha=alpha)
            vecs.append(vec)
        return vecs

    def retrive_pretrained_vecs(self, doc_vecs, n):
        vecs = np.array([]).reshape(0, n)
        for i, doc in enumerate(doc_vecs):
            vecs = np.vstack((vecs, doc))
            if i == len(doc_vecs) - 1:
                break
        return vecs