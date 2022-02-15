"""
Download glove word embeddings and convert then to get word2vec embeddings.
"""
import io
import requests
import zipfile
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec_inner import REAL
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pylab as plt

class WordEmbeddings:
    def __init__(self):
        self.file_dir = 'data/pretrain/'
        self.glove_files = ['glove.6B.300d.txt']
        self.word2vec_files = ['word2vec.6B.300d.txt']
        for i in range(len(self.glove_files)):
            self.glove_files[i] = self.file_dir+self.glove_files[i]
            self.word2vec_files[i] = self.file_dir+self.word2vec_files[i]

        # for training word embeddings
        self.summary_preprocessed_file = 'data/wikipedia/preprocessed_new.txt'
        self.summary_prep_col = 'summary_preprocessed'

    def download_glove(self):
        glove_zip_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        r = requests.get(glove_zip_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(self.file_dir)
        return None

    def convert_glove_to_word2vec(self):
        for g_file, w_file in zip(self.glove_files, self.word2vec_files):
            glove2word2vec(g_file, w_file)
        return None

    def get_pretrain_vectors(self):
        # self.download_glove()         # uncomment if glove vectors are not downloaded.
        self.convert_glove_to_word2vec()
        return None

    def train_embeddings(self):
        df_summary = pd.read_csv(self.summary_preprocessed_file, sep='\t')
        df_summary.fillna('', inplace=True)
        df_summary = df_summary[df_summary[self.summary_prep_col] != '']
        sents = df_summary[self.summary_prep_col].tolist()
        sents = [sent.split() for sent in sents]
        print(f'Total sentences in wiki corpus: {len(sents)}')
        we_trainer = Word2vecTrainer()
        we_trainer.train(sents, self.word2vec_files[-1])    # using 300d word2vec file as pretrained vectors.
        we_trainer.save_model()
        we_trainer.save_vectors()
        we_trainer.save_loss()
        return None


class Word2vecTrainer:
    def __init__(self):
        self.callback = CallbackOnEpoch()
        self.model = None
        self.epochs = 250
        self.vector_size = 300
        self.window = 10
        self.min_count = 1
        self.workers = 16
        self.vectors_save_file = 'data/model/word2vec_trained_new.txt'
        self.model_save_file = 'data/model/word2vec_model.model'
        self.loss_save_file = 'data/model/train_loss.pdf'

    def train(self, sentences, pretrained_file=None):   # retraining of pretrained vectors is not possible in gensim 4 word2vec model.
        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers)
        self.model.build_vocab(sentences) # initially update=False. words satisfying criteria added to vocab not all.
        total_sents = self.model.corpus_count
        self.model.wv.vectors_lockf = np.ones(len(self.model.wv), dtype=REAL)   # gensim 4. BUG. initialize array manually for lock.

        if pretrained_file:
            vocab = self.model.wv.key_to_index.keys()
            pretrained_vocab = KeyedVectors.load_word2vec_format(pretrained_file).key_to_index.keys()
            common_words = list(set(vocab) & set(pretrained_vocab))
            print('Intersecting %d common vocabularies for transfer learning', len(common_words))
            # self.model.build_vocab([pretrained_vocab], update=True, min_count=1)  # update=True: add new words and initialize their vectors, min_count=1: include every word because pretrained_vocab is list of unique words

            self.model.wv.intersect_word2vec_format(pretrained_file, lockf=1.0) # intersecting word vectors are replaced by pretrained vectors. lockf=1 means allow imported vectors to be trained. lockf=0 imported vectors are not trained.
            # TODO: before training on wiki train on brown etc. corpus and some food related corpus.
            # TODO: lock vectors then train and then unlock(load again) and train so that remaining vectors are trained according to pretrained and then every vectors trained.
        self.model.train(
            sentences,
            total_examples=total_sents,
            epochs=self.epochs,
            compute_loss=True,
            callbacks=[self.callback])


    def save_model(self):
        self.model.save(self.model_save_file)

    def save_vectors(self):
        self.model.wv.save_word2vec_format(self.vectors_save_file)

    def save_loss(self):
        lists = sorted(self.callback.loss.items())  # sorted by key, return a list of tuples
        x, y = zip(*lists)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(self.loss_save_file)


class CallbackOnEpoch(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss = {}
        self.previous_loss = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            actual_loss = loss
        else:
            actual_loss = loss - self.previous_loss

        print('Loss after epoch %d: %d', self.epoch, actual_loss)
        self.loss[self.epoch] = actual_loss
        self.previous_loss = loss
        self.epoch += 1


if __name__ == '__main__':
    we = WordEmbeddings()
    # we.get_pretrain_vectors()
    we.train_embeddings()