
import pandas as pd
import re
import gensim.utils as gensim_utils
from gensim.models.phrases import Phrases, Phraser
import gensim.parsing.preprocessing as gpp
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# nltk.download('averaged_perceptron_tagger')
# nltk.download("wordnet")


class FDCPreprocess:

    def __init__(self):
        self.synonym_file = 'data/preprocess/synonyms.txt'
        self.add_stopwords_file = 'data/preprocess/stopwords_add.txt'
        self.rmv_stopwords_file = 'data/preprocess/stopwords_remove.txt'
        self.strip_short_size = 3   # remove word with length less than this

        # for Phrases generation
        self.generate_phrase = False
        self.min_count = 5
        self.threshold = 10.0   # minimum score to be phrase
        self.max_vocab_size = 40000000
        self.progress_per = 10000
        self.scoring = 'default'
        self.phrase_model_file = 'data/model/phrase_model.pkl'
        self.phrase_dump_file = 'output/phrases.txt'

    def preprocess_columns(self, pd_series, load_phrase_model=False):
        custom_filters = self.custom_filters()
        pd_series_processed = pd_series.apply(lambda x: gpp.preprocess_string(x, custom_filters), convert_dtype=False)
        pd_series_phrases = self.generate_phrases(pd_series_processed, load_phrase_model)
        return pd_series_phrases.apply(lambda x: ' '.join(x))

    def generate_phrases(self, pd_series, load_model=False):  # pd_series is series of list of words.
        sentences = pd_series.tolist()
        if not self.generate_phrase:
            print('Phrase detection skipped')
            return pd_series
        if load_model:
            model = Phraser.load(self.phrase_model_file)
            pd_series = pd_series.apply(lambda x: model[x], convert_dtype=False)
            return pd_series
        # detect bi-gram phrases
        model = Phrases(    # supplied parameters are default values in gensim
            sentences,  # list of list of words. (each sentence should be list of words)
            min_count=self.min_count,
            threshold=self.threshold,
            max_vocab_size=self.max_vocab_size,
            progress_per=self.progress_per,
            scoring=self.scoring)

        pd_series = pd_series.apply(lambda x: model[x], convert_dtype=False) # x is list of words. and adjacent words are concatenated by '_' in case of phrase
        model.save(self.phrase_model_file)
        phrase_score = []
        for phrase, score in model.export_phrases().items():    # export all found phrases
            phrase_score.append([phrase, score])
        df = pd.DataFrame(phrase_score, columns=['phrase', 'score'])
        df.drop_duplicates(subset='phrase', inplace=True)
        df.to_csv(self.phrase_dump_file, sep='\t', index=False)
        return pd_series


    def custom_filters(self):   # create list of filters to apply on string.
        filters = []
        filters.append(lambda x: x.lower())
        syn_dict = self.load_synonym_dict()
        filters.append(lambda x: self.replace_synonyms(x, syn_dict))
        filters.append(gpp.strip_punctuation)
        filters.append(gpp.strip_multiple_whitespaces)
        filters.append(gpp.strip_numeric)
        stopwords = self.generate_stopwords()
        filters.append(lambda x: self.remove_stopwords(x, stopwords))
        filters.append(lambda x: gpp.strip_short(x, minsize=self.strip_short_size))
        filters.append(self.lemmatize)
        return filters

    def load_synonym_dict(self):
        df = pd.read_csv(self.synonym_file, sep='\t', index_col='from')
        return df['to'].to_dict()

    def replace_synonyms(self, text, syn_dict): # replace all the keys in text by values in syn_dict
        pattern = '|'.join(r'\b%s\b' % re.escape(s) for s in syn_dict)  # take all keys as pattern
        return re.sub(pattern, lambda x: syn_dict[x.group(0)], text)    # substitute keys by values

    def generate_stopwords(self):
        # get gensim stopwords and add few custom stopwords, remove few custom stopwords.
        stopwords = list(gpp.STOPWORDS)

        with open(self.add_stopwords_file, 'r') as file:    # stopwords to add
            add_list = file.read().splitlines()
        stopwords.extend(add_list)

        with open(self.rmv_stopwords_file, 'r') as file: # stopwords to remove
            remove_list = file.read().splitlines()
        stopwords = [x for x in stopwords if x not in remove_list]
        return frozenset(stopwords)

    def remove_stopwords(self, string, stopwords):   # remove stopwords from string
        string = gensim_utils.to_unicode(string)
        return ' '.join(w for w in string.split() if w not in stopwords)

    def lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        tag_dict = {'V': wordnet.VERB, 'N': wordnet.NOUN, 'J': wordnet.ADJ, 'R': wordnet.ADV}   # gensim.utils lemmatizer uses only these four.
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(text))     # tag starts with 'N': noun, 'V': verb, 'J': adjective, 'R': adverb
        wordnet_tagged = [(x[0], tag_dict[x[1][0]]) for x in pos_tagged if x[1][0] in ['V', 'N', 'J', 'R']] # convert pos to wordnet tag only for (noun, verb, adj, adv). x[0] is word in tuple and x[1] is tag where x[1][0] is first char.
        return ' '.join([lemmatizer.lemmatize(word, tag) for word, tag in wordnet_tagged])

