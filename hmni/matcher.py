# HMNI (Hello my name is)
# Fuzzy Name Matching with Machine Learning
# Author: Christopher Thornton (christopher_thornton@outlook.com)
# 2020-2020
# MIT Licence

import os
import re
import heapq
import joblib
import unidecode
import numpy as np
import pandas as pd
from random import randint

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from fuzzywuzzy import fuzz

from collections import Counter
from hmni import syllable_tokenizer
from hmni import input_helpers
from hmni import preprocess
import tarfile

from abydos.phones import *
from abydos.phonetic import PSHPSoundexFirst, PSHPSoundexLast, Ainsworth
from abydos.distance import (IterativeSubString, BISIM, DiscountedLevenshtein, Prefix, LCSstr, MLIPNS, Strcmp95,
                             MRA, Editex, SAPS, FlexMetric, JaroWinkler, HigueraMico, Sift4, Eudex, ALINE, Covington,
                             PhoneticEditDistance)

import logging

logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import sys

sys.modules['syllable_tokenizer'] = syllable_tokenizer
sys.modules['input_helpers'] = input_helpers
sys.modules['preprocess'] = preprocess


# guard against uncomputable recursion with max name length
class CovingtonGuard(Covington):
    def dist(self, src, tar, max_length=11):
        src = src[:max_length]
        tar = tar[:max_length]
        normalizer = self._weights[5] * min(len(src), len(tar))
        if len(src) != len(tar):
            normalizer += self._weights[7]
        normalizer += self._weights[6] * abs(abs(len(src) - len(tar)) - 1)
        return self.dist_abs(src, tar) / normalizer


class Matcher:

    def __init__(self, model='latin', prefilter=True, allow_alt_surname=True, allow_initials=True,
                 allow_missing_components=True):

        # user-provided parameters
        self.model = model
        self.allow_alt_surname = allow_alt_surname
        self.allow_initials = allow_initials
        self.allow_missing_components = allow_missing_components
        self.prefilter = prefilter
        if self.prefilter:
            self.refined_soundex = {
                'b': 1, 'p': 1,
                'f': 2, 'v': 2,
                'c': 3, 'k': 3, 's': 3,
                'g': 4, 'j': 4,
                'q': 5, 'x': 5, 'z': 5,
                'd': 6, 't': 6,
                'l': 7,
                'm': 8, 'n': 8,
                'r': 9
            }

        # verify user-supplied class arguments
        model_dir = self.validate_parameters()

        self.impH = input_helpers.InputHelper()
        # Phonetic Encoder
        self.pe = Ainsworth()
        # Soundex Firstname Algorithm
        self.pshp_soundex_first = PSHPSoundexFirst()
        # Soundex Lastname Algorithm
        self.pshp_soundex_last = PSHPSoundexLast()

        # String Distance algorithms
        self.algos = [IterativeSubString(), BISIM(), DiscountedLevenshtein(), Prefix(), LCSstr(), MLIPNS(),
                      Strcmp95(), MRA(), Editex(), SAPS(), FlexMetric(), JaroWinkler(mode='Jaro'), HigueraMico(),
                      Sift4(), Eudex(), ALINE(), CovingtonGuard(), PhoneticEditDistance()]
        self.algo_names = ['iterativesubstring', 'bisim', 'discountedlevenshtein', 'prefix', 'lcsstr', 'mlipns',
                           'strcmp95', 'mra', 'editex', 'saps', 'flexmetric', 'jaro', 'higueramico',
                           'sift4', 'eudex', 'aline', 'covington', 'phoneticeditdistance']

        # String Distance Pipeline (Level 0/Base Model)
        self.baseModel = joblib.load(os.path.join(model_dir, 'base.pkl'))

        # Character Embedding Network (Level 0/Base Model)
        self.vocab = preprocess.VocabularyProcessor(max_document_length=15, min_frequency=0).restore(
            os.path.join(model_dir, 'vocab'))

        siamese_model = os.path.join(model_dir, 'siamese')

        # start tensorflow session
        graph = tf.Graph()
        with graph.as_default() as graph:
            self.sess = tf.Session() if tf.__version__[0] == '1' else tf.compat.v1.Session()
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                if tf.__version__[0] == '1':
                    saver = tf.train.import_meta_graph('{}.meta'.format(siamese_model))
                    self.sess.run(tf.global_variables_initializer())
                else:
                    saver = tf.compat.v1.train.import_meta_graph('{}.meta'.format(siamese_model))
                    self.sess.run(tf.compat.v1.global_variables_initializer())
                saver.restore(self.sess, siamese_model)
                # Get the placeholders from the graph by name
            self.input_x1 = graph.get_operation_by_name('input_x1').outputs[0]
            self.input_x2 = graph.get_operation_by_name('input_x2').outputs[0]

            self.dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            self.prediction = graph.get_operation_by_name('output/distance').outputs[0]
            self.sim = graph.get_operation_by_name('accuracy/temp_sim').outputs[0]

        # Logreg (Level 1/Meta Model)
        self.metaModel = joblib.load(os.path.join(model_dir, 'meta.pkl'))

        # seen names (mapping dict from raw name to processed name)
        self.seen_names = {}
        # seen pairs (mapping dict from name pair tuple to similarity)
        self.seen_pairs = {}
        # user scores (mapping dict from name pair tuple to similarity)
        self.user_scores = {}

    def validate_parameters(self):
        # extract model tarball into directory if doesnt exist
        model_dir = os.path.join(os.path.dirname(__file__), "models", self.model)
        if not os.path.exists(model_dir):
            tar = tarfile.open(os.path.join(os.path.dirname(__file__), "models", self.model + ".tar.gz"), "r:gz")
            os.makedirs(model_dir)
            tar.extractall(model_dir)
            tar.close()
        return model_dir

    def output_sim(self, sim, prob, threshold):
        if prob:
            return sim
        return 1 if sim >= threshold else 0

    def assign_similarity(self, name_a, name_b, score):
        if not (isinstance(name_a, str) and isinstance(name_b, str)):
            raise TypeError('Only strings are supported in add_score method')
        if score < 0 or score > 1:
            raise ValueError('Score must be a number between 0 and 1 (inclusive)')
        pair = tuple(sorted((name_a.lower().strip(), name_b.lower().strip()),
                            key=lambda item: (-len(item), item)))
        self.user_scores[hash(pair)] = score

    def similarity(self, name_a, name_b, prob=True, threshold=0.5, surname_first=False):
        # input validation
        if not (isinstance(name_a, str) and isinstance(name_b, str)):
            raise TypeError('Only string comparison is supported in similarity method')

        if len(self.user_scores) != 0:
            # return user score if match
            pair = tuple(sorted((name_a.lower().strip(), name_b.lower().strip()),
                                key=lambda item: (-len(item), item)))
            score = self.seen_set(pair, self.user_scores)
            if score is not None:
                return self.output_sim(score, prob=prob, threshold=threshold)

        # empty or single character string returns 0
        if len(name_a) < 2 or len(name_b) < 2:
            return 0

        # exact match returns 1
        if name_a == name_b:
            return 1

        # preprocess names
        name_a = self.preprocess(name_a)
        name_b = self.preprocess(name_b)

        # empty or single character string returns 0
        if len(name_a) == 0 or len(name_b) == 0:
            return 0

        # check for missing name components
        missing_component = False
        one_component = False
        if len(name_a) == 1 and len(name_b) == 1:
            one_component = True
        elif len(name_a) == 1 or len(name_b) == 1:
            if not self.allow_missing_components:
                return 0
            missing_component = True

        if surname_first:
            fname_a, lname_a, fname_b, lname_b = name_a[-1], name_a[0], name_b[-1], name_b[0]
        else:
            fname_a, lname_a, fname_b, lname_b = name_a[0], name_a[-1], name_b[0], name_b[-1]

        # check for initials in first and lastnames
        if not self.allow_initials and any(len(x) == 1 for x in [fname_a, lname_a, fname_b, lname_b]):
            return 0

        # lastname conditions
        initial_lname = False
        if len(lname_a) == 1 or len(lname_b) == 1:
            if lname_a[0] != lname_b[0] and not missing_component:
                return 0
            initial_lname = True
        elif not one_component:
            if self.allow_alt_surname:
                if self.pshp_soundex_last.encode(lname_a) != self.pshp_soundex_last.encode(lname_b):
                    if not missing_component:
                        return 0
                elif missing_component:
                    return self.output_sim(0.5, prob=prob, threshold=threshold)
            elif lname_a != lname_b and not missing_component:
                return 0
            elif missing_component:
                return self.output_sim(0.5, prob=prob, threshold=threshold)

        # check initial match in firstname
        if len(fname_a) == 1 or len(fname_b) == 1:
            if fname_a[0] == fname_b[0]:
                return self.output_sim(0.5, prob=prob, threshold=threshold)
            return 0

        # check if firstname is same
        if fname_a == fname_b:
            if not missing_component and not initial_lname:
                return 1
            return self.output_sim(0.5, prob=prob, threshold=threshold)

        # sort pair to normalize
        pair = tuple(sorted((fname_a, fname_b), key=lambda item: (-len(item), item)))

        # prefilter candidates using heuristics on firstname
        if self.prefilter and not missing_component and pair[0][0] != pair[1][0]:
            encoded1 = set(self.refined_soundex.get(c) for c in set(pair[0][1:]))
            encoded2 = set(self.refined_soundex.get(c) for c in set(pair[1][1:]))
            encoded1.discard(None)
            encoded2.discard(None)
            if encoded1.isdisjoint(encoded2):
                return 0

        # return pair score if seen
        seen = self.seen_set(pair, self.seen_pairs)
        if seen is not None:
            if initial_lname:
                seen = min(0.5, seen)
            return self.output_sim(seen, prob=prob, threshold=threshold)

        # generate features for base-level model
        features = self.featurize(pair)
        # make inference on meta model
        sim = self.meta_inf(pair, features)

        if not missing_component:
            # add pair score to the seen dictionary
            self.seen_pairs[hash(pair)] = sim

        if initial_lname:
            sim = min(0.5, sim)

        return self.output_sim(sim, prob=prob, threshold=threshold)

    def fuzzymerge(self, df1, df2, how='inner', on=None, left_on=None, right_on=None, indicator=False,
                   limit=1, threshold=0.5, allow_exact_matches=True, surname_first=False):
        # parameter validation
        if not isinstance(df1, pd.DataFrame):
            df1 = pd.DataFrame(df1)
        if not isinstance(df2, pd.DataFrame):
            df2 = pd.DataFrame(df2)

        if not (0 < threshold < 1):
            raise ValueError('threshold must be decimal number between 0 and 1 (given = {})'.format(threshold))
        if how.lower() == 'right':
            df1, df2 = df2, df1
            left_on, right_on = right_on, left_on
            how = 'left'

        if on is None:
            k1, k2 = left_on, right_on
        else:
            k1, k2 = on, on
            right_on = on

        key = 'key'
        # if name key in columns - generate random integer
        while key in df1.columns:
            key = str(randint(1, 1000000))

        df1[key] = df1[k1].apply(
            lambda x: self.get_top_matches(x, df2[k2], limit=limit, thresh=threshold,
                                           exact=allow_exact_matches, surname_first=surname_first))
        df1 = df1.explode(key)
        df1[key] = df1.apply(lambda row: row.key[0], axis=1)
        df1 = df1.merge(df2, how=how, left_on=key, right_on=right_on, indicator=indicator)
        del df1[key]
        return df1

    def dedupe(self, names, threshold=0.5, keep='longest', replace=False, reverse=True, surname_first=False, limit=3):
        # parameter validation
        if keep not in ('longest', 'frequent'):
            raise ValueError(
                'invalid arguement {} for parameter \'keep\', use one of -- longest, frequent, alpha'.format(keep))

        if keep == 'frequent':
            # make frequency counter
            count = Counter(names)

        if not replace:
            # early filtering of dupes by converting to set
            seen = set()
            seen_add = seen.add
            names = [x for x in names if not (x in seen or seen_add(x))]

        results = []
        for item in names:
            if item in results and replace is False:
                pass
            # find fuzzy matches
            matches = self.get_top_matches(item, names, limit=limit, thresh=threshold,
                                           exact=True, surname_first=surname_first)
            # no duplicates found
            if len(matches) == 0:
                results.append(item)

            else:
                # sort matches
                if keep == 'longest':
                    # sort by longest to shortest
                    matches = sorted(matches, key=lambda x: len(x[0]), reverse=reverse)
                elif keep == 'frequent':
                    # sort by most frequent, then longest
                    matches = sorted(matches, key=lambda x: (count[x[0]], len(x[0])), reverse=reverse)
                else:
                    # sort alphabetically
                    matches = sorted(matches, key=lambda x: x[0], reverse=reverse)
                if not (replace is False and matches[0][0] in results):
                    results.append(matches[0][0])
        return results

    def sum_ipa(self, name_a, name_b):
        feat1 = ipa_to_features(self.pe.encode(name_a))
        feat2 = ipa_to_features(self.pe.encode(name_b))
        score = sum(cmp_features(f1, f2) for f1, f2 in zip(feat1, feat2)) / len(feat1)
        return score

    def preprocess(self, name):
        # lookup name
        seen = self.seen_set(name, self.seen_names)
        if seen is not None:
            return seen
        # chained processing steps
        processed_name = re.sub('[^a-zA-Z\W]+', '', unidecode.unidecode(name).lower().strip()) \
            .replace('\'s', '').replace('\'', '')
        processed_name = [x for x in re.split('\W+', processed_name) if x != '']
        # add processed name to the seen dictionary
        self.seen_names[hash(name)] = processed_name
        return processed_name

    def featurize(self, pair):
        if len(pair) != 2:
            raise ValueError(
                'Length mismatch: Expected axis has 2 elements, new values have {} elements'.format(len(pair)))
        # syllable tokenize names
        syll_a = syllable_tokenizer.syllables(pair[0])
        syll_b = syllable_tokenizer.syllables(pair[1])

        # generate unique features
        features = np.zeros(23)
        features[0] = fuzz.partial_ratio(syll_a, syll_b)  # partial ratio
        features[1] = fuzz.token_sort_ratio(syll_a, syll_b)  # sort ratio
        features[2] = fuzz.token_set_ratio(syll_a, syll_b)  # set ratio
        features[3] = self.sum_ipa(pair[0], pair[1])  # sum IPA
        features[4] = 1 if self.pshp_soundex_first.encode(pair[0]) == self.pshp_soundex_first.encode(
            pair[1]) else 0  # PSHPSoundexFirst
        # generate remaining features
        for i, algo in enumerate(self.algos):
            features[i + 5] = algo.sim(pair[0], pair[1])
        return features

    def transform_names(self, pair):
        x1 = np.asarray(list(self.vocab.transform(np.asarray([pair[0]]))))
        x2 = np.asarray(list(self.vocab.transform(np.asarray([pair[1]]))))
        return x1, x2

    def siamese_inf(self, df):
        x1, x2 = self.transform_names(df)

        # collect the predictions here
        (prediction, sim) = self.sess.run([self.prediction, self.sim], {
            self.input_x1: x1,
            self.input_x2: x2,
            self.dropout_keep_prob: 1.0,
        })
        sim = 1 - prediction[0]
        return sim

    def base_model_inf(self, x):
        # get the positive class prediction from model
        y_pred = self.baseModel.predict_proba(x.reshape(1, -1))[0, 1]
        return y_pred

    def meta_inf(self, pair, base_features):
        meta_features = np.zeros(5)
        meta_features[0] = self.base_model_inf(base_features)
        meta_features[1] = self.siamese_inf(pair)
        # add base features to meta_features ('tkn_set', 'iterativesubstring', 'strcmp95')
        meta_features[2] = base_features[2]  # tkn_set
        meta_features[3] = base_features[5]  # iterativesubstring
        meta_features[4] = base_features[11]  # strcmp95

        sim = self.metaModel.predict_proba(meta_features.reshape(1, -1))[0, 1]
        return sim

    def seen_set(self, item, mapping):
        h = hash(item)
        if h in mapping:
            return mapping[h]

    def get_top_matches(self, name, choices, thresh=0.5, exact=True, limit=1, surname_first=False):
        sl = self.get_matches(name, choices, thresh, exact, surname_first=surname_first)
        return heapq.nlargest(limit, sl, key=lambda i: i[1]) if limit is not None else sorted(
            sl, key=lambda i: i[1], reverse=True)

    def get_matches(self, name, choices, score_cutoff=0.5, exact=True, surname_first=False):
        # catch generators without lengths
        if choices is None or len(choices) == 0:
            return

        exact = 2 if exact is True else 1
        for choice in choices:
            score = self.similarity(name, choice, surname_first=surname_first)
            if exact > score >= score_cutoff:
                yield choice, score
