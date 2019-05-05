import tensorflow as tf
import pandas as pd
import numpy as np

from string import punctuation
from collections import Counter

class DataLoader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train = pd.DataFrame(columns=['review', 'sentiment'])
        self.train = self.train.append(self.create_df_from_filepath('train', 'pos.txt', 1.0), ignore_index=True)
        self.train = self.train.append(self.create_df_from_filepath('train', 'neg.txt', -1.0), ignore_index=True)
        self.train['sentiment'] = pd.to_numeric(self.train['sentiment'])

        self.test = pd.DataFrame(columns=['review', 'sentiment'])
        self.test = self.test.append(self.create_df_from_filepath('test', 'pos.txt', 1.0), ignore_index=True)
        self.test = self.test.append(self.create_df_from_filepath('test', 'neg.txt', -1.0), ignore_index=True)
        self.test['sentiment'] = pd.to_numeric(self.test['sentiment'])

        self.preprocessing()

    @staticmethod
    def create_df_from_filepath(train_test, filename, label):
        with open('data/' + train_test + '/' + filename, 'r') as file:
            review = file.read().split('\n')
            review = pd.DataFrame({'review': review})
            review['sentiment'] = pd.Series(np.array([label] * len(review)), index=review.index)
            return review

    def preprocessing(self):
        self.train['review'] = self.train['review'].apply(lambda x: x.lower())
        self.test['review'] = self.test['review'].apply(lambda x: ''.join([c for c in x if c not in punctuation]))

        all_text = ' '.join(self.train['review'])
        words = all_text.split()
        count_words = Counter(words)
        total_words = len(words)
        sorted_words = count_words.most_common(total_words)

        self.vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

        self.reviews_int = []
        for review in self.train['review']:
            r = [self.vocab_to_int[w] for w in review.split()]
            self.reviews_int.append(r)

        features = self.pad_features(200).reshape(-1, self.batch_size, 200)
        targets = self.train['sentiment'].values.reshape(-1, self.batch_size, 1)

        self.train = self.np_to_tf_dataset(features, targets)

    def pad_features(self, seq_length):
        ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
        '''
        features = np.zeros((len(self.reviews_int), seq_length), dtype=int)

        for i, review in enumerate(self.reviews_int):
            review_len = len(review)

            if review_len <= seq_length:
                zeroes = list(np.zeros(seq_length - review_len))
                new = zeroes + review
            elif review_len > seq_length:
                new = review[0:seq_length]

            features[i, :] = np.array(new)

        return features

    @staticmethod
    def np_to_tf_dataset(np_X, np_y):
        return tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(np_X, tf.float32),
                tf.cast(np_y, tf.float32)
            )
        )