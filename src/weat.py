"""
Implementation of the Word Embedding Association Test (WEAT).

The code was copied and adapted from https://github.com/ryansteed/weat/.
"""

import logging
import math
import itertools as it
import numpy as np
import scipy.special
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity
import argparse

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=logging.INFO)


class WordEmbeddingAssociationTest:
    def __init__(self, X, Y, A, B, names=None):
        """
        A WEAT Test.
        :param X: A set of target embeddings
        :param Y: A set of target embeddings
        :param A: A set of attribute embeddings
        :param B: A set of attribute embeddings
        :param names: Optional set of names for X, Y, A, and B, in order
        :return: the effect size and p-value
        """
        self.X = X
        self.Y = Y
        self.A = A
        self.B = B
        self.names = names if names is not None else ["X", "Y", "A", "B"]
        self.reset_calc()

    def reset_calc(self):
        self.similarity_matrix = self.similarities()
        self.s_AB = None
        self.calc_s_AB()

    def run(self, randomized=False, **kwargs):
        """
        Run the test.
        """
        if randomized:
            X_orig = self.X
            Y_orig = self.Y
            A_orig = self.A
            B_orig = self.B
            D = np.concatenate((self.X, self.Y, self.A, self.B))
            np.random.shuffle(D)
            self.X = D[:X_orig.shape[0], :]
            self.Y = D[X_orig.shape[0]:2 * X_orig.shape[0], :]
            self.A = D[2 * X_orig.shape[0]:2 * X_orig.shape[0] + A_orig.shape[0], :]
            self.B = D[2 * X_orig.shape[0] + A_orig.shape[0]:, :]
            self.reset_calc()

        p = self.p(**kwargs)

        e = self.effect_size()

        if randomized:
            self.X = X_orig
            self.Y = Y_orig
            self.A = A_orig
            self.B = B_orig
            self.reset_calc()
        return e, p

    def similarities(self):
        """
        :return: an array of size (len(XY), len(AB)) containing cosine similarities
        between items in XY and items in AB.
        """
        XY = np.concatenate((self.X, self.Y))
        AB = np.concatenate((self.A, self.B))
        return cosine_similarity(XY, AB)

    def calc_s_AB(self):
        self.s_AB = self.s_wAB(np.arange(self.similarity_matrix.shape[0]))

    def s_wAB(self, w):
        """
        Return vector of s(w, A, B) across w, where
            s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
        :param w: Mask on the XY axis of similarity matrix
        """
        return self.similarity_matrix[w, :self.A.shape[0]].mean(axis=1) - self.similarity_matrix[w,
                                                                          self.A.shape[0]:].mean(axis=1)

    def s_XAB(self, mask):
        r"""
        Given indices of target concept X and precomputed s_wAB values,
        return slightly more computationally efficient version of WEAT
        statistic for p-value computation.
        Caliskan defines the WEAT statistic s(X, Y, A, B) as
            sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
        where s(w, A, B) is defined as
            mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
        The p-value is computed using a permutation test on (X, Y) over all
        partitions (X', Y') of X union Y with |X'| = |Y'|.
        However, for all partitions (X', Y') of X union Y,
            s(X', Y', A, B)
          = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
          = C,
        a constant.  Thus
            sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
          = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
          = C + 2 sum_{x in X'} s(x, A, B).
        By monotonicity,
            s(X', Y', A, B) > s(X, Y, A, B)
        if and only if
            [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
        that is,
            sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
        Thus we only need use the first component of s(X, Y, A, B) as our
        test statistic.
        :param mask: some random X partition of XY - in the form of a mask on XY
        """
        return self.s_AB[mask].sum()

    def s_XYAB(self, X, Y):
        r"""
        Given indices of target concept X and precomputed s_wAB values,
        the WEAT test statistic for p-value computation.
        :param X: Mask for XY indicating the values in partition X
        :param Y: Mask for XY indicating the values in partition Y
        """
        return self.s_XAB(X) - self.s_XAB(Y)

    def p(self, n_samples=10000, parametric=False):
        """
        Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
        """
        assert self.X.shape[0] == self.Y.shape[0]
        size = self.X.shape[0]

        XY = np.concatenate((self.X, self.Y))

        if parametric:
            # logging.info('Using parametric test')
            s = self.s_XYAB(np.arange(self.X.shape[0]), np.arange(self.X.shape[0], self.X.shape[0] + self.Y.shape[0]))

            samples = []
            for _ in range(n_samples):
                a = np.arange(XY.shape[0])
                np.random.shuffle(a)
                Xi = a[:size]
                Yi = a[size:]
                assert len(Xi) == len(Yi)
                si = self.s_XYAB(Xi, Yi)
                samples.append(si)

            # Compute sample standard deviation and compute p-value by
            # assuming normality of null distribution
            # logging.info('Inferring p-value based on normal distribution')
            (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
            #logging.info('Shapiro-Wilk normality test statistic: {:.2g}, p-value: {:.2g}'.format(
            #    shapiro_test_stat, shapiro_p_val))
            sample_mean = np.mean(samples)
            sample_std = np.std(samples, ddof=1)
            #logging.info('Sample mean: {:.2g}, sample standard deviation: {:.2g}'.format(
            #    sample_mean, sample_std))
            p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
            return p_val

        else:
            # logging.info('Using non-parametric test')
            s = self.s_XAB(np.arange(self.X.shape[0]))
            total_true = 0
            total_equal = 0
            total = 0

            num_partitions = int(scipy.special.binom(2 * self.X.shape[0], self.X.shape[0]))
            if num_partitions > n_samples:
                # We only have as much precision as the number of samples drawn;
                # bias the p-value (hallucinate a positive observation) to
                # reflect that.
                total_true += 1
                total += 1
            
                for i in range(n_samples - 1):
                    a = np.arange(XY.shape[0])
                    np.random.shuffle(a)
                    Xi = a[:size]
                    assert 2 * len(Xi) == len(XY)
                    si = self.s_XAB(Xi)
                    if si > s:
                        total_true += 1
                    elif si == s:  # use conservative test
                        total_true += 1
                        total_equal += 1
                    total += 1
            else:
                # logging.info('Using exact test ({} partitions)'.format(num_partitions))
                # iterate through all possible X-length combinations of the indices of XY
                for Xi in it.combinations(np.arange(XY.shape[0]), self.X.shape[0]):
                    assert 2 * len(Xi) == len(XY)
                    si = self.s_XAB(np.array(Xi))
                    if si > s:
                        total_true += 1
                    elif si == s:  # use conservative test
                        total_true += 1
                        total_equal += 1
                    total += 1

            #if total_equal:
            #    logging.warning('Equalities contributed {}/{} to p-value'.format(total_equal, total))

            return total_true / total

    def effect_size(self):
        """
        Compute the effect size, which is defined as
            [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
                [ stddev_{w in X u Y} s(w, A, B) ]
        args:
            - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
        """
        #print(np.mean(self.s_wAB(np.arange(self.X.shape[0]))))
        #print(np.mean(
        #    self.s_wAB(np.arange(self.X.shape[0], self.similarity_matrix.shape[0]))))
        #print(np.std(self.s_AB, ddof=1))
        numerator = np.mean(self.s_wAB(np.arange(self.X.shape[0]))) - np.mean(
            self.s_wAB(np.arange(self.X.shape[0], self.similarity_matrix.shape[0])))
        denominator = np.std(self.s_AB, ddof=1)
        return numerator / denominator
