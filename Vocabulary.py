# Vocabulary.py
from collections import Counter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize


class Vocabulary:
    """Builds a PAD/UNK-aware word-index mapping."""

    def __init__(self, corpus, min_freq: int = 2):
        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus, min_freq)
        self.size = len(self.word2idx)

    # --------------------------------------------------------
    # Tokenisation
    # --------------------------------------------------------
    @staticmethod
    def tokenize(text: str):
        return [w.lower() for w in word_tokenize(text)]

    # --------------------------------------------------------
    # Public helpers
    # --------------------------------------------------------
    def most_common(self, k: int):
        return sorted(self.freq, key=self.freq.get, reverse=True)[:k]

    def text2idx(self, text: str):
        return [self.word2idx.get(t, 1) for t in self.tokenize(text)]  # 1 = <UNK>

    def idx2text(self, idxs):
        return [self.idx2word.get(i, "<UNK>") for i in idxs]

    # --------------------------------------------------------
    # Core builder
    # --------------------------------------------------------
    def build_vocab(self, corpus, min_freq):
        # word counts
        freq_counter = Counter()
        for doc in corpus:
            freq_counter.update(self.tokenize(doc))

        # initialise dicts with special tokens
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        idx2word = {0: "<PAD>", 1: "<UNK>"}

        # add tokens above threshold
        for token, count in sorted(freq_counter.items(), key=lambda x: (-x[1], x[0])):
            if count >= min_freq:
                idx = len(word2idx)
                word2idx[token] = idx
                idx2word[idx] = token

        # keep full frequency map (including specials)
        freq_counter["<PAD>"] = 0
        freq_counter["<UNK>"] = 0

        return word2idx, idx2word, dict(freq_counter)

    # --------------------------------------------------------
    # Optional visualisations
    # --------------------------------------------------------
    def make_vocab_charts(self):
        counts = [self.freq[w] for w in self.idx2word.values()]
        counts_sorted = sorted(counts, reverse=True)

        # Zipf plot
        plt.figure()
        plt.plot(counts_sorted)
        plt.yscale("log")
        plt.title("Token-frequency distribution")
        plt.xlabel("Token rank")
        plt.ylabel("Frequency (log)")
        plt.axhline(y=2, color="r", linestyle="--", label="min_freq")
        plt.legend()
        plt.show()

        # Cumulative coverage
        cum, total = [], sum(counts_sorted)
        running = 0
        for c in counts_sorted:
            running += c
            cum.append(running / total)

        plt.figure()
        plt.plot(cum)
        plt.title("Cumulative coverage")
        plt.xlabel("Token rank")
        plt.ylabel("Fraction of total tokens covered")
        plt.show()
