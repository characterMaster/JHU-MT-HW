#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import math

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="iterations", default=10, type="int", help="Number of EM iterations (default=10)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.1, type="float", help="Threshold for alignment (default=0.1)")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# load data
sys.stderr.write("Reading data...\n")
bitext = []
for (n, (f_sent, e_sent)) in enumerate(zip(open(f_data), open(e_data))):
    f_words = f_sent.strip().split()
    e_words = e_sent.strip().split()
    if len(f_words) > 0 and len(e_words) > 0:  # skip the empty
        bitext.append((f_words, e_words))
    if len(bitext) >= opts.num_sents:
        break

sys.stderr.write("Read %d sentence pairs\n" % len(bitext))

f_vocab = set()  # French
e_vocab = set()  # English

for (f_sent, e_sent) in bitext:
    f_vocab.update(f_sent)
    e_vocab.update(e_sent)

# initialize t(f|e)
t = defaultdict(lambda: defaultdict(float))
uniform_prob = 1.0 / len(e_vocab)

for f in f_vocab:
    for e in e_vocab:
        t[f][e] = uniform_prob

sys.stderr.write("French vocabulary size: %d\n" % len(f_vocab))
sys.stderr.write("English vocabulary size: %d\n" % len(e_vocab))

# initialize prob
sys.stderr.write("Initializing IBM Model 1...\n")
t = defaultdict(lambda: defaultdict(float))

uniform_prob = 1.0 / len(e_vocab)
for (f_sent, e_sent) in bitext:
    for f in f_sent:
        for e in e_sent:
            if t[f][e] == 0:
                t[f][e] = uniform_prob

# EM training
for iter_num in range(opts.iterations):
    sys.stderr.write("Iteration %d..." % (iter_num + 1))

    count = defaultdict(lambda: defaultdict(float))
    total = defaultdict(float)

    # E-step
    for (f_sent, e_sent) in bitext:
        # for every single French word
        for f in f_sent:
            # calculate z
            z = sum(t[f][e] for e in e_sent)

            # collect
            for e in e_sent:
                delta = t[f][e] / z if z > 0 else 0
                count[f][e] += delta
                total[e] += delta

    # M-step
    for f in f_vocab:
        for e in e_vocab:
            if total[e] > 0:
                t[f][e] = count[f][e] / total[e]
            else:
                t[f][e] = 0

    sys.stderr.write(" done\n")

sys.stderr.write("Generating alignments...\n")
for (f_sent, e_sent) in bitext:
    alignment = []
    for (i, f) in enumerate(f_sent):
        best_j = -1
        best_prob = opts.threshold
        for (j, e) in enumerate(e_sent):
            # adding diagonal bias
            distance_penalty = math.exp(-abs(i - j * len(f_sent) / len(e_sent)))
            prob = t[f][e] * distance_penalty

            if prob > best_prob:
                best_prob = prob
                best_j = j
        if best_j >= 0:
            alignment.append(f"{i}-{best_j}")
    print(" ".join(alignment))
