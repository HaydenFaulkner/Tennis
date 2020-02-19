"""
Code for making the Tennis Set word embeddings

Mostly taken from https://gluon-nlp.mxnet.io/examples/word_embedding/word_embedding_training.html
"""

import warnings
warnings.filterwarnings('ignore')

import itertools
import time
import os

import mxnet as mx
import gluonnlp as nlp
from mxnet.gluon.data import SimpleDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.embeddings.data import transform_data_fasttext
from utils.embeddings.model import SG as SkipGramNet

ADD_EXTRA = True
BATCH_SIZE = 32
EMB_SIZE = 100
OVERWRITE = False

# context = mx.cpu()  # Enable this to run on CPU
context = mx.gpu(0)  # Enable this to run on GPU

with open(os.path.join('data', 'annotations', 'captions.txt')) as f:
    lines = f.readlines()
    lines = [line.rstrip().split()[1:] for line in lines]

if ADD_EXTRA:
    with open(os.path.join('data', 'annotations', 'captions_extra_001-045.txt')) as f:
        lines_extra = f.readlines()
        lines_extra = [line.rstrip().split()[1:] for line in lines_extra]   
    lines += lines_extra

tennis_caps = SimpleDataset(lines)

counter = nlp.data.count_tokens(itertools.chain.from_iterable(tennis_caps))
vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None, bos_token=None, eos_token=None, min_freq=1)
idx_to_counts = [counter[w] for w in vocab.idx_to_token]


def code(cap):
    return [vocab[token] for token in cap if token in vocab]


tennis_caps = tennis_caps.transform(code, lazy=False)

print('# sentences:', len(tennis_caps))
for sentence in tennis_caps[:3]:
    print('# tokens:', len(sentence), sentence[:5])

data = nlp.data.SimpleDataStream([tennis_caps])

data, batchify_fn, subword_function = transform_data_fasttext(
    data, vocab, idx_to_counts, cbow=False, ngrams=[3, 4, 5, 6], ngram_buckets=100, batch_size=BATCH_SIZE, window_size=3)

batches = data.transform(batchify_fn)

idx_to_subwordidxs = subword_function(vocab.idx_to_token)
for word, subwords in zip(vocab.idx_to_token[:3], idx_to_subwordidxs[:3]):
    print('<'+word+'>', subwords, sep='\t')

negatives_weights = mx.nd.array(idx_to_counts)
embedding = SkipGramNet(
    vocab.token_to_idx, EMB_SIZE, BATCH_SIZE, negatives_weights, subword_function, num_negatives=3, smoothing=0.75)
embedding.initialize(ctx=context)
embedding.hybridize()
trainer = mx.gluon.Trainer(embedding.collect_params(), 'adagrad', dict(learning_rate=0.05))

print(embedding)


def norm_vecs_by_row(x):
    return x / (mx.nd.sum(x * x, axis=1) + 1e-10).sqrt().reshape((-1, 1))


def get_k_closest_tokens(vocab, embedding, k, word):
    word_vec = norm_vecs_by_row(embedding[[word]])
    vocab_vecs = norm_vecs_by_row(embedding[vocab.idx_to_token])
    dot_prod = mx.nd.dot(vocab_vecs, word_vec.T)
    indices = mx.nd.topk(
        dot_prod.reshape((len(vocab.idx_to_token), )),
        k=k + 1,
        ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    result = [vocab.idx_to_token[i] for i in indices[1:]]
    print('closest tokens to "%s": %s' % (word, ", ".join(result)))


example_token = "hits"
get_k_closest_tokens(vocab, embedding, 10, example_token)

log_interval = 500


def train_embedding(num_epochs):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        l_avg = 0
        log_wc = 0

        print('Beginning epoch %d and resampling data.' % epoch)
        for i, batch in enumerate(batches):
            batch = [array.as_in_context(context) for array in batch]
            with mx.autograd.record():
                l = embedding(*batch)
            l.backward()
            trainer.step(1)

            l_avg += l.mean()
            log_wc += l.shape[0]
            if i % log_interval == 0:
                mx.nd.waitall()
                wps = log_wc / (time.time() - start_time)
                l_avg = l_avg.asscalar() / log_interval
                print('epoch %d, iteration %d, loss %.2f, throughput=%.2fK wps'
                      % (epoch, i, l_avg, wps / 1000))
                start_time = time.time()
                log_wc = 0
                l_avg = 0

        get_k_closest_tokens(vocab, embedding, 10, example_token)
        print("")


train_embedding(num_epochs=2)

# now lets save the embeddings to a file
vocab_vecs = norm_vecs_by_row(embedding[vocab.idx_to_token]).asnumpy()

if ADD_EXTRA:
    save_path = os.path.join('data', 'embeddings-ex.txt')
else:
    save_path = os.path.join('data', 'embeddings.txt')

if OVERWRITE or not os.path.exists(save_path):
    with open(save_path, 'w') as f:
        for i, word in enumerate(vocab.idx_to_token):
            f.write('%s %s\n' % (word, ' '.join([str(x) for x in list(vocab_vecs[i, :])])))


def visualise():
    vocab_vecs = norm_vecs_by_row(embedding[vocab.idx_to_token]).asnumpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(vocab_vecs)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.scatter(
        x=tsne_results[:, 0], y=tsne_results[:, 1]
    )
    for i, txt in enumerate(vocab.idx_to_token):
        ax.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]))
    plt.show()

    with open('embs-for-vis.txt', 'w') as f:
        f.write('"word","x","y"\n')
        for i, txt in enumerate(vocab.idx_to_token):
            f.write('"%s",%f,%f\n' % (txt, tsne_results[i, 0], tsne_results[i, 1]))
    print()


visualise()
