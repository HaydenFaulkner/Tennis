"""Word embedding models.

From https://github.com/dmlc/gluon-nlp/blob/ef57ca08ee507902a9b2fbb4dbcc4ca110d84cd3/scripts/word_embeddings/model.py
"""
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
import mxnet as mx
import numpy as np

import gluonnlp as nlp


class Net(mx.gluon.HybridBlock):
    """Base class for word2vec and fastText SkipGram and CBOW networks.
    Parameters
    ----------
    token_to_idx : dict
        token_to_idx mapping of the vocabulary that this model is to be trained
        with. token_to_idx is used for __getitem__ and __contains__. For
        len(token_to_idx) is used during initialization to obtain the input_dim
        of the embedding matrix.
    output_dim : int
        Dimension of the dense embedding.
    batch_size : int
        Batchsize this model will be trained with. TODO temporary until
        random_like ops are supported
    negatives_weights : mxnet.nd.NDArray
        Weights for UnigramCandidateSampler for sampling negatives.
    smoothing : float, default 0.75
        Smoothing factor applied to negatives_weights. Final weights are
        mxnet.nd.power(negative_weights, smoothing).
    num_negatives : int, default 5
        Number of negatives to sample for each real sample.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.
    dtype : str, default 'float32'
        dtype argument passed to gluon.nn.Embedding
    """

    # pylint: disable=abstract-method
    def __init__(self, token_to_idx, output_dim, batch_size, negatives_weights,
                 subword_function=None, num_negatives=5, smoothing=0.75,
                 sparse_grad=True, dtype='float32', **kwargs):
        super(Net, self).__init__(**kwargs)

        self._kwargs = dict(
            input_dim=len(token_to_idx), output_dim=output_dim, dtype=dtype,
            sparse_grad=sparse_grad, num_negatives=num_negatives)

        with self.name_scope():
            if subword_function is not None:
                self.embedding = nlp.model.train.FasttextEmbeddingModel(
                    token_to_idx=token_to_idx,
                    subword_function=subword_function,
                    output_dim=output_dim,
                    weight_initializer=mx.init.Uniform(scale=1 / output_dim),
                    sparse_grad=sparse_grad,
                )
            else:
                self.embedding = nlp.model.train.CSREmbeddingModel(
                    token_to_idx=token_to_idx,
                    output_dim=output_dim,
                    weight_initializer=mx.init.Uniform(scale=1 / output_dim),
                    sparse_grad=sparse_grad,
                )
            self.embedding_out = mx.gluon.nn.Embedding(
                len(token_to_idx), output_dim=output_dim,
                weight_initializer=mx.init.Zero(), sparse_grad=sparse_grad,
                dtype=dtype)

            self.negatives_sampler = UnigramCandidateSampler(weights=negatives_weights**smoothing, dtype='int64')

    def __getitem__(self, tokens):
        return self.embedding[tokens]


class SG(Net):
    """SkipGram network"""

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, center, context, center_words):
        """SkipGram forward pass.
        Parameters
        ----------
        center : mxnet.nd.NDArray or mxnet.sym.Symbol
            Sparse CSR array of word / subword indices of shape (batch_size,
            len(token_to_idx) + num_subwords). Embedding for center words are
            computed via F.sparse.dot between the CSR center array and the
            weight matrix.
        context : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array of context words of shape (batch_size, ). Also used for
            row-wise independently masking negatives equal to one of context.
        center_words : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array of center words of shape (batch_size, ). Only used for
            row-wise independently masking negatives equal to one of
            center_words.
        """

        # negatives sampling
        negatives = []
        mask = []
        for _ in range(self._kwargs['num_negatives']):
            negatives.append(self.negatives_sampler(center_words))
            mask_ = negatives[-1] != center_words
            mask_ = F.stack(mask_, (negatives[-1] != context))
            mask.append(mask_.min(axis=0))

        negatives = F.stack(*negatives, axis=1)
        mask = F.stack(*mask, axis=1).astype(np.float32)

        # center - context pairs
        emb_center = self.embedding(center).expand_dims(1)
        emb_context = self.embedding_out(context).expand_dims(2)
        pred_pos = F.batch_dot(emb_center, emb_context).squeeze()
        loss_pos = (F.relu(pred_pos) - pred_pos + F.Activation(
            -F.abs(pred_pos), act_type='softrelu')) / (mask.sum(axis=1) + 1)

        # center - negatives pairs
        emb_negatives = self.embedding_out(negatives).reshape(
            (-1, self._kwargs['num_negatives'],
             self._kwargs['output_dim'])).swapaxes(1, 2)
        pred_neg = F.batch_dot(emb_center, emb_negatives).squeeze()
        mask = mask.reshape((-1, self._kwargs['num_negatives']))
        loss_neg = (F.relu(pred_neg) + F.Activation(
            -F.abs(pred_neg), act_type='softrelu')) * mask
        loss_neg = loss_neg.sum(axis=1) / (mask.sum(axis=1) + 1)

        return loss_pos + loss_neg


class UnigramCandidateSampler(mx.gluon.HybridBlock):
    """Unigram Candidate Sampler
    Draw random samples from a unigram distribution with specified weights
    using the alias method.
    Parameters
    ----------
    weights : mx.nd.NDArray
        Unnormalized class probabilities. Samples are drawn and returned on the
        same context as weights.context.
    dtype : str or np.dtype, default 'float32'
        Data type of the candidates. Make sure that the dtype precision is
        large enough to represent the size of your weights array precisely. For
        example, float32 can not distinguish 2**24 from 2**24 + 1.
    """

    def __init__(self, weights, dtype='float32'):
        super(UnigramCandidateSampler, self).__init__()
        self._dtype = dtype
        self.N = weights.size

        if (np.dtype(dtype) == np.float32 and weights.size > 2**24) or \
           (np.dtype(dtype) == np.float16 and weights.size > 2**11):
            s = 'dtype={dtype} can not represent all weights'
            raise ValueError(s.format(dtype=dtype))

        total_weights = weights.sum()
        prob = (weights * self.N / total_weights).asnumpy().tolist()
        alias = [0] * self.N

        # sort the data into the outcomes with probabilities
        # that are high and low than 1/N.
        low = []
        high = []
        for i in range(self.N):
            if prob[i] < 1.0:
                low.append(i)
            else:
                high.append(i)

        # pair low with high
        while len(low) > 0 and len(high) > 0:
            l = low.pop()
            h = high.pop()

            alias[l] = h
            prob[h] = prob[h] - (1.0 - prob[l])

            if prob[h] < 1.0:
                low.append(h)
            else:
                high.append(h)

        for i in low + high:
            prob[i] = 1
            alias[i] = i

        # store
        prob = mx.nd.array(prob, dtype='float64')
        alias = mx.nd.array(alias, dtype='float64')
        self.prob = self.params.get_constant('prob', prob)
        self.alias = self.params.get_constant('alias', alias)

    def __repr__(self):
        s = '{block_name}({len_weights}, {dtype})'
        return s.format(block_name=self.__class__.__name__, len_weights=self.N,
                        dtype=self._dtype)

    # pylint: disable=arguments-differ, unused-argument
    def hybrid_forward(self, F, candidates_like, prob, alias):
        """Draw samples from uniform distribution and return sampled candidates.
        Parameters
        ----------
        candidates_like: mxnet.nd.NDArray or mxnet.sym.Symbol
            This input specifies the shape of the to be sampled candidates. #
        Returns
        -------
        samples: mxnet.nd.NDArray or mxnet.sym.Symbol
            The sampled candidates of shape candidates_like.shape. Candidates
            are sampled based on the weights specified on creation of the
            UnigramCandidateSampler.
        """
        candidates_flat = candidates_like.reshape((-1, )).astype('float64')
        idx = F.random.uniform_like(candidates_flat, low=0, high=self.N).floor()
        prob = F.gather_nd(prob, idx.reshape((1, -1)))
        alias = F.gather_nd(alias, idx.reshape((1, -1)))
        where = F.random.uniform_like(candidates_flat) < prob
        hit = idx * where
        alt = alias * (1 - where)
        candidates = (hit + alt).reshape_like(candidates_like)

        return candidates.astype(self._dtype)