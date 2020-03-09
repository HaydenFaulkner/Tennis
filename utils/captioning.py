# coding: utf-8

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

from absl.flags import FLAGS
import io
import logging
from mxnet import gluon
import gluonnlp as nlp
import gluonnlp.data.batchify as btf


def get_dataloaders(data_train, data_val, data_test, use_average_length=False, num_shards=0, num_workers=8):
    """Create data loaders for training/validation/test."""
    data_train_lengths = data_train.get_data_lens()  # get_data_lengths(data_train)
    data_val_lengths = data_val.get_data_lens()  # get_data_lengths(data_val)
    data_test_lengths = data_test.get_data_lens()  # get_data_lengths(data_test)
    train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                                  btf.Stack(dtype='float32'), btf.Stack(dtype='float32'))
    test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                                 btf.Stack(dtype='float32'), btf.Stack(dtype='float32'),
                                 btf.Stack())
    target_val_lengths = list(map(lambda x: x[-1], data_val_lengths))
    target_test_lengths = list(map(lambda x: x[-1], data_test_lengths))
    if FLAGS.bucket_scheme == 'constant':
        bucket_scheme = nlp.data.ConstWidthBucket()
    elif FLAGS.bucket_scheme == 'linear':
        bucket_scheme = nlp.data.LinearWidthBucket()
    elif FLAGS.bucket_scheme == 'exp':
        bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
    else:
        raise NotImplementedError
    train_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_train_lengths,
                                                      batch_size=FLAGS.batch_size,
                                                      num_buckets=FLAGS.num_buckets,
                                                      ratio=FLAGS.bucket_ratio,
                                                      shuffle=True,
                                                      use_average_length=use_average_length,
                                                      num_shards=num_shards,
                                                      bucket_scheme=bucket_scheme)
    logging.info('Train Batch Sampler:\n%s', train_batch_sampler.stats())
    train_data_loader = nlp.data.ShardedDataLoader(data_train,
                                                   batch_sampler=train_batch_sampler,
                                                   batchify_fn=train_batchify_fn,
                                                   num_workers=num_workers)

    val_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_val_lengths,
                                                    batch_size=FLAGS.test_batch_size,
                                                    num_buckets=FLAGS.num_buckets,
                                                    ratio=FLAGS.bucket_ratio,
                                                    shuffle=False,
                                                    use_average_length=use_average_length,
                                                    bucket_scheme=bucket_scheme)
    logging.info('Valid Batch Sampler:\n%s', val_batch_sampler.stats())
    val_data_loader = gluon.data.DataLoader(data_val,
                                            batch_sampler=val_batch_sampler,
                                            batchify_fn=test_batchify_fn,
                                            num_workers=num_workers)
    test_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_test_lengths,
                                                     batch_size=FLAGS.test_batch_size,
                                                     num_buckets=FLAGS.num_buckets,
                                                     ratio=FLAGS.bucket_ratio,
                                                     shuffle=False,
                                                     use_average_length=use_average_length,
                                                     bucket_scheme=bucket_scheme)
    logging.info('Test Batch Sampler:\n%s', test_batch_sampler.stats())
    test_data_loader = gluon.data.DataLoader(data_test,
                                             batch_sampler=test_batch_sampler,
                                             batchify_fn=test_batchify_fn,
                                             num_workers=num_workers)
    return train_data_loader, val_data_loader, test_data_loader


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            if isinstance(sent, (list, tuple)):
                of.write(u' '.join(sent) + u'\n')
            else:
                of.write(sent + u'\n')


def read_sentences(file_path):
    return NotImplementedError


def get_comp_str(tgts, prds):
    str_ = ''
    for tgt, prd in zip(tgts, prds):
        str_ += 'GT:\t'
        if isinstance(tgt, (list, tuple)):
            str_ += ' '.join(tgt) + '\n'
        else:
            str_ += tgt + '\n'

        str_ += '\nPD:\t'
        if isinstance(prd, (list, tuple)):
            str_ += ' '.join(prd) + '\n'
        else:
            str_ += prd + '\n'

        str_ += '\n\n'

    return str_
