# -*- coding: utf-8 -*-
# 어휘 사전과 워드 임베딩을 만들고, 학습을 위해 대화 데이터를 읽어들이는 유틸리티들의 모음

import tensorflow as tf
import numpy as np
import re
import codecs
from konlpy.tag import Twitter

from config import FLAGS


class News():

    _PAD_ = "_PAD_"  # 빈칸 채우는 심볼
    _STA_ = "_STA_"  # 디코드 입력 시퀀스의 시작 심볼
    _EOS_ = "_EOS_"  # 디코드 입출력 시퀀스의 종료 심볼
    _UNK_ = "_UNK_"  # 사전에 없는 단어를 나타내는 심볼

    _PAD_ID_ = 0
    _STA_ID_ = 1
    _EOS_ID_ = 2
    _UNK_ID_ = 3
    _PRE_DEFINED_ = [_PAD_ID_, _STA_ID_, _EOS_ID_, _UNK_ID_]
    _PRE_DEFINED_STR_ = [_PAD_, _STA_, _EOS_, _UNK_]


    def __init__(self):
        self.max_desc_len = 200
        self.vocab_list = []
        self.vocab_dict = {}
        self.vocab_size = 0
        self.description = []
        self.headlines = []
        # self.examples = []

        self._index_in_epoch = 0
        self.pos_tagger = Twitter()
        self.batches = None

    def decode(self, indices, string=False):
        tokens = [[self.vocab_list[i] for i in dec] for dec in indices]

        if string:
            return self.decode_to_string(tokens[0])
        else:
            return tokens

    def decode_to_string(self, tokens):
        text = ' '.join(tokens)
        return text.strip()

    def cut_eos(self, indices):
        eos_idx = indices.index(self._EOS_ID_)
        return indices[:eos_idx]

    def is_eos(self, voc_id):
        return voc_id == self._EOS_ID_

    def is_defined(self, voc_id):
        return voc_id in self._PRE_DEFINED_

    def max_len(self, batch_set):
        max_len = 0

        for i in range(0, len(batch_set)):
            len_input = len(batch_set[i])
            if len_input > max_len:
                max_len = len_input

        return max_len

    def pad(self, seq, max_len, start=None, eos=None):
        if start:
            padded_seq = [self._STA_ID_] + seq
        elif eos:
            padded_seq = seq + [self._EOS_ID_]
        else:
            padded_seq = seq

        if len(padded_seq) < max_len:
            return padded_seq + ([self._PAD_ID_] * (max_len - len(padded_seq)))
        else:
            return padded_seq[:max_len]

    # def pad_left(self, seq, max_len):
    #     if len(seq) < max_len:
    #         return ([self._PAD_ID_] * (max_len - len(seq))) + seq
    #     else:
    #         return seq

    def transform(self, input, output, input_max, output_max):
        # 지정된 max 길이까지만 input(desc)에 사용
        input_max = min([input_max, self.max_desc_len])
        enc_input = self.pad(input, input_max)
        dec_input = self.pad(output, output_max, start=True)
        target = self.pad(output, output_max, eos=True)

        # # 구글 방식으로 입력을 인코더에 역순으로 입력한다.
        # enc_input.reverse()
        #
        # enc_input = np.eye(self.vocab_size)[enc_input]
        # dec_input = np.eye(self.vocab_size)[dec_input]
        # target = np.eye(self.vocab_size)[target]

        return enc_input, dec_input, target

    def next_batch(self, batch_size):
        enc_input = []
        enc_seq_len = []
        dec_input = []
        dec_seq_len = []
        target = []

        start = self._index_in_epoch

        if self._index_in_epoch + batch_size < len(self.headlines) - 1:
            self._index_in_epoch = self._index_in_epoch + batch_size
        else:
            self._index_in_epoch = 0

        batch_set_head = self.headlines[start:(start+batch_size)]
        batch_set_desc = self.description[start:(start+batch_size)]


        # TODO: 구글처럼 버킷을 이용한 방식으로 변경
        # 간단하게 만들기 위해 구글처럼 버킷을 쓰지 않고 같은 배치는 같은 사이즈를 사용하도록 만듬
        max_len_output = self.max_len(batch_set_head) + 1
        max_len_input = self.max_len(batch_set_desc)

        for i in range(len(batch_set_head)):
            enc, dec, tar = self.transform(batch_set_desc[i], batch_set_head[i],
                                           max_len_input, max_len_output)

            # dec = [dec[0]]
            enc_input.append(enc)
            enc_seq_len.append(len(enc))
            dec_input.append(dec)
            dec_seq_len.append(len(dec))
            target.append(tar)

        return enc_input, dec_input, target, enc_seq_len, dec_seq_len

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def build_batches(self, batch_size, num_epochs, shuffle=True):
        self.batches = self.batch_iter(
                            list(zip(self.description, self.headlines)), batch_size, num_epochs, shuffle)

    def next_feed(self):
        batch = next(self.batches)

        enc_input = []
        dec_input = []
        target = []

        # start = self._index_in_epoch
        #
        # if self._index_in_epoch + batch_size < len(self.headlines) - 1:
        #     self._index_in_epoch = self._index_in_epoch + batch_size
        # else:
        #     self._index_in_epoch = 0
        #
        # batch_set_head = self.headlines[start:(start + batch_size)]
        # batch_set_desc = self.description[start:(start + batch_size)]
        batch_set_head = batch[:, 1]
        batch_set_desc = batch[:, 0]

        # TODO: 구글처럼 버킷을 이용한 방식으로 변경
        # 간단하게 만들기 위해 구글처럼 버킷을 쓰지 않고 같은 배치는 같은 사이즈를 사용하도록 만듬
        max_len_output = self.max_len(batch_set_head) + 1
        max_len_input = self.max_len(batch_set_desc)

        for i in range(len(batch_set_head)):
            enc, dec, tar = self.transform(batch_set_desc[i], batch_set_head[i],
                                           max_len_input, max_len_output)

            enc_input.append(enc)
            dec_input.append(dec)
            target.append(tar)

        return enc_input, dec_input, target

    def tokens_to_ids(self, tokens):
        ids = []

        for t in tokens:
            if t in self.vocab_dict:
                ids.append(self.vocab_dict[t])
            else:
                ids.append(self._UNK_ID_)

        return ids

    def ids_to_tokens(self, ids):
        tokens = []

        for i in ids:
            tokens.append(self.vocab_list[i])

        return tokens

    def read_data(self, data_path):
        self.description = []
        self.headlines = []

        with open(data_path, 'r', encoding='utf-8') as content_file:
            data = [line.split('\t') for line in content_file.read().splitlines()]
            data = data[1:]  # header 제외
            # data = [sentence for line in data for sentence in line if len(sentence) > 2]
            desc_tokens = [self.tokenizer(row[1].strip()) for row in data]
            desc_ids = [self.tokens_to_ids(tokens) for tokens in desc_tokens]
            self.description = desc_ids
            head_tokens = [self.tokenizer(row[0].strip()) for row in data]
            head_ids = [self.tokens_to_ids(tokens) for tokens in head_tokens]
            self.headlines = head_ids


    def tokenizer_w_pos(self, pos_tagger, sentence):
        # return ['_'.join(t) for t in pos_tagger.pos(sentence, norm=True, stem=True)]
        return [t[0] for t in pos_tagger.pos(sentence, norm=True, stem=True)]

    def tokenizer(self, sentence):
        # 공백으로 나누고 특수문자는 따로 뽑아낸다.
        words = []
        # _TOKEN_RE_ = re.compile(b"([.,!?\"':;)(])")
        _TOKEN_RE_ = re.compile("([.,!?\"':;)(])")

        for fragment in sentence.strip().split():
            words.extend(_TOKEN_RE_.split(fragment))

        return [w for w in words if w]

    def build_vocab(self, data_path, vocab_path):
        # TODO: 빈도 수 높은 단어 위주로 vocabulary 빌드. input/output 별도 vocab. 빌드
        with open(data_path, 'r', encoding='utf-8') as content_file:
            content = content_file.read()
            # words = self.tokenizer_w_pos(self.pos_tagger, content)
            words = self.tokenizer(content)
            words = list(set(words))

        with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
            for w in words:
                vocab_file.write(w + '\n')

    def load_vocab(self, vocab_path):
        self.vocab_list = self._PRE_DEFINED_STR_ + []
        # self.vocab_list = self._PRE_DEFINED_ + []

        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                self.vocab_list.append(line.strip())

        # {'_PAD_': 0, '_STA_': 1, '_EOS_': 2, '_UNK_': 3, 'Hello': 4, 'World': 5, ...}
        self.vocab_dict = {n: i for i, n in enumerate(self.vocab_list)}
        self.vocab_size = len(self.vocab_list)


def main(_):
    news = News()

    if FLAGS.data_path and FLAGS.voc_test:
        print ("다음 데이터로 어휘 사전을 테스트합니다.", FLAGS.data_path)
        news.load_vocab(FLAGS.voc_path)
        news.read_data(FLAGS.data_path)

        enc, dec, target = news.next_batch(10)
        print (target)
        enc, dec, target = news.next_batch(10)
        print (target)

    elif FLAGS.data_path and FLAGS.voc_build:
        print ("다음 데이터에서 어휘 사전을 생성합니다.", FLAGS.data_path)
        news.build_vocab(FLAGS.data_path, FLAGS.voc_path)

    elif FLAGS.voc_test:
        news.load_vocab(FLAGS.voc_path)
        print (news.vocab_dict)


if __name__ == "__main__":
    tf.app.run()
