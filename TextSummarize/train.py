# -*- coding: utf-8 -*-

import tensorflow as tf
import random
import math
import os
import numpy as np

from config import FLAGS
# from model import Seq2Seq
from news import News
from mySeq2Seq import Seq2Seq_text_embedding, Seq2Seq_text_embedding_with_prev_output, Seq2Seq_basic


def train(news, batch_size=100, epoch=10):
    # model = Seq2Seq(news.vocab_size)
    # model = Seq2Seq_text_embedding(news.vocab_size, 128)
    model = Seq2Seq_text_embedding_with_prev_output(news.vocab_size, 128, mode='train')

    with tf.Session() as sess:
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print ("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())
        # print ("새로운 모델을 생성하는 중 입니다.")
        # sess.run(tf.global_variables_initializer())


        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # old
        total_batch = int(math.ceil(len(news.headlines)/float(batch_size)))

        for step in range(total_batch * epoch):
            enc_input, dec_input, targets, enc_seq_len, dec_seq_len = news.next_batch(batch_size)

            _, loss = model.train(sess, enc_input, dec_input, targets, enc_seq_len, dec_seq_len)

            if (step + 1) % 100 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets, enc_seq_len, dec_seq_len)

                print ('Step:', '%06d' % model.global_step.eval(),\
                      'cost =', '{:.6f}'.format(loss))


        # # New
        # batches_in_epoch = int((len(news.headlines) - 1) / batch_size) + 1  # 1000
        # max_batches = batches_in_epoch * epoch - 1  # 3001
        # news.build_batches(batch_size=batch_size, num_epochs=epoch, shuffle=True)
        #
        # for batch in range(max_batches):
        #     enc_input, dec_input, targets = news.next_feed()
        #
        #     _, loss = model.train(sess, enc_input, dec_input, targets)
        #
        #     if (batch + 1) % 100 == 0:
        #         model.write_logs(sess, writer, enc_input, dec_input, targets)
        #
        #         print ('Step:', '%06d' % model.global_step.eval(),\
        #               'cost =', '{:.6f}'.format(loss))

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print ('최적화 완료!')


def test(news, batch_size=100):
    print ("\n=== 예측 테스트 ===")

    # model = Seq2Seq_text_embedding(news.vocab_size, 128)
    model = Seq2Seq_text_embedding_with_prev_output(news.vocab_size, 128, mode='inference')

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print ("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        # Old
        enc_input, dec_input, targets, enc_seq_len, dec_seq_len = news.next_batch(batch_size)

        # # New
        # epoch = 1
        # batches_in_epoch = int((len(news.headlines) - 1) / batch_size) + 1  # 1000
        # max_batches = batches_in_epoch * epoch - 1  # 3001
        # news.build_batches(batch_size=batch_size, num_epochs=epoch, shuffle=True)
        #
        # enc_input, dec_input, targets = news.next_feed()

        expect, outputs, accuracy = model.test(sess, enc_input, enc_seq_len, targets)

        expect = news.decode(expect)
        outputs = news.decode(outputs)

        pick = random.randrange(0, len(expect))
        input = news.decode([news.description[pick]], True)
        expect = news.decode([news.headlines[pick]], True)
        # outputs = news.cut_eos(outputs[pick])
        outputs = outputs[pick]

        print ("\n정확도:", accuracy)
        print ("랜덤 결과\n",)
        print ("    입력값:", input)
        print ("    실제값:", expect)
        print ("    예측값:", ' '.join(outputs))


def inference(news, batch_size=100):
    print ("\n=== 예측값 생성 테스트 ===")

    model = Seq2Seq_text_embedding_with_prev_output(news.vocab_size, 128, mode='inference')

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print ("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        # Old
        enc_input, dec_input, targets, enc_seq_len, dec_seq_len = news.next_batch(batch_size)

        # # New
        # epoch = 1
        # batches_in_epoch = int((len(news.headlines) - 1) / batch_size) + 1  # 1000
        # max_batches = batches_in_epoch * epoch - 1  # 3001
        # news.build_batches(batch_size=batch_size, num_epochs=epoch, shuffle=True)
        #
        # enc_input, dec_input, targets = news.next_feed()

        outputs = model.inference(sess, enc_input, enc_seq_len)

        outputs = news.decode(outputs)

        pick = random.randrange(0, len(outputs))
        input = news.decode([news.description[pick]], True)
        expect = news.decode([news.headlines[pick]], True)
        # outputs = news.cut_eos(outputs[pick])
        outputs = outputs[pick]

        # print ("\n정확도:", accuracy)
        print ("랜덤 결과\n",)
        print ("    입력값:", input)
        print ("    실제값:", expect)
        print ("    예측값:", ' '.join(outputs))


def main(_):
    news = News()

    tf.set_random_seed(1)
    news.load_vocab(FLAGS.voc_path)
    news.read_data(FLAGS.data_path)

    if FLAGS.train:
        train(news, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
    elif FLAGS.test:
        test(news, batch_size=FLAGS.batch_size)
    elif FLAGS.inference:
        inference(news, batch_size=FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()
