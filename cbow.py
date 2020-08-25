# 参考 lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 2020-8-25

# 输入层 C*V   w=V*N  h=1*N (C个1*N向量，取平均)   w'=N*V   u=h*w'=1*V
# V=单词个数(one-hot维度：1*V)  C=2A   A=滑动窗口大小  C=上下文单词数   N=最终获得词向量的维度

import math
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from load_data import *

with open('corpus.txt', 'r', encoding='utf-8')as file:
    text = file.read()
    corpus = []
    for i in text:
        corpus.append(i)
    print(corpus)
    # 新增
class CBOW:
    def __init__(self, tran_flag=True):
        self.data_index = 0
        self.min_count = 5  # 默认最低频次的单词
        self.batch_size = 100  # 每次迭代训练选取的样本数目
        self.embedding_size = 300  # 生成词向量的维度
        self.window_size = 2  # 考虑前后几个词，窗口大小
        self.num_steps = 4000  # 定义最大迭代次数，创建并设置默认的session，开始实际训练
        self.num_sampled = 50  # 负采样数量
        # self.trainfilepath = 'E:/Pycharm文件/nlp/CBOW模型/test2/corpus.txt'
        self.modelpath1 = 'E:/Pycharm文件/nlp/CBOW模型/test2/embedding.txt'
        self.modelpath2 = 'E:/Pycharm文件/nlp/CBOW模型/test2/softmax.txt'
        self.dataset = corpus
        # self.dataset = DataLoader().dataset
        self.words = self.read_data(self.dataset)
        self.tran_flag = tran_flag

    # 定义读取数据的函数，并把数据转成列表
    def read_data(self, dataset):
        words = []
        for data in dataset:
            words.extend(data)  # extend传入的是可迭代对象，这里append应该也是可以的，因为我传入的是一个汉字
        return words

    # 创建数据集
    def build_dataset(self, words, min_count):
        # 创建词汇表，过滤低频次词语，这里使用的人是mincount>=5，其余单词认定为Unknown,编号为0,
        # 这一步在gensim提供的wordvector中，采用的是minicount的方法
        # 对原words列表中的单词使用字典中的ID进行编号，即将单词转换成整数，储存在data列表中，同时对UNK进行计数
        count = [['UNK', -1]]
        reserved_words = [item for item in collections.Counter(words).most_common() if item[1] >= min_count]
        # 例子：item = ('hao', 2)，item[1]代表出现的次数2
        count.extend(reserved_words)  # count是按出现次数排列的（字，次数）列表
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)  # 获得字的出现位置
        data = list()
        unk_count = 0
        for word in words:  # 得到每一个字的【索引】
            if word in dictionary:
                index = dictionary[word]  # 字在已有字典内时，索引是它出现的位置
            else:
                index = 0  # dictionary['UNK']  # 字不在字典里时，索引为0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count  # 将未知字符的次数赋值给第一个字，即Unk_Count
        print(len(count))
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    # 生成训练样本

    def generate_batch(self, batch_size, skip_window, data):
        # 该函数根据训练样本中词的顺序抽取形成训练集
        # batch_size:每个批次训练多少样本
        # skip_window:单词最远可以联系的距离（本次实验设为5，即目标单词只能和相邻的两个单词生成样本），2*skip_window>=num_skips
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        buffer = collections.deque(maxlen=span)  # 指定一个固定长度的队列，新数据加入时自动剔除最老的那个数据

        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)  # 原代码data_indexdata_index 修改为data_index

        for i in range(batch_size):
            target = skip_window
            target_to_avoid = [skip_window]
            col_idx = 0
            for j in range(span):
                if j == span // 2:
                    continue   # 偶数不执行以下操作
                batch[i, col_idx] = buffer[j]
                col_idx += 1
            labels[i, 0] = buffer[target]

            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)

        assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
        # assert断言：申明其布尔值必须为真的判定，如果发生异常，就表示为假
        return batch, labels

    def train_wordvec(self, vocabulary_size, batch_size, embedding_size, window_size, num_sampled, num_steps, data):
        # 定义 CBOW的Word2Vec模型的网络结构
        graph = tf.Graph()
        with graph.as_default(), tf.device('/gpu:0'):
            train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            # softmax_weights = tf.Variable(tf.truncated_normal([embedding_size, vocabulary_size], stddev=1.0 / math.sqrt(embedding_size)))

            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
            # 与skipgram不同， cbow的输入是上下文向量的均值，因此需要做相应变换
            context_embeddings = []
            for i in range(2 * window_size):
                context_embeddings.append(tf.nn.embedding_lookup(embeddings, train_dataset[:, i]))
            avg_embed = tf.reduce_mean(tf.stack(axis=0, values=context_embeddings), 0, keep_dims=False)
            # 将训练数据按行重叠打包，之后求平均

            # if self.tran_flag:
            loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=avg_embed,
                                       labels=train_labels, num_sampled=num_sampled,
                                       num_classes=vocabulary_size))
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)  # 优化器
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            # else:
            #     logtis
            #     v * batch

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = self.generate_batch(batch_size, window_size, data)
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
            final_embeddings = normalized_embeddings.eval()
            print('avg_embed:')
            print(avg_embed)
            # print(softmax_weights)
            # print(context_embeddings)

        return final_embeddings

    # 保存embedding文件
    def save_embedding(self, final_embeddings, model_path1, reverse_dictionary):
        f = open(model_path1, 'w+', encoding='utf-8')
        for index, item in enumerate(final_embeddings):
            f.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
        f.close()
        # print(final_embeddings)
    # def save_softmax(self, softmax, model_path2):
    #     f = open(model_path2, 'w+', encoding='utf-8')
    #     f.write(softmax)

    #训练主函数
    def train(self):
        data, count, dictionary, reverse_dictionary = self.build_dataset(self.words, self.min_count)
        vocabulary_size = len(count)
        final_embeddings = self.train_wordvec(vocabulary_size, self.batch_size, self.embedding_size, self.window_size, self.num_sampled, self.num_steps, data)
        self.save_embedding(final_embeddings, self.modelpath1, reverse_dictionary)

def test():
    vector = CBOW()
    vector.train()

test()
