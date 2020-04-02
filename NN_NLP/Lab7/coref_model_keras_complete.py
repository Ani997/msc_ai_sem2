import tensorflow as tf
from keras import Input,Model
from keras import backend as K
from keras.layers import Dropout,Dense,LSTM,Bidirectional,Lambda,Reshape
import numpy as np
import json,time,metrics,collections,random


class CorefModel(object):
  def __init__(self,embedding_path, embedding_size):
    self.embedding_path = embedding_path
    self.embedding_size = embedding_size
    self.embedding_dropout_rate = 0.5
    self.max_ant = 250
    self.hidden_size = 50
    self.ffnn_layer = 2
    self.hidden_dropout_rate = 0.2
    self.neg_ratio = 2
    self.embedding_dict = self.load_embeddings()

  def load_embeddings(self):
    print("Loading word embeddings from {}...".format(self.embedding_path))
    embeddings = collections.defaultdict(lambda: np.zeros(self.embedding_size))
    for line in open(self.embedding_path):
      splitter = line.find(' ')
      emb = np.fromstring(line[splitter + 1:], np.float32, sep=' ')
      assert len(emb) == self.embedding_size
      embeddings[line[:splitter]] = emb
    print("Finished loading word embeddings")
    return embeddings

  def build(self):
    word_embeddings = Input(shape=(None,None,self.embedding_size,))
    mention_pairs = Input(shape=(None,4,),dtype='int32')


    word_embeddings_no_batch = Lambda(lambda x: K.squeeze(x,0))(word_embeddings)
    word_embeddings_no_batch = Dropout(self.embedding_dropout_rate)(word_embeddings_no_batch)
    word_output = Bidirectional(LSTM(self.hidden_size,recurrent_dropout=self.hidden_dropout_rate,return_sequences=True))(word_embeddings_no_batch)
    word_output = Bidirectional(LSTM(self.hidden_size,recurrent_dropout=self.hidden_dropout_rate,return_sequences=True))(word_output)

    flatten_word_output = Lambda(lambda x:K.reshape(x, [-1, 2 * self.hidden_size]))(word_output)
    flatten_word_output = Dropout(self.hidden_dropout_rate)(flatten_word_output)

    mention_pair_emb = Lambda(lambda x: K.gather(x[0], x[1]))([flatten_word_output,mention_pairs])
    ffnn_input = Reshape((-1,8*self.hidden_size))(mention_pair_emb)

    for i in range(self.ffnn_layer):
      ffnn_output = Dense(self.hidden_size,activation='relu')(ffnn_input)
      ffnn_output = Dropout(self.hidden_dropout_rate)(ffnn_output)
      ffnn_input = ffnn_output
    mention_pair_scores = Dense(1,activation='sigmoid')(ffnn_input)
    mention_pair_scores = Lambda(lambda x: K.squeeze(x,2))(mention_pair_scores)

    self.model = Model(inputs=[word_embeddings,mention_pairs],outputs=mention_pair_scores)
    self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    self.model.summary()


  def get_feed_dict_list(self, path,is_training=False):
    feed_dict_list = []
    for line in open(path):
      doc = json.loads(line)
      clusters = doc['clusters']
      if len(clusters) == 0:
        continue
      gold_mentions = sorted([tuple(m) for cl in clusters for m in cl])
      num_mention = len(gold_mentions)
      gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}

      cluster_ids = [0]*num_mention
      for cid, cluster in enumerate(clusters):
        for mention in cluster:
          cluster_ids[gold_mention_map[tuple(mention)]] = cid

      raw_starts, raw_ends = zip(*gold_mentions)
      starts, ends = [],[]
      sentences = doc['sentences']
      sent_lengths = [len(sent) for sent in sentences]
      max_sent_length = max(sent_lengths)

      word_emb = np.zeros([1,len(sentences), max_sent_length, self.embedding_size])
      raw_pre,padded_pre = 0,0
      for i, sent in enumerate(sentences):
        #to associate the gold mention indices with padded sentences
        for s, e in gold_mentions:
          if raw_pre <=s <=e < raw_pre+len(sent):
            starts.append(s-raw_pre+padded_pre)
            ends.append(e-raw_pre+padded_pre)
        raw_pre+= len(sent)
        padded_pre+=max_sent_length

        for j, word in enumerate(sent):
          word_emb[0, i, j] = self.embedding_dict[word.lower()]

      mention_pairs = [[]]
      mention_pair_labels = [[]]
      raw_mention_pairs = []
      if is_training:
        for ana in range(num_mention):
          pos = 1
          s = 0 if ana < self.max_ant else (ana - self.max_ant)
          for ant in range(s,ana):
            l = cluster_ids[ana] == cluster_ids[ant]
            if l:
              pos+=self.neg_ratio
              mention_pairs[0].append([starts[ana],ends[ana],starts[ant],ends[ant]])
              mention_pair_labels[0].append(1)
            elif pos > 0:
              pos -=1
              mention_pairs[0].append([starts[ana],ends[ana],starts[ant],ends[ant]])
              mention_pair_labels[0].append(0)
      else:
        for ana in range(num_mention):
          s = 0 if ana < self.max_ant else (ana - self.max_ant)
          for ant in range(s,ana):
            mention_pairs[0].append([starts[ana], ends[ana], starts[ant], ends[ant]])
            raw_mention_pairs.append([(raw_starts[ana], raw_ends[ana]), (raw_starts[ant], raw_ends[ant])])

      mention_pairs, mention_pair_labels = np.array(mention_pairs),np.array(mention_pair_labels)

      feed_dict_list.append((
        word_emb,
        mention_pairs,
        mention_pair_labels,
        clusters,
        raw_mention_pairs
      ))

    return feed_dict_list

  def get_predicted_clusters(self, mention_pairs):
    mention_to_predicted = {}
    predicted_clusters = []
    for anaphora, predicted_antecedent in mention_pairs:
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      predicted_clusters[predicted_cluster].append(anaphora)
      mention_to_predicted[anaphora] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, mention_pairs, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(mention_pairs)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

  def batch_generator(self, fd_list):
    random.shuffle(fd_list)
    for word_embeddings, mention_pairs, mention_pair_labels, _, _ in fd_list:
      yield [word_embeddings, mention_pairs], mention_pair_labels

  def train(self, train_path, dev_path, test_path, epochs):
    train_fd_list = self.get_feed_dict_list(train_path, is_training=True)
    print("Load {} training documents from {}".format(len(train_fd_list), train_path))

    dev_fd_list = self.get_feed_dict_list(dev_path)
    print("Load {} dev documents from {}".format(len(dev_fd_list), dev_path))

    test_fd_list = self.get_feed_dict_list(test_path)
    print("Load {} test documents from {}".format(len(test_fd_list), test_path))

    start_time = time.time()
    for epoch in range(epochs):
      print("\nStarting training epoch {}/{}".format(epoch + 1, epochs))
      epoch_time = time.time()

      self.model.fit_generator(self.batch_generator(train_fd_list), steps_per_epoch=2775)

      print("Time used for epoch {}: {}".format(epoch + 1, self.time_used(epoch_time)))
      dev_time = time.time()
      print("Evaluating on dev set after epoch {}/{}:".format(epoch + 1, epochs))
      self.eval(dev_fd_list)
      print("Time used for evaluate on dev set: {}".format(self.time_used(dev_time)))

    print("\nTraining finished!")
    print("Time used for training: {}".format(self.time_used(start_time)))

    print("\nEvaluating on test set:")
    test_time = time.time()
    self.eval(test_fd_list)
    print("Time used for evaluate on test set: {}".format(self.time_used(test_time)))

  def eval(self, eval_fd_list):
    coref_evaluator = metrics.CorefEvaluator()

    for word_embeddings, mention_pairs, _, gold_clusters, raw_mention_pairs in eval_fd_list:
      mention_pair_scores = self.model.predict_on_batch([word_embeddings, mention_pairs])
      predicted_antecedents = {}
      best_antecedent_scores = {}
      for (ana, ant), score in zip(raw_mention_pairs, mention_pair_scores[0]):
        if score >= 0.5 and score > best_antecedent_scores.get(ana,0):
          predicted_antecedents[ana] = ant
          best_antecedent_scores[ana] = score

      predicted_mention_pairs = [[k,v] for k,v in predicted_antecedents.items()]

      self.evaluate_coref(predicted_mention_pairs, gold_clusters, coref_evaluator)

    p, r, f = coref_evaluator.get_prf()
    print("Average F1 (py): {:.2f}%".format(f * 100))
    print("Average precision (py): {:.2f}%".format(p * 100))
    print("Average recall (py): {:.2f}%".format(r * 100))

  def time_used(self, start_time):
    curr_time = time.time()
    used_time = curr_time - start_time
    m = used_time // 60
    s = used_time - 60 * m
    return "%d m %d s" % (m, s)

if __name__ == '__main__':
  embedding_path = 'glove.6B.100d.txt.filtered'
  train_path = 'train.english.20sent.jsonlines'
  dev_path = 'dev.english.20sent.jsonlines'
  test_path = 'test.english.20sent.jsonlines'
  embedding_size = 100
  model = CorefModel(embedding_path,embedding_size)
  model.build()
  model.train(train_path,dev_path,test_path,5)