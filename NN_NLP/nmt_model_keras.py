from keras.layers import Embedding,LSTM,Dropout,Dense,Layer, Multiply
from keras import Model,Input
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import keras.backend as K
# from keras.activations import softmax
from tensorflow.keras.utils import plot_model
import collections
import numpy as np
import time
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

class NmtModel(object):
  def __init__(self,source_dict,target_dict,use_attention):
    self.hidden_size = 200
    self.embedding_size = 100
    self.hidden_dropout_rate=0.2
    self.embedding_dropout_rate = 0.2
    self.batch_size = 100
    self.max_target_step = 30
    self.vocab_target_size = len(target_dict.vocab)
    self.vocab_source_size = len(source_dict.vocab)
    self.target_dict = target_dict
    self.source_dict = source_dict
    self.SOS = target_dict.word2ids['<start>']
    self.EOS = target_dict.word2ids['<end>']
    self.use_attention = use_attention

    print("source vocab: %d, target vocab:%d" % (self.vocab_source_size,self.vocab_target_size))


  def build(self):
    source_words = Input(shape=(None,),dtype='int32')
    target_words = Input(shape=(None,), dtype='int32')

    # Task 1 encoder
    
    # Start
 
    embedding_source = Embedding(input_dim=self.vocab_source_size, embeddings_initializer='random_uniform', mask_zero=True, output_dim=self.embedding_size,input_length=source_words.shape[1])
    source_words_embeddings = embedding_source(source_words)
    source_words_embeddings = Dropout(self.embedding_dropout_rate)(source_words_embeddings)
    encoder_lstm = LSTM(self.hidden_size,recurrent_dropout=self.hidden_dropout_rate,return_sequences=True,return_state=True)
    encoder_outputs,encoder_state_h,encoder_state_c = encoder_lstm(source_words_embeddings)

    embedding_target = Embedding(input_dim=self.vocab_target_size, embeddings_initializer='random_uniform', mask_zero=True, output_dim=self.embedding_size,input_length=target_words.shape[1])
    target_words_embeddings = embedding_target(target_words)
    target_words_embeddings = Dropout(self.embedding_dropout_rate)(target_words_embeddings)

    # End Task 1

    encoder_states = [encoder_state_h,encoder_state_c]

    decoder_lstm = LSTM(self.hidden_size,recurrent_dropout=self.hidden_dropout_rate,return_sequences=True,return_state=True)
    decoder_outputs_train,_,_ = decoder_lstm(target_words_embeddings,initial_state=encoder_states)


    if self.use_attention:
        decoder_attention = AttentionLayer()
        decoder_outputs_train = decoder_attention([encoder_outputs,decoder_outputs_train])

    decoder_dense = Dense(self.vocab_target_size,activation='softmax')
    decoder_outputs_train = decoder_dense(decoder_outputs_train)

    adam = Adam(lr=0.01,clipnorm=5.0)
    self.train_model = Model([source_words,target_words], decoder_outputs_train)
    self.train_model.compile(optimizer=adam,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    self.train_model.summary()
    plot_model(self.train_model, to_file='train_model.png')


    #Inference Models

    self.encoder_model = Model(source_words,[encoder_outputs,encoder_state_h,encoder_state_c])
    self.encoder_model.summary()
    plot_model(self.encoder_model, to_file='encoder_model.png')

    decoder_state_input_h = Input(shape=(self.hidden_size,))
    decoder_state_input_c = Input(shape=(self.hidden_size,))
    encoder_outputs_input = Input(shape=(None,self.hidden_size,))

    """
    Task 2 decoder for inference
    
    Start
    """
    decoder_state = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs_test,decoder_state_output_h,decoder_state_output_c = decoder_lstm(target_words_embeddings,initial_state=decoder_state)
    if self.use_attention:
        decoder_attention = AttentionLayer()
        decoder_outputs_test = decoder_attention([encoder_outputs_input,decoder_outputs_test])
    
    decoder_outputs_test = decoder_dense(decoder_outputs_test)    
    
    """
    End Task 2 
    """

    self.decoder_model = Model([target_words,decoder_state_input_h,decoder_state_input_c,encoder_outputs_input],
                               [decoder_outputs_test,decoder_state_output_h,decoder_state_output_c])
    self.decoder_model.summary()
    plot_model(self.decoder_model, to_file='decoder_model.png')



  def time_used(self, start_time):
    curr_time = time.time()
    used_time = curr_time-start_time
    m = used_time // 60
    s = used_time - 60 * m
    return "%d m %d s" % (m, s)

  def train(self,train_data,dev_data,test_data, epochs):
    start_time = time.time()
    for epoch in range(epochs):
      print("Starting training epoch {}/{}".format(epoch + 1, epochs))
      epoch_time = time.time()
      source_words_train, target_words_train, target_words_train_labels = train_data

      self.train_model.fit([source_words_train,target_words_train],target_words_train_labels,batch_size=self.batch_size)

      print("Time used for epoch {}: {}".format(epoch + 1, self.time_used(epoch_time)))
      dev_time = time.time()
      print("Evaluating on dev set after epoch {}/{}:".format(epoch + 1, epochs))
      self.eval(dev_data)
      print("Time used for evaluate on dev set: {}".format(self.time_used(dev_time)))

    print("Training finished!")
    print("Time used for training: {}".format(self.time_used(start_time)))

    print("Evaluating on test set:")
    test_time = time.time()
    self.eval(test_data)
    print("Time used for evaluate on test set: {}".format(self.time_used(test_time)))



  def get_target_sentences(self, sents,vocab,reference=False):
    str_sents = []
    num_sent, max_len = sents.shape
    for i in range(num_sent):
      str_sent = []
      for j in range(max_len):
        t = sents[i,j].item()
        if t == self.SOS:
          continue
        if t == self.EOS:
          break

        str_sent.append(vocab[t])
      if reference:
        str_sents.append([str_sent])
      else:
        str_sents.append(str_sent)
    return str_sents


  def eval(self, dataset):
    source_words, target_words_labels = dataset
    vocab = self.target_dict.vocab

    encoder_outputs, state_h,state_c = self.encoder_model.predict(source_words,batch_size=self.batch_size)
    predictions = []
    step_target_words = np.ones([source_words.shape[0],1]) * self.SOS
    for _ in range(self.max_target_step):
      step_decoder_outputs, state_h,state_c = self.decoder_model.predict([step_target_words,state_h,state_c,encoder_outputs],batch_size=self.batch_size)
      step_target_words = np.argmax(step_decoder_outputs,axis=2)
      predictions.append(step_target_words)

    candidates = self.get_target_sentences(np.concatenate(predictions,axis=1),vocab)
    references = self.get_target_sentences(target_words_labels,vocab,reference=True)
    print("candidates: ",' '.join(candidates[10]))
    print("references: ",' '.join(references[10][0]))
    score = corpus_bleu(references,candidates)
    print("Model BLEU score: %.2f" % (score*100.0))

class AttentionLayer(Layer):
  def compute_mask(self, inputs, mask=None):
    if mask == None:
      return None
    return mask[1]

  def compute_output_shape(self, input_shape):
    return (input_shape[1][0],input_shape[1][1],input_shape[1][2]*2)


  def call(self, inputs, mask=None):
    encoder_outputs, decoder_outputs = inputs

    """
    Task 3 attention
    
    Start
    """
    luong_score = tf.matmul(decoder_outputs, encoder_outputs, transpose_b=True)    
    alignment = tf.nn.softmax(luong_score, axis=2)
    context = tf.matmul(K.expand_dims(alignment,axis=2), K.expand_dims(encoder_outputs,axis=1))
    encoder_vector = K.squeeze(context,axis=2)    
    
    """
    End Task 3
    """
    # [batch,max_dec,2*emb]
    new_decoder_outputs = K.concatenate([decoder_outputs, encoder_vector])

    return new_decoder_outputs


class LanguageDict():
  def __init__(self, sents):
    word_counter = collections.Counter(tok.lower() for sent in sents for tok in sent)

    self.vocab = []
    self.vocab.append('<pad>') #zero paddings
    self.vocab.append('<unk>')
    self.vocab.extend([t for t,c in word_counter.items() if c > 10])

    self.word2ids = {w:id for id, w in enumerate(self.vocab)}
    self.UNK = self.word2ids['<unk>']
    self.PAD = self.word2ids['<pad>']



def load_dataset(source_path,target_path, max_num_examples=30000):
  source_lines = open(source_path).readlines()
  target_lines = open(target_path).readlines()
  assert len(source_lines) == len(target_lines)
  if max_num_examples > 0:
    max_num_examples = min(len(source_lines), max_num_examples)
    source_lines = source_lines[:max_num_examples]
    target_lines = target_lines[:max_num_examples]

  source_sents = [[tok.lower() for tok in sent.strip().split(' ')] for sent in source_lines]
  target_sents = [[tok.lower() for tok in sent.strip().split(' ')] for sent in target_lines]
  for sent in target_sents:
    sent.append('<end>')
    sent.insert(0,'<start>')

  source_lang_dict = LanguageDict(source_sents)
  target_lang_dict = LanguageDict(target_sents)

  unit = len(source_sents)//10

  source_words = [[source_lang_dict.word2ids.get(tok,source_lang_dict.UNK) for tok in sent] for sent in source_sents]
  source_words_train = pad_sequences(source_words[:8*unit],padding='post')
  source_words_dev = pad_sequences(source_words[8*unit:9*unit],padding='post')
  source_words_test = pad_sequences(source_words[9*unit:],padding='post')

  eos = target_lang_dict.word2ids['<end>']

  target_words = [[target_lang_dict.word2ids.get(tok,target_lang_dict.UNK) for tok in sent[:-1]] for sent in target_sents]
  target_words_train = pad_sequences(target_words[:8*unit],padding='post')

  target_words_train_labels = [sent[1:]+[eos] for sent in target_words[:8*unit]]
  target_words_train_labels = pad_sequences(target_words_train_labels,padding='post')
  target_words_train_labels = np.expand_dims(target_words_train_labels,axis=2)

  target_words_dev_labels = pad_sequences([sent[1:] + [eos] for sent in target_words[8 * unit:9 * unit]], padding='post')
  target_words_test_labels = pad_sequences([sent[1:] + [eos] for sent in target_words[9 * unit:]], padding='post')

  train_data = [source_words_train,target_words_train,target_words_train_labels]
  dev_data = [source_words_dev,target_words_dev_labels]
  test_data = [source_words_test,target_words_test_labels]

  return train_data,dev_data,test_data,source_lang_dict,target_lang_dict



if __name__ == '__main__':
  max_example = 30000
  use_attention = True
  train_data, dev_data, test_data, source_dict, target_dict = load_dataset("data.30.vi","data.30.en",max_num_examples=max_example)
  print("read %d/%d/%d train/dev/test batches" % (len(train_data[0]),len(dev_data[0]), len(test_data[0])))

  model = NmtModel(source_dict,target_dict,use_attention)
  model.build()
  # plot_model(model, to_file='model.png')
  model.train(train_data,dev_data,test_data,10)


