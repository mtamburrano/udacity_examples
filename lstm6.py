# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import time
from random import randint

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()
  
text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


alphabet_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
vocabulary_size = (alphabet_size)**2
first_letter = ord(string.ascii_lowercase[0])

################### UTILS ###################

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '
    
def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]


def printData(batch):
    batch_str = [reverse_dictionary[batch[i]] for i in range(len(batch))]
    print("DATA: ",batch_str)
    
def printLabels(labels):
    print(labels[0])
    labels_str = [characters(labels[i].reshape((1,alphabet_size))) for i in range(labels.shape[0])]
    print("LABELS: ",labels_str)
    
def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  for b in batches:
    printData(b)
    
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_sample():
  """Generate a random column of probabilities."""
  random_int = randint(0,vocabulary_size-1)
  sample = np.empty(shape=(1), dtype=np.int)
  sample[0] = random_int
  return sample
    
###########################################

#################### BUILD DATASET ##################

#creates all possible bigrams with our alphabet
def create_words():
    words = []
    onechar_voc = []
    onechar_voc.append(' ')
    for i in range(len(string.ascii_lowercase)):
        onechar_voc.append(chr(i + first_letter))
    print(onechar_voc)
    for i in range(len(onechar_voc)):
        word_i = onechar_voc[i]
        for j in range(len(onechar_voc)):
            word_ij = word_i + onechar_voc[j]
            words.append(word_ij)
    return words

def build_dataset(words):
  dictionary = dict()
  data = list()
  for word in words:
    dictionary[word] = len(dictionary)
    data.append(len(dictionary)-1)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, dictionary, reverse_dictionary


words = create_words()
data, dictionary, reverse_dictionary = build_dataset(words)

################################################


#################### BUILD BATCHES ##################

batch_size=64
num_unrollings=10
embedding_size= 16

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size), dtype=np.int32)
    for b in range(self._batch_size):
      batch[b] = dictionary[self._text[self._cursor[b]]+self._text[self._cursor[b]+1]]
      self._cursor[b] = (self._cursor[b] + 1) % (self._text_size -1)
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

################################################

####################### MODEL #####################

num_nodes = 64
dropout_prob = 0.5

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # [ix, fx, cx, ox]
  x_m = tf.Variable(tf.truncated_normal([embedding_size, num_nodes*4], -0.1, 0.1))
  # [im, fm, cm, om]
  o_m = tf.Variable(tf.truncated_normal([num_nodes, num_nodes*4], -0.1, 0.1))
  # [ib, fb, cb, ob]
  b_vec = tf.Variable(tf.zeros([1, 4*num_nodes]))

  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, alphabet_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([alphabet_size]))
  
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    m = tf.matmul(i, x_m) + tf.matmul(o, o_m) + b_vec
    #offsets
    off = [0, num_nodes, num_nodes*2, num_nodes*3, num_nodes*4]
    
    input_gate = tf.sigmoid(m[:, off[0]:off[1]])
    forget_gate = tf.sigmoid(m[:, off[1]:off[2]])
    update = m[:, off[2]:off[3]]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(m[:, off[3]:off[4]])
    
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  train_labels = list()
  for _ in range(num_unrollings):
    train_data.append(
      tf.placeholder(tf.int32, shape=[batch_size]))
    train_labels.append(
      tf.placeholder(tf.float32, shape=[batch_size, alphabet_size]))
    
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_data:
    embed = tf.nn.embedding_lookup(embeddings, i)
    embed = tf.nn.dropout(embed, dropout_prob)
    output, state = lstm_cell(embed, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    concat_outputs = tf.concat(0, outputs)
    concat_outputs = tf.nn.dropout(concat_outputs, dropout_prob)
    logits = tf.nn.xw_plus_b(concat_outputs, w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.65, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1])
  embed_sample = tf.nn.embedding_lookup(embeddings, sample_input)
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    embed_sample, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

###############################################################

################# TRAIN ##################################

num_steps = 200001
summary_frequency = 100


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    labels_batches = []
    feed_dict = dict()
    for i in range(num_unrollings):
        
      l_onehot_batch = np.zeros(shape=(batch_size, alphabet_size), dtype=np.float)
      lab = batches[i+1]
      for b in range(batch_size):
            bigram = reverse_dictionary[lab[b]]
            gt_letter = bigram[1]
            l_onehot_batch[b, char2id(gt_letter)] = 1.0
      labels_batches.append(l_onehot_batch)
      
      feed_dict[train_data[i]] = batches[i]
      feed_dict[train_labels[i]] = l_onehot_batch
    
    #printBatch(batches[0])
    #printLabels(labels_batches[0])
    #printBatch(batches[1])
    #printLabels(labels_batches[1])
    #printBatch(batches[2])
    #printLabels(labels_batches[2])
    
    _, l, predictions, logs, last_output, lr = session.run(
      [optimizer, loss, train_prediction, logits, output, learning_rate], feed_dict=feed_dict)
    
    #print("predictions chars: ", characters(predictions))
    
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      
      labels = np.concatenate(list(labels_batches)[:])
      #print(labels)
      #print(predictions.shape)
      #print(labels_batches[0].shape)
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = random_sample()
          sentence = reverse_dictionary[feed[0]]
          reset_sample_state.run()
          #print("init sentence: -%s-" % sentence)
          for _ in range(79):
            #print(" --- feed: ", reverse_dictionary[feed[0]])
            prediction = sample_prediction.eval({sample_input: feed})
            #print(prediction)
            #print("prediction: -%s-" % characters(prediction)[0])
            #print("bigram: -%s-" % (sentence[-1]+characters(prediction)[0]))
            #print(next_bigram)
            next_bigram = dictionary[sentence[-1]+characters(sample(prediction))[0]]
            feed = np.array([next_bigram])
            sentence += reverse_dictionary[next_bigram][-1]
            
          print(sentence)
        print('=' * 80)

#################################



