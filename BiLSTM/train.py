import cPickle
import numpy
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from utils import getNLUpred, getUserUtterMaxlen, to_categorical, writeUtterTagTxt, eval_slotTagging


train_set, valid_set, dicts = cPickle.load(open('dataset/atis.pkl', 'rb'))
word2id, tag2id = dicts['words2idx'], dicts['labels2idx']
train_x, _, train_label = train_set
val_x, _, val_label = valid_set


train_x = [[x+1 for x in w] for w in train_x]
train_label = [[x for x in w] for w in train_label]
val_x = [[x+1 for x in w] for w in val_x]
val_label = [[x for x in w] for w in val_label]
word2id = {k:v+1 for k,v in word2id.items()}

id2word  = {word2id[k]:k for k in word2id}
id2tag = {tag2id[k]:k for k in tag2id}

maxlen = getUserUtterMaxlen(train_x)
n_classes = len(id2tag)
n_vocab = len(id2word)

train_utter_txt = [list(map(lambda x: id2word[x], w)) for w in train_x]
train_tag_txt = [list(map(lambda x: id2tag[x], y)) for y in train_label]
train_target_fname = '/output/train_utter_slot.txt'
writeUtterTagTxt(train_utter_txt, train_tag_txt, train_target_fname)

dev_utter_txt = [list(map(lambda x: id2word[x], w)) for w in val_x]
dev_tag_txt = [list(map(lambda x: id2tag[x], y)) for y in val_label]
dev_target_fname = '/output/dev_utter_slot.txt'
writeUtterTagTxt(dev_utter_txt, dev_tag_txt, dev_target_fname)


for idx, seq in enumerate(train_x):
    seq = seq[:maxlen]
    seq += [0]*(maxlen - len(seq))
    train_x[idx] = seq
    
for idx, seq in enumerate(val_x):
    seq = seq[:maxlen]
    seq += [0]*(maxlen - len(seq))
    val_x[idx] = seq

for idx, label in enumerate(train_label):
    label = label[:maxlen]
    label += [126]*(maxlen - len(label))
    train_label[idx] = label
    
for idx, label in enumerate(val_label):
    label = label[:maxlen]
    label += [126]*(maxlen - len(label))
    val_label[idx] = label


train_label = to_categorical(numpy.asarray(train_label), n_classes)
val_label = to_categorical(numpy.asarray(val_label), n_classes)


X_train = numpy.array(train_x)
tag_train = train_label
X_dev = numpy.array(val_x)
tag_dev = val_label

mask_array_train = numpy.zeros_like(X_train)
mask_array_train[X_train != 0] = 1
mask_array_dev = numpy.zeros_like(X_dev)
mask_array_dev[X_dev != 0] = 1


words_input = Input(shape=(maxlen,), dtype='int32', name='words_input')
embeddings = Embedding(input_dim=n_vocab + 1, output_dim=512, input_length=maxlen, mask_zero=True)(words_input)
embeddings = Dropout(0.5)(embeddings)
lstm_forward = LSTM(units=128, return_sequences=True, name='LSTM_forward')(embeddings)
lstm_forward = Dropout(0.5)(lstm_forward)
lstm_backward = LSTM(units=128, return_sequences=True, go_backwards=True, name='LSTM_backward')(embeddings)
lstm_backward = Dropout(0.5)(lstm_backward)
lstm_concat = Concatenate(axis=-1, name='merge_bidirections')([lstm_forward, lstm_backward])
slot_output = TimeDistributed(Dense(units=n_classes, activation='softmax'), name='slot_output')(lstm_concat)
model = Model(inputs=words_input, outputs=slot_output)
model.compile(optimizer='adam', sample_weight_mode='temporal', loss='categorical_crossentropy')

for ep in xrange(300):
    print('<Epoch {}>'.format(ep))
    model.fit(X_train, tag_train, sample_weight=mask_array_train, batch_size=32, epochs=1, verbose=2)
    tag_probs = model.predict(X_dev)
    precision_tag, recall_tag, fscore_tag, accuracy_frame_tag = eval_slotTagging(tag_probs, mask_array_dev, tag_dev, tag2id['O'])
    print('SlotTagging: ep={}, precision={:.4f}, recall={:.4f}, fscore={:.4f}, accuracy_frame={:.4f}'.format(ep, precision_tag, recall_tag, fscore_tag, accuracy_frame_tag))
    dev_tag_pred_txt = getNLUpred(tag_probs, mask_array_dev, id2tag)
    dev_results_fname = '/output/dev_result.txt'
    writeUtterTagTxt(dev_utter_txt, dev_tag_pred_txt, dev_results_fname)
    
    weights_fname = '/output/ep={}_tagF1={:.4f}frameAcc={:.4f}.h5'.format(ep, fscore_tag, accuracy_frame_tag)
    print('Saving Model: {}'.format(weights_fname))
    model.save_weights(weights_fname, overwrite=True)