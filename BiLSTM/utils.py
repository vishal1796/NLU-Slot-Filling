import numpy
from sklearn.metrics import precision_recall_fscore_support


def getUserUtterMaxlen(words_train):
    maxlen = 0
    for index, utter in enumerate(words_train):
        utter_len = len(utter)
        if utter_len > maxlen:
            maxlen = utter_len     
    return maxlen

def getNLUpred(tag_probs, tag_mask, id2tag, threshold=0.391):
    tag_txt = list()
    pred_tag_ids = numpy.argmax(tag_probs, axis=-1) * tag_mask
    for sample in pred_tag_ids:
        line_txt = [id2tag[tag_id] for tag_id in sample if tag_id != 0]
        tag_txt.append(' '.join(line_txt))
    return numpy.asarray(tag_txt)


def to_categorical(y_seq, nb_classes):
    ''' transform into a 1hot matrix
        Input:
            y_seq: shape = (sample_nb, maxlen_userUtter), elements are token ids.
            nb_classes: scalar, tag_vocab_size
        Output:
            Y: shape = (sample_nb, maxlen_userUtter, tag_vocab_size)
    '''
    Y = numpy.zeros(y_seq.shape + (nb_classes,))
    for sample_idx, sample in enumerate(y_seq):
        for tag_idx, tag in enumerate(sample):
            if tag != 0:
                Y[sample_idx, tag_idx, int(tag)] = 1
    return Y


def writeUtterTagTxt(utter_txt, tag_txt, target_fname):
    with open(target_fname, 'wb') as f:
        for (utter, tag) in zip(utter_txt, tag_txt):
            tag_new = [token for token in tag]
            new_line = '{}\n{}'.format(utter, ' '.join(tag_new))
            f.write('{}\n'.format(new_line))


def calculate_FrameAccuracy(pred, true):
    ''' calculate frame-level accuracy = hit / sample_nb
            inputs:
                pred: shape = (sample_nb, dim_size), predicted ids matrix
                true: shape is the same to pred, true ids matrix
            Outputs:
                accuracy_frame
    '''
    compare_array = numpy.all((pred - true) == 0, axis=-1)
    hit = numpy.sum(compare_array.astype(int))
    sample_nb = true.shape[0]
    accuracy_frame = hit * 1. / sample_nb
    return accuracy_frame


def eval_slotTagging(tag_probs, mask_array, tag_trueLabel, O_id):
    ''' Evaluation for slot tagging.
        
        Inputs:
            tag_probs: shape = (sample_nb, maxlen_userUtter, tag_vocab_size), predicted probs
            mask_array: shape = (sample_nb, maxlen_userUtter), mask array with 0s for padding.
            tag_trueLabel: shape is the same to tag_probs, indicator sparse matrix. If all zeros in one sample, the padding is assumed.
            id2tag: dict of id to tag string
            conll_fname: file name of .conll format that is suitable for conlleval.pl as input
        Outputs: 
            precision, recall, and f1_score at token level using conlleval.pl, FYI, 'O' is not counted as a token.
            accuracy at frame level.
    '''
    pred_tag_ids_masked = numpy.argmax(tag_probs, axis=-1) * mask_array
    true_tag_ids_masked = numpy.argmax(tag_trueLabel, axis=-1) * mask_array
    pred_tag_ids_noO = numpy.array(pred_tag_ids_masked)
    true_tag_ids_noO = numpy.array(true_tag_ids_masked)
    pred_tag_ids_noO[pred_tag_ids_masked == O_id] = 0  # exclude 'O' token
    true_tag_ids_noO[true_tag_ids_masked == O_id] = 0  # exclude 'O' token
    nb_classes = tag_probs.shape[-1]
    pred_tag_1hot_noO = to_categorical(pred_tag_ids_noO, nb_classes)
    true_tag_1hot_noO = to_categorical(true_tag_ids_noO, nb_classes)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_tag_1hot_noO.ravel(), pred_tag_1hot_noO.ravel(), beta=1.0, pos_label=1, average='binary')
    accuracy_frame = calculate_FrameAccuracy(pred_tag_ids_masked, true_tag_ids_masked)
    return (precision, recall, f1_score, accuracy_frame)