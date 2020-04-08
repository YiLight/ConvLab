import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
import random
import re
import numpy as np
from collections import OrderedDict
from random import shuffle
import pickle
from embeddings import GloveEmbedding, KazumaCharEmbedding

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import json
# import pandas as pd

from convlab.modules.word_dst.multiwoz.trade.trade_config import *

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

USE_CUDA = True if torch.cuda.is_available() else False
MAX_LENGTH = 10

args = {}
args['load_embedding'] = True
args["fix_embedding"] = False
if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")

replacements = []
class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, nb_train_vocab=0):
        super(TRADE, self).__init__()
        self.name = "TRADE"
        self.task = task
        self.hidden_size = hidden_size
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()
        self.data_dir = path

        self.encoder = EncoderRNN(
            self.lang.n_words, hidden_size, self.dropout, data_dir=path)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout,
                                 self.slots, self.nb_gate)

        # if path:
        #     if USE_CUDA:
        #         print("MODEL {} LOADED".format(str(path)))
        #         trained_encoder = torch.load(str(path) + '/enc.th')
        #         trained_decoder = torch.load(str(path) + '/dec.th')
        #     else:
        #         print("MODEL {} LOADED".format(str(path)))
        #         trained_encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
        #         trained_decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)

        #     self.encoder.load_state_dict(trained_encoder)
        #     self.decoder.load_state_dict(trained_decoder)

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self, dec_type):
        directory = 'save/TRADE-' + args["addName"] + args['dataset'] + str(self.task) + '/' + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, reset=0):
        if reset:
            self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(data, use_teacher_forcing,
                                                                                            slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            # [:,:len(self.point_slots)].contiguous(),
            data["generate_y"].contiguous(),
            data["y_lengths"])  # [:,:len(self.point_slots)])
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
                                       data["gating_label"].contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr

        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def optimize_GEM(self, clip):
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):
        # Build unknown mask for memory to encourage generalization
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial(
                [np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA:
                rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(
            story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(
            2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size,
                                                                                                     encoded_hidden,
                                                                                                     encoded_outputs,
                                                                                                     data[
                                                                                                         'context_len'],
                                                                                                     story, max_res_len,
                                                                                                     data['generate_y'],
                                                                                                     use_teacher_forcing,
                                                                                                     slot_temp)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k)
                                     for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(
                data_dev, False, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg == self.gating_dict["none"]:
                            continue
                        elif sg == self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e == 'EOS':
                                    break
                                else:
                                    st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(
                                    slot_temp[si] + "-" + str(st))
                        else:
                            predict_belief_bsz_ptr.append(
                                slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS':
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(
                                slot_temp[si] + "-" + str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"]
                                                   [bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                    print("True", set(data_dev["turn_belief"][bi]))
                    print("Pred", set(predict_belief_bsz_ptr), "\n")

        if args["genSample"]:
            json.dump(all_prediction, open(
                "all_prediction_{}.json".format(self.name), 'w'), indent=4)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr",
                                                                                      slot_temp)

        evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                              "Joint F1": F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        # (joint_acc_score_ptr + joint_acc_score_class)/2
        joint_acc_score = joint_acc_score_ptr
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
            return joint_acc_score

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(
                    set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(
                    set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total != 0 else 0
        turn_acc_score = turn_acc / float(total) if total != 0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / \
                float(precision + recall) if (precision + recall) != 0 else 0
        else:
            if len(pred) == 0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count
    
    def read_replacements(self):
        mapping_pair_url = os.path.join(self.data_dir, 'mapping.pair')
        fin_mapping_pair = open(mapping_pair_url, 'r')
        global replacements
        for line in fin_mapping_pair.readlines():
            tok_from, tok_to = line.replace('\n', '').split('\t')
            replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1, data_dir=''):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if args["load_embedding"]:
            with open(os.path.join(data_dir, 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad",
                  self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches,
                use_teacher_forcing, slot_temp):
        all_point_outputs = torch.zeros(
            len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(
            len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA:
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()

        # Get the slot embedding
        slot_emb_dict = {}
        for slot in slot_temp:
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA:
                    domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA:
                    slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            slot_emb_dict[slot] = domain_emb + slot_emb

        words_class_out = []
        # Compute pointer-generator output
        words_point_out = []
        counter = 0
        for slot in slot_temp:  # TODO: Parallel this part to make it train faster
            hidden = encoded_hidden
            words = []
            slot_emb = slot_emb_dict[slot]
            decoder_input = self.dropout_layer(
                slot_emb).expand(batch_size, self.hidden_size)
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(
                    decoder_input.expand_as(hidden), hidden)
                context_vec, logits, prob = self.attend(
                    encoded_outputs, hidden.squeeze(0), encoded_lens)
                if wi == 0:
                    all_gate_outputs[counter] = self.W_gate(context_vec)
                p_vocab = self.attend_vocab(
                    self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat(
                    [dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size())
                if USE_CUDA:
                    p_context_ptr = p_context_ptr.cuda()
                p_context_ptr.scatter_add_(1, story, prob)
                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words.append([self.lang.index2word[w_idx.item()]
                              for w_idx in pred_word])
                all_point_outputs[counter, :, wi, :] = final_p_vocab
                if use_teacher_forcing:
                    decoder_input = self.embedding(
                        target_batches[:, counter, wi])  # Chosen word is next input
                else:
                    decoder_input = self.embedding(pred_word)
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
            counter += 1
            words_point_out.append(words)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def cross_entropy(logits, target):
    batch_size = logits.size(0)
    log_probs_flat = functional.log_softmax(logits)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target)
    loss = losses_flat.sum() / batch_size
    return loss


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))

    # logits_flat: (batch * max_len, num_classes)
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def masked_binary_cross_entropy(logits, target, length):
    '''
    logits: (batch, max_len, num_class)
    target: (batch, max_len, num_class)
    '''
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    bce_criterion = nn.BCEWithLogitsLoss()
    loss = 0
    for bi in range(logits.size(0)):
        for i in range(logits.size(1)):
            if i < length[bi]:
                loss += bce_criterion(logits[bi][i], target[bi][i])
    loss = loss / length.float().sum()
    return loss


def masked_cross_entropy_(logits, target, length, take_log=False):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))

    # logits_flat: (batch * max_len, num_classes)
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    if take_log:
        logits_flat = torch.log(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(logits_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def masked_coverage_loss(coverage, attention, length):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    mask = sequence_mask(sequence_length=length)
    min_ = torch.min(coverage, attention)
    mask = mask.unsqueeze(2).expand_as(min_)
    min_ = min_ * mask.float()
    loss = min_.sum() / (len(length)*1.0)
    return loss


def masked_cross_entropy_for_slot(logits, target, mask, use_softmax=True):
    # print("logits", logits)
    # print("target", target)
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # print(logits_flat.size())
    if use_softmax:
        log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    else:
        log_probs_flat = logits_flat  # torch.log(logits_flat)
    # print("log_probs_flat", log_probs_flat)
    target_flat = target.view(-1, 1)
    # print("target_flat", target_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())  # b * |s|
    losses = losses * mask.float()
    loss = losses.sum() / (losses.size(0)*losses.size(1))
    # print("loss inside", loss)
    return loss


def masked_cross_entropy_for_value(logits, target, mask):
    # logits: b * |s| * m * |v|
    # target: b * |s| * m
    # mask:   b * |s|
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # print(logits_flat.size())
    log_probs_flat = torch.log(logits_flat)
    # print("log_probs_flat", log_probs_flat)
    target_flat = target.view(-1, 1)
    # print("target_flat", target_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())  # b * |s| * m
    loss = masking(losses, mask)
    return loss


def masking(losses, mask):
    mask_ = []
    batch_size = mask.size(0)
    max_len = losses.size(2)
    for si in range(mask.size(1)):
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if mask[:, si].is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = mask[:, si].unsqueeze(
            1).expand_as(seq_range_expand)
        mask_.append((seq_range_expand < seq_length_expand))
    mask_ = torch.stack(mask_)
    mask_ = mask_.transpose(0, 1)
    if losses.is_cuda:
        mask_ = mask_.cuda()
    losses = losses * mask_.float()
    loss = losses.sum() / (mask_.sum().float())
    return loss


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS",
                           EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split():
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
                for v in value.split():
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def normalize_text(text):
    global replacements
    # lower case every word
    text = text.lower()
    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def prepare_data_seq(data_dir, training, task="dst", sequicity=0, batch_size=100):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    file_train = os.path.join(data_dir, 'train_dials.json')  # 'data/train_dials.json'
    file_dev = os.path.join(data_dir, 'dev_dials.json')  # 'data/dev_dials.json'
    file_test = os.path.join(data_dir, 'test_dials.json')  # 'data/test_dials.json'

    # Create saving folder
    if args['path']:
        folder_name = args['path'].rsplit('/', 2)[0] + '/'
    else:
        folder_name = os.path.join(data_dir, 'save/')
        #  folder_name = 'save/{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    ontology = json.load(open("data/multiwoz/db/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    if training:
        pair_train, train_max_len, slot_train = read_langs(file_train, gating_dict, ALL_SLOTS, "train", lang, mem_lang, sequicity, training)
        train = get_seq(pair_train, lang, mem_lang, batch_size, True, sequicity)
        nb_train_vocab = lang.n_words
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)
        if os.path.exists(folder_name+lang_name) and os.path.exists(folder_name+mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(folder_name+lang_name, 'rb') as handle:
                lang = pickle.load(handle)
            with open(folder_name+mem_lang_name, 'rb') as handle:
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(folder_name+lang_name, 'wb') as handle:
                pickle.dump(lang, handle)
            with open(folder_name+mem_lang_name, 'wb') as handle:
                pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

    test_4d = []
    if args['except_domain']!="":
        pair_test_4d, _, _ = read_langs(file_test, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        test_4d  = get_seq(pair_test_4d, lang, mem_lang, eval_batch, False, sequicity)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % nb_train_vocab )
    print("Vocab_size Belief %s" % mem_lang.n_words )
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
    print(SLOTS_LIST[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    print(SLOTS_LIST[3])
    LANG = [lang, mem_lang]
    return train, dev, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def read_langs(file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line=None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {}
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first
        for dial_dict in dials:
            if (args["all_vocab"] or dataset == "train") and training:
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')
        # determine training data ratio, default is 100%
        if training and dataset == "train" and args["data_ratio"] != 100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials) * 0.01 * args["data_ratio"])]

        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict[
                "domains"]) or \
                    (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict[
                        "domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history += (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict(
                            [(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]

                if (args["all_vocab"] or dataset == "train") and training:
                    mem_lang.index_words(turn_belief_dict, 'belief')

                class_label, generate_y, slot_mask, gating_label = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys():
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])

                data_detail = {
                    "ID": dial_dict["dialogue_idx"],
                    "domains": dial_dict["domains"],
                    "turn_domain": turn_domain,
                    "turn_id": turn_id,
                    "dialog_history": source_text,
                    "turn_belief": turn_belief_list,
                    "gating_label": gating_label,
                    "turn_uttr": turn_uttr_strip,
                    'generate_y': generate_y
                }
                data.append(data_detail)

                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())

            cnt_lin += 1
            if (max_line and cnt_lin >= max_line):
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len - 1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    print("domain_counter", domain_counter)
    return data, max_resp_len, slot_temp


def fix_general_label_error(labels, type, slots):
    label_dict = dict([(l[0], l[1]) for l in labels]) if type else dict(
        [(l["slots"][0][0], l["slots"][0][1]) for l in labels])

    GENERAL_TYPO = {
        # type
        "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house",
        "mutiple sports": "multiple sports",
        "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool",
        "concerthall": "concert hall",
        "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum",
        "ol": "architecture",
        "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum",
        "churches": "church",
        # area
        "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north",
        "cen": "centre", "east side": "east",
        "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre",
        "centre of cambridge": "centre",
        "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre",
        "in town": "centre", "north part of town": "north",
        "centre of town": "centre", "cb30aq": "none",
        # price
        "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
        # day
        "next friday": "friday", "monda": "monday",
        # parking
        "free parking": "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4", "4 stars": "4", "0 star rarting": "none",
        # others
        "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none",
        "not mentioned": "none",
        '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
    }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])

            # miss match slot and value
            if slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast",
                                                             "centre", "venetian", "intern", "a cheap -er hotel"] or \
                    slot == "hotel-internet" and label_dict[slot] == "4" or \
                    slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                    slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery",
                                                                       "science", "m"] or \
                    "area" in slot and label_dict[slot] in ["moderate"] or \
                    "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4",
                                                               "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no":
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we":
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent":
                    label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we":
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                    slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum",
                                                                       "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict


def get_seq(pairs, lang, mem_lang, batch_size, type, sequicity):
    if(type and args['fisher_sample']>0):
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, lang.word2index, lang.word2index, sequicity, mem_lang.word2index)

    if args["imbalance_sampler"] and type:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  # shuffle=type,
                                                  collate_fn=collate_fn,
                                                  sampler=ImbalancedDatasetSampler(dataset))
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=type,
                                                  collate_fn=collate_fn)
    return data_loader


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.sequicity = sequicity
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index]
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]

        item_info = {
            "ID": ID,
            "turn_id": turn_id,
            "turn_belief": turn_belief,
            "gating_label": gating_label,
            "context": context,
            "context_plain": context_plain,
            "turn_uttr_plain": turn_uttr,
            "turn_domain": turn_domain,
            "generate_y": generate_y,
        }
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book", "").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4, "hospital": 5, "bus": 6,
                   "police": 7}
        return domains[turn_domain]


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len - len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)  # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i, :end, :] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    return item_info


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)
