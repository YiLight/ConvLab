# Modified by Microsoft Corporation.
# Licensed under the MIT license.
from tqdm import tqdm

from convlab.modules.word_dst.multiwoz.trade.trade_utils import *
from convlab.modules.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from convlab.modules.dst.state_tracker import Tracker
from convlab.modules.dst.multiwoz.dst_util import init_state, normalize_value
import copy
import json
import os
import torch
import pickle
import urllib
import tarfile
from pprint import pprint

import numpy as np

train_batch_size = 1
batches_per_eval = 10
no_epochs = 600
device = "cuda"
start_batch = 0
MAX_RES_LEN = 10


HDD = 400
decoder = 'TRADE'
BSZ = 32
MAX_QUERY_LENGTH = 200

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # renamed_module = module
        if module == "utils.utils_multiWOZ_DST":
            renamed_module = "convlab.modules.word_dst.multiwoz.trade.trade_utils"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


class TRADETracker(Tracker):
    """
    Transferable multi-domain dialogue state tracker, adopted from https://github.com/jasonwu0731/trade-dst.
    """

    def __init__(self, data_dir='data/trade', model_file=''):
        Tracker.__init__(self)
        self.data_dir = data_dir
        self.lang_url = os.path.join(data_dir, 'lang-all.pkl')
        self.model_url = os.path.join(data_dir, 'HDD400BSZ32DR0.2ACC-0.4900')
        self.mem_lang_url = os.path.join(data_dir, 'mem-lang-all.pkl')
        self.gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
        self.slot_list_url = os.path.join(data_dir, 'slot-list.pkl')

        if not os.path.exists(data_dir):
            if model_file == '':
                raise Exception(
                    'Please provide remote model file path in config')
            resp = urllib.request.urlretrieve(model_file)[0]
            temp_file = tarfile.open(resp)
            temp_file.extractall('data')
            assert os.path.exists(data_dir)

        self.lang = renamed_load(open(self.lang_url, 'rb'))
        self.mem_lang = renamed_load(open(self.mem_lang_url, 'rb'))
        self.SLOT_LIST = pickle.load(open(self.slot_list_url, 'rb'))
        self.LANG = [self.lang, self.mem_lang]

        self.trade_model = TRADE(HDD, lang=self.LANG, path=data_dir, task='dst', lr=0, dropout=0, slots=self.SLOT_LIST,
                                 gating_dict=self.gating_dict, nb_train_vocab=self.lang.n_words)

        self.state = {}
        self.param_restored = False

        self.det_dic = {}
        for domain, dic in REF_USR_DA.items():
            for key, value in dic.items():
                assert '-' not in key
                self.det_dic[key.lower()] = key + '-' + domain
                self.det_dic[value.lower()] = key + '-' + domain
        self.value_dict = json.load(
            open(os.path.join(data_dir, 'value_dict.json')))

        self.cached_res = {}

    def init_session(self):
        self.state = init_state()
        if not self.param_restored:
            self.load_param()
            self.param_restored = True

    def load_param(self):
        if USE_CUDA:
            self.trade_model.encoder.load_state_dict(
                torch.load(os.path.join(self.model_url, 'enc.th')))
            self.trade_model.decoder.load_state_dict(
                torch.load(os.path.join(self.model_url, 'dec.th')))
        else:
            self.trade_model.encoder.load_state_dict(
                torch.load(os.path.join(self.model_url, 'enc.th'), lambda storage, loc: storage))
            self.trade_model.decoder.load_state_dict(
                torch.load(os.path.join(self.model_url, 'dec.th'), lambda storage, loc: storage))

    def update(self, user_act=None):
        """Update the dialogue state with the generated tokens from TRADE"""
        if not isinstance(user_act, str):
            raise Exception(
                f'Expected user_act is str but found {type(user_act)}')
        prev_state = self.state

        actual_history = copy.deepcopy(prev_state['history'])
        actual_history[-1].append(user_act)  # [sys, user], [sys,user]

        query = self.construct_query(actual_history)
        pred_states = self.predict(query)

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        for state in pred_states:
            domain, slot, value = state.split('-')
            if slot not in ['name', 'book']:
                if domain not in new_belief_state:
                    raise Exception(
                        'Error: domain <{}> not in belief state'.format(domain))
            slot = REF_SYS_DA[domain.capitalize()].get(slot, slot)
            assert 'semi' in new_belief_state[domain]
            assert 'book' in new_belief_state[domain]
            if 'book' in slot:
                assert slot.startswith('book ')
                slot = slot.strip().split()[1]
            if slot == 'arriveby':
                slot = 'arriveBy'
            elif slot == 'leaveat':
                slot = 'leaveAt'
            domain_dic = new_belief_state[domain]
            if slot in domain_dic['semi']:
                new_belief_state[domain]['semi'][slot] = normalize_value(
                    self.value_dict, domain, slot, value)
            elif slot in domain_dic['book']:
                new_belief_state[domain]['book'][slot] = value
            elif slot.lower() in domain_dic['book']:
                new_belief_state[domain]['book'][slot.lower()] = value
            else:
                with open('trade_tracker_unknown_slot.log', 'a+') as f:
                    f.write(
                        f'unknown slot name <{slot}> with value <{value}> of domain <{domain}>\nitem: {state}\n\n')

        new_request_state = copy.deepcopy(prev_state['request_state'])
        # update request_state
        user_request_slot = self.detect_requestable_slots(user_act)
        for domain in user_request_slot:
            for key in user_request_slot[domain]:
                if domain not in new_request_state:
                    new_request_state[domain] = {}
                if key not in new_request_state[domain]:
                    new_request_state[domain][key] = user_request_slot[domain][key]

        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state
        self.state = new_state
        # print((pred_states, query))
        return self.state

    def predict(self, query):
        if query in self.cached_res.keys():
            return self.cached_res[query]

        self.trade_model.encoder.train(False)
        self.trade_model.decoder.train(False)
        story = [self.lang.word2index[i] if i in self.lang.word2index.keys() else self.lang.word2index['UNK'] for i in
                 query.split()]
        story = torch.tensor([story]).cuda(
        ) if USE_CUDA else torch.tensor([story])

        story_len = story.size(1)
        encoded_outputs, encoded_hidden = self.trade_model.encoder(
            story.transpose(0, 1), torch.tensor([story_len]))

        batch_size = 1
        context_len = [story_len]  # story length
        generated_y = [5, 5, 5, 5, 5]  # targeted y, can be random
        max_res_len = MAX_RES_LEN
        use_teacher_forcing = False
        _, gates, words, class_words = \
            self.trade_model.decoder.forward(batch_size, encoded_hidden,
                                             encoded_outputs, context_len, story, max_res_len, generated_y,
                                             use_teacher_forcing, self.SLOT_LIST[2])
        predict_belief_bsz_ptr = []
        bi = 0
        gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)
        inverse_unpoint_slot = dict([(v, k)
                                     for k, v in self.gating_dict.items()])
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
                        self.SLOT_LIST[2][si] + "-" + str(st))
            else:
                predict_belief_bsz_ptr.append(
                    self.SLOT_LIST[2][si] + "-" + inverse_unpoint_slot[sg.item()])
        # predicted belief state, using ptr
        self.cached_res[query] = predict_belief_bsz_ptr
        return predict_belief_bsz_ptr

    def train(self):
        train, dev, test, test_special, lang, slots_list, gating_dict, max_word = prepare_data_seq(self.data_dir,
                                                                                                   True,
                                                                                                   args['task'],
                                                                                                   False,
                                                                                                   batch_size=int(
                                                                                                       args['batch']))

        model = TRADE(
            hidden_size=int(args['hidden']),
            lang=lang,
            path=args['path'],
            task=args['task'],
            lr=float(args['learn']),
            dropout=float(args['drop']),
            slots=slots_list,
            gating_dict=gating_dict,
            nb_train_vocab=max_word)

        avg_best, cnt, acc = 0.0, 0, 0.0
        early_stop = None

        for epoch in range(200):
            print("Epoch:{}".format(epoch))
            pbar = tqdm(enumerate(train), total=len(train))
            for i, training_data in pbar:
                model.train_batch(training_data, int(args['clip']), slots_list[1], reset=(i == 0))
                model.optimize(args['clip'])

            if (epoch + 1) % int(args['evalp']) == 0:

                acc = model.evaluate(dev, avg_best, slots_list[2], early_stop)
                model.scheduler.step(acc)

                if acc >= avg_best:
                    avg_best = acc
                    cnt = 0
                    best_model = model
                else:
                    cnt += 1

                if cnt == args["patience"] or (acc == 1.0 and early_stop is None):
                    print("Ran out of patient, early stop...")
                    break

    def test(self):
        directory = args['path'].split("/")
        test_HDD = directory[2].split('HDD')[1].split('BSZ')[0]
        test_decoder = directory[1].split('-')[0]
        test_BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
        args["decoder"] = test_decoder
        args["HDD"] = test_HDD
        print("HDD", test_HDD, "decoder", test_decoder, "BSZ", test_BSZ)

        train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(False, args['task'],
                                                                                                   False,
                                                                                                   batch_size=test_BSZ)

        model = test_decoder(
            int(test_HDD),
            lang=lang,
            path=args['path'],
            task=args["task"],
            lr=0,
            dropout=0,
            slots=SLOTS_LIST,
            gating_dict=gating_dict,
            nb_train_vocab=max_word)

        if args["run_dev_testing"]:
            print("Development Set ...")
            acc_dev = model.evaluate(dev, 1e7, SLOTS_LIST[2])

        if args['except_domain'] != "" and args["run_except_4d"]:
            print("Test Set on 4 domains...")
            acc_test_4d = model.evaluate(test_special, 1e7, SLOTS_LIST[2])

        print("Test Set ...")
        acc_test = model.evaluate(test, 1e7, SLOTS_LIST[3])

    def construct_query(self, context):
        '''Construct query from context'''
        query = ''
        for sys_ut, user_ut in context:
            if sys_ut.lower() == 'null':
                query += ' ; ' + normalize_text(user_ut)
            else:
                query += ' ; ' + \
                    normalize_text(sys_ut) + ' ; ' + normalize_text(user_ut)
        len_ = len(query)
        query = query[-1 * min(len_, MAX_QUERY_LENGTH):]
        # print(query)
        return query

    def detect_requestable_slots(self, observation):
        result = {}
        observation = observation.lower()
        _observation = ' {} '.format(observation)
        for value in self.det_dic.keys():
            _value = ' {} '.format(value.strip())
            if _value in _observation:
                key, domain = self.det_dic[value].split('-')
                if domain not in result:
                    result[domain] = {}
                result[domain][key] = 0
        return result    


def test_update():
    # lower case, tokenized.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    trade_tracker = TRADETracker()
    trade_tracker.init_session()
    trade_tracker.state['history'] = [
        ['null', 'i am trying to find an restaurant in the center'],
        ['the cambridge chop is an good restaurant']
    ]
    from timeit import default_timer as timer
    start = timer()
    pprint(trade_tracker.update('what is the area ?'))
    end = timer()
    print(end - start)

    start = timer()
    pprint(trade_tracker.update('what is the area '))
    end = timer()
    print(end - start)


if __name__ == '__main__':
    print("start/n")
    test_update()
    print("end")
