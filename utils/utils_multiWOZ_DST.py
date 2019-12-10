import json
import random
from collections import OrderedDict

from utils.config import *
import os
import pickle

from utils.config import args
from utils.fix_label import fix_general_label_error

from utils.utils_dataset_processor import dump_pretrained_emb, DialogProcessor

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


class WozProcessor(DialogProcessor):
    def __init__(self, training, sequicity):
        super().__init__(training, sequicity)
        self.path_train = 'data/multi-woz/train_dials.json'
        self.path_dev = 'data/multi-woz/dev_dials.json'
        self.path_test = 'data/multi-woz/test_dials.json'
        self.dataset = 'multi-woz'
        self.domains = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4, "hospital": 5, "bus": 6,
                        "police": 7}

        ontology = json.load(open("data/multi-woz/MULTIWOZ2.1/ontology.json", 'r'))
        self.ALL_SLOTS = get_slot_information(ontology)

    def prepare_data_seq_woz(self, task="dst", sequicity=0, batch_size=100):
        eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size

        folder_name = self.create_folder()
        # load domain-slot pairs from ontology

        # Vocabulary
        self.lang.index_words(self.ALL_SLOTS, 'slot')
        self.mem_lang.index_words(self.ALL_SLOTS, 'slot')
        self.lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
        self.mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

        if self.training:
            pair_train, train_max_len, slot_train = self.read_langs(self.path_train, "train")
            train = self.get_seq(pair_train, batch_size, True)
            nb_train_vocab = self.lang.n_words
            pair_dev, dev_max_len, slot_dev = self.read_langs(self.path_dev, "dev")
            dev = self.get_seq(pair_dev, eval_batch, False)
            pair_test, test_max_len, slot_test = self.read_langs(self.path_test, "test")
            test = self.get_seq(pair_test, eval_batch, False)

            if os.path.exists(folder_name + self.lang_name) and os.path.exists(folder_name + self.mem_lang_name):
                print("[Info] Loading saved lang files...")
                self.open_saved_lang()
            else:
                print("[Info] Dumping lang files...")
                self.write_saved_lang()

            emb_dump_path = 'data/emb{}.json'.format(len(self.lang.index2word))
            if not os.path.exists(emb_dump_path) and args["load_embedding"]:
                dump_pretrained_emb(self.lang.word2index, self.lang.index2word, emb_dump_path)
        else:
            with open(folder_name + self.lang_name, 'rb') as handle:
                lang = pickle.load(handle)
            with open(folder_name + self.mem_lang_name, 'rb') as handle:
                mem_lang = pickle.load(handle)

            pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
            pair_dev, dev_max_len, slot_dev = self.read_langs(self.path_dev, "dev")
            dev = self.get_seq(pair_dev, eval_batch, False)
            pair_test, test_max_len, slot_test = self.read_langs(self.path_test, "test")
            test = self.get_seq(pair_test, eval_batch, False)

        test_4d = []
        if args['except_domain'] != "":
            pair_test_4d, _, _ = self.read_langs(self.path_test, "dev")
            test_4d = self.get_seq(pair_test_4d, eval_batch, False)

        max_word = max(train_max_len, dev_max_len, test_max_len) + 1

        print("Read %s pairs train" % len(pair_train))
        print("Read %s pairs dev" % len(pair_dev))
        print("Read %s pairs test" % len(pair_test))
        print("Vocab_size: %s " % lang.n_words)
        print("Vocab_size Training %s" % nb_train_vocab)
        print("Vocab_size Belief %s" % mem_lang.n_words)
        print("Max. length of dialog words for RNN: %s " % max_word)
        print("USE_CUDA={}".format(USE_CUDA))

        SLOTS_LIST = [self.ALL_SLOTS, slot_train, slot_dev, slot_test]
        print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
        print(SLOTS_LIST[2])
        print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
        print(SLOTS_LIST[3])
        LANG = [lang, mem_lang]
        return train, dev, test, test_4d, LANG, SLOTS_LIST, self.gating_dict, nb_train_vocab

    def read_langs(self, file_name, dataset, max_line=None):
        print(("Reading from {}".format(file_name)))
        data = []
        max_resp_len, max_value_len = 0, 0
        domain_counter = {}
        with open(file_name) as f:
            dials = json.load(f)
            # create vocab first
            for dial_dict in dials:
                if (args["all_vocab"] or dataset == "train") and self.training:
                    for ti, turn in enumerate(dial_dict["dialogue"]):
                        self.lang.index_words(turn["system_transcript"], 'utter')
                        self.lang.index_words(turn["transcript"], 'utter')
            # determine training data ratio, default is 100%
            if self.training and dataset == "train" and args["data_ratio"] != 100:
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
                    turn_belief_dict = fix_general_label_error(turn["belief_state"], False, self.ALL_SLOTS)

                    # Generate domain-dependent slot list
                    slot_temp = self.ALL_SLOTS
                    if dataset == "train" or dataset == "dev":
                        if args["except_domain"] != "":
                            slot_temp = [k for k in self.ALL_SLOTS if args["except_domain"] not in k]
                            turn_belief_dict = OrderedDict(
                                [(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                        elif args["only_domain"] != "":
                            slot_temp = [k for k in self.ALL_SLOTS if args["only_domain"] in k]
                            turn_belief_dict = OrderedDict(
                                [(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                    else:
                        if args["except_domain"] != "":
                            slot_temp = [k for k in self.ALL_SLOTS if args["except_domain"] in k]
                            turn_belief_dict = OrderedDict(
                                [(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                        elif args["only_domain"] != "":
                            slot_temp = [k for k in self.ALL_SLOTS if args["only_domain"] in k]
                            turn_belief_dict = OrderedDict(
                                [(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                    turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]

                    if (args["all_vocab"] or dataset == "train") and self.training:
                        self.mem_lang.index_words(turn_belief_dict, 'belief')

                    class_label, generate_y, slot_mask, gating_label = [], [], [], []
                    start_ptr_label, end_ptr_label = [], []
                    for slot in slot_temp:
                        if slot in turn_belief_dict.keys():
                            generate_y.append(turn_belief_dict[slot])

                            if turn_belief_dict[slot] == "dontcare":
                                gating_label.append(self.gating_dict["dontcare"])
                            elif turn_belief_dict[slot] == "none":
                                gating_label.append(self.gating_dict["none"])
                            else:
                                gating_label.append(self.gating_dict["ptr"])

                            if max_value_len < len(turn_belief_dict[slot]):
                                max_value_len = len(turn_belief_dict[slot])

                        else:
                            generate_y.append("none")
                            gating_label.append(self.gating_dict["none"])

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
        if "t{}".format(max_value_len - 1) not in self.mem_lang.word2index.keys() and self.training:
            for time_i in range(max_value_len):
                self.mem_lang.index_words("t{}".format(time_i), 'utter')

        print("domain_counter", domain_counter)
        return data, max_resp_len, slot_temp


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS
