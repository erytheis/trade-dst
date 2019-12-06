import json
from random import shuffle

import pickle
import os
import torch
from embeddings import GloveEmbedding, KazumaCharEmbedding
from torch.utils import data as data
from tqdm import tqdm

from utils.config import PAD_token, SOS_token, EOS_token, UNK_token, USE_CUDA, args

def prepare_data_seq(training, sequicity):
    if args["dataset"] == "multiwoz":
        from utils.utils_multiWOZ_DST import WozProcessor
        dialog_processor = WozProcessor(training, sequicity)
    elif args["dataset"] == "schema":
        from utils.utils_schema_DST import SchemaProcessor
        dialog_processor = SchemaProcessor(training, sequicity)
    else:
        raise NotImplementedError

    return dialog_processor.prepare_data_seq(training, sequicity)


class DialogProcessor:
    def __init__(self, training, sequicity):
        self.training = training
        self.sequicity = sequicity
        self.dataset = ""

        self.gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
        self.ALL_SLOTS = []
        self.dev_slots = []
        self.train_slots = []
        self.test_slots = []

        self.path_train = ""
        self.path_dev = ""
        self.path_test = ""

        self.lang, self.mem_lang = Lang(), Lang()

    def get_seq(self, pairs, batch_size, type):
        if (type and args['fisher_sample'] > 0):
            shuffle(pairs)
            pairs = pairs[:args['fisher_sample']]

        data_info = {}
        data_keys = pairs[0].keys()
        for k in data_keys:
            data_info[k] = []

        for pair in pairs:
            for k in data_keys:
                data_info[k].append(pair[k])

        dataset = Dataset(data_info, self.lang.word2index, self.lang.word2index, self.sequicity,
                          self.mem_lang.word2index)

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

    def read_langs(self, file_name, dataset, max_line=None):
        raise NotImplementedError

    def prepare_data_seq(self, task="dst", sequicity=0, batch_size=args["batch"]):

        eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
        self.create_folder()

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
            pair_test, test_max_len, slot_test = self.read_langs(self.path_test, "test")

            dev = self.get_seq(pair_dev, eval_batch, False)
            test = self.get_seq(pair_test, eval_batch, False)

            if os.path.exists(self.folder_name + self.lang_name) and os.path.exists(self.folder_name + self.mem_lang_name):
                print("[Info] Loading saved lang files...")
                self.open_saved_lang()
            else:
                print("[Info] Dumping lang files...")
                self.write_saved_lang()

            emb_dump_path = 'data/' + self.dataset + '/emb{}.json'.format(len(self.lang.index2word))
            if not os.path.exists(emb_dump_path) and args["load_embedding"]:
                dump_pretrained_emb(self.lang.word2index, self.lang.index2word, emb_dump_path)
        else:
            pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
            pair_dev, dev_max_len, slot_dev = self.read_langs(self.path_dev, "dev")
            pair_test, test_max_len, slot_test = self.read_langs(self.path_test, "test")

            dev = self.get_seq(pair_dev, eval_batch, False)
            test = self.get_seq(pair_test, eval_batch, False)

        test_4d = []
        if args['except_domain'] != "":
            pair_test_4d, _, _ = self.read_langs(self.path_test, "dev")
            test_4d = self.get_seq(pair_test_4d, eval_batch, False)

        max_word = max(train_max_len, dev_max_len, test_max_len) + 1

        print("Read %s pairs train" % len(pair_train))
        print("Read %s pairs dev" % len(pair_dev))
        print("Read %s pairs test" % len(pair_test))
        print("Vocab_size: %s " % self.lang.n_words)
        print("Vocab_size Training %s" % nb_train_vocab)
        print("Vocab_size Belief %s" % self.mem_lang.n_words)
        print("Max. length of dialog words for RNN: %s " % max_word)
        print("USE_CUDA={}".format(USE_CUDA))

        SLOTS_LIST = [self.ALL_SLOTS, slot_train, slot_dev, slot_test]
        print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
        print(SLOTS_LIST[2])
        print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
        print(SLOTS_LIST[3])
        LANG = [self.lang, self.mem_lang]

        return train, dev, test, test_4d, LANG, SLOTS_LIST, self.gating_dict, nb_train_vocab

    def open_saved_lang(self):
        with open(self.folder_name + self.lang_name, 'rb') as handle:
            self.lang = pickle.load(handle)
        with open(self.folder_name + self.mem_lang_name, 'rb') as handle:
            self.mem_lang = pickle.load(handle)

    def write_saved_lang(self):
        with open(self.folder_name + self.lang_name, 'wb') as handle:
            pickle.dump(self.lang, handle)
        with open(self.folder_name + self.mem_lang_name, 'wb') as handle:
            pickle.dump(self.mem_lang, handle)

    def create_folder(self):
        # Create saving folder
        if args['path']:
            self.folder_name = args['path'].rsplit('/', 2)[0] + '/'
        else:
            self.folder_name = 'save/{}-'.format(args["decoder"]) + args["addName"] + args['dataset'] + str(
                args['task']) + '/'
        print("folder_name", self.folder_name)
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)
        elif type == 'value':
            for v in sent.split(" "):
                for vv in v.split("_"):
                    self.index_word(vv)



    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


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