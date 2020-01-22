import sys, os
from utils.config import *
from models.TRADE import *
from utils.utils_dataset_processor import *


def preprocess_utterance(utterance, word2idx):
    """Converts words to ids."""
    story = [int(word2idx[word]) if word in word2idx else UNK_token for word in utterance.split()]
    story = torch.tensor(story, dtype=torch.int64)
    return story.unsqueeze(1)

def postprocess_output():
    pass

def boot_model(path):
    args['path'] = path
    directory = args['path'].split("/")
    max_res_len = 10
    HDD = directory[-2].split('HDD')[1].split('BSZ')[0]
    decoder = directory[-3].split('-')[0]
    args["decoder"] = decoder
    args["HDD"] = HDD
    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(False, False)
    model = globals()[decoder](
        int(HDD),
        lang=lang,
        path=args['path'],
        task=args["task"],
        lr=0,
        dropout=0,
        slots=SLOTS_LIST,
        gating_dict=gating_dict,
        nb_train_vocab=lang[0].n_words)
    return model, lang, SLOTS_LIST

def predict(model, story, *resources):
    """
    :param model:
    :param story:
    :param args: [0] lang object
                 [1] SLOTS_LIST
    :return:
    """
    lang = resources[0]
    slots = resources[1][0]
    max_res_len = 10
    story = preprocess_utterance(story, lang[0].word2index)
    context_len = [story.size(0)]
    BSZ = story.size(1)
    model.encoder.train(False)

    encoded_outputs, encoded_hidden = model.encoder(story, context_len)
    all_point_outputs, all_gate_outputs, words_point_out, words_class_out = model.decoder.forward(BSZ,\
                                                                                                      encoded_hidden,
                                                                                                      encoded_outputs,
                                                                                                      context_len,
                                                                                                      story.transpose(0, 1),
                                                                                                      max_res_len,
                                                                                                      None,
                                                                                                      False,
                                                                                                      slots)

    # transorm output
    inverse_unpoint_slot = dict([(v, k) for k, v in model.gating_dict.items()])
    for bi in range(BSZ):
        predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
        gate = torch.argmax(all_gate_outputs.transpose(0, 1)[bi], dim=1)

        # pointer-generator results
        if args["use_gate"]:
            for si, sg in enumerate(gate):
                if sg == model.gating_dict["none"]:
                    continue
                elif sg == model.gating_dict["ptr"]:
                    pred = np.transpose(words_point_out[si])[bi]
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
                        predict_belief_bsz_ptr.append(slots[si] + "-" + str(st))
                else:
                    predict_belief_bsz_ptr.append(slots[si] + "-" + inverse_unpoint_slot[sg.item()])
        else:
            for si, _ in enumerate(gate):
                pred = np.transpose(words_point_out[si])[bi]
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
                    predict_belief_bsz_ptr.append(slots[si] + "-" + str(st))


    return predict_belief_bsz_ptr


def boot_and_predict(story):
    args['path'] = 'save/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC/'
    directory = args['path'].split("/")
    max_res_len = 10
    HDD = directory[2].split('HDD')[1].split('BSZ')[0]
    decoder = directory[1].split('-')[0]
    BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
    args["decoder"] = decoder
    args["HDD"] = HDD
    print("HDD", HDD, "decoder", decoder, "BSZ", BSZ)
    if args['dataset'] != 'multiwoz' or args['dataset'] != 'schema':
        print("You need to provide the --dataset information")
        sys.exit()
    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(False, False)

    model = globals()[decoder](
        int(HDD),
        lang=lang,
        path=args['path'],
        task=args["task"],
        lr=0,
        dropout=0,
        slots=SLOTS_LIST,
        gating_dict=gating_dict,
        nb_train_vocab=lang[0].n_words)
    story = preprocess_utterance(story, lang[0].word2index)
    context_len = [story.size(0)]
    model.encoder.train(False)
    encoded_outputs, encoded_hidden = model.encoder(story, context_len)
    all_point_outputs, all_gate_outputs, words_point_out, words_class_out = model.decoder.forward(BSZ, \
                                                                                                  encoded_hidden,
                                                                                                  encoded_outputs,
                                                                                                  context_len,
                                                                                                  story.transpose(0, 1),
                                                                                                  max_res_len,
                                                                                                  None,
                                                                                                  False,
                                                                                                  SLOTS_LIST[0])



if __name__ == '__main__':
    pass
