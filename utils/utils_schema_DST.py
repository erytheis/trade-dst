import os
from os.path import dirname, join
import json
from utils.utils_multiWOZ_DST import Lang
from utils.config import *

souce_dir = dirname(dirname(__file__))
dataset_dir = join(souce_dir, 'data/schema/')
dirnames = {"dev": join(dataset_dir, 'dev/'),
            "test": join(dataset_dir, 'test/'),
            "train": join(dataset_dir, "train/")}

# Get the directories of json files
file_paths = []
for phase, folder in dirnames.items():
    for _, __, filenames in os.walk(folder):
        file_paths.extend([join(folder, filename) for filename in filenames])
        break


def write_ontology_file():
    schema_files = []
    [schema_files.append(file) for file in file_paths if "schema.json" in file]
    domains = []
    for file in schema_files:
        with open(file) as f:
            dials = json.load(f)
            [domains.append(get_domain_name(service["service_name"])) for service in dials if
             service["service_name"] not in domains]
            domain_slot_values = {
                str(get_domain_name(service["service_name"].split("_")) + "-" + slot["name"]): slot["possible_values"]
                for service in dials for slot in service["slots"]}
    with open(join(dataset_dir, 'ontology.json'), 'w') as j:
        json.dump(domain_slot_values, j, indent=4)


def read_ontology_file(path_to_file):
    """
    :param path_to_file: full path to the ontology.json
    :return: ontology {}, all domains [], domain-slot pairs, {}
    """
    with open(path_to_file) as f:
        domains = []
        ontology = json.load(f)
        domain_slots = list(ontology.keys())
        domains = [pair.split("-")[0] for pair in domain_slots if pair.split("-")[0] not in domains]
    return ontology, domains, domain_slots


def process_dialogue(dialogue_dict, SLOTS):
    """
    Transform SCHEMA DSTC8 dialogue dictionary into a data dictionary, compatible with the Trade DST algorithm
    :param dialogue_dict: dictionary of a dialogue in DSTC8 format
    :param SLOTS: list of all possible slots
    :return: data dict
    """

    def combine_turns(user_turn, system_turn):
        assert user_turn["speaker"] == "USER" and system_turn["speaker"] == "SYSTEM"
        turn_domain = get_domain_name(user_turn["frames"][-1]["service"])
        turn_belief = ["-".join([get_domain_name(frame["service"]), slot, value]) for frame in user_turn["frames"] for
                       slot, values in
                       frame["state"]["slot_values"].items() for value in values]
        turn_utterance = system_turn["utterance"].lower() + " ; " + user_turn["utterance"].lower() + " "

        gating_label = [2] * len(SLOTS)
        generate_y = ['none'] * len(SLOTS)

        for elem in turn_belief:
            [domain, slot, value] = elem.split("-", 2)
            slot_idx = SLOTS.index("-".join([domain, slot]))
            if value == "dontcare":
                gating_label[slot_idx] = 1
                generate_y[slot_idx] = "dontcare"
            elif value == "No":
                gating_label[slot_idx] = 2
                generate_y[slot_idx] = "none"
            else:
                gating_label[slot_idx] = 0
                generate_y[slot_idx] = value

        return {'turn_domain': turn_domain,
                'turn_utterance': turn_utterance,
                'turn_belief': turn_belief,
                'generate_y': generate_y,
                'gating_label': gating_label}

    idx = dialogue_dict["dialogue_id"]
    domains = [get_domain_name(service) for service in dialogue_dict["services"]]
    initial_turn = [{"speaker": "SYSTEM",
                     "utterance": ""}]
    dialogue_dict["turns"] = initial_turn + dialogue_dict["turns"]
    dialogue_history = ""
    turn_id = 0
    data = []
    assert initial_turn[-1]["speaker"] == "SYSTEM"

    for system_turn, user_turn in pairwise(dialogue_dict["turns"]):
        new_dict = combine_turns(user_turn, system_turn)
        dialogue_history = dialogue_history + new_dict["turn_utterance"]
        new_dict["dialog_history"] = dialogue_history.strip()
        new_dict["turn_utterance"] = new_dict["turn_utterance"].strip()
        new_dict["ID"] = idx
        new_dict["turn_id"] = turn_id
        turn_id += 1
        new_dict["domains"] = domains
        data.append(new_dict)

    assert len(data) > 1
    dialog_length = len(data[-1]["dialogue_history"])

    return data, dialog_length


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    return zip(*[iter(iterable)] * 2)


def get_domain_name(service_string):
    """
    Processes the name of the service in the schema dialogue format
    :param service_string:
    :return: string of a domain, lowered
    """
    return service_string.split("_")[0].lower()


def process_all_dialogues(directory, SLOTS):
    """
    Processes all dialogues in the folder and combines them into one "data" dictionary
    :param directory: destination of the folder with .json dictinaries of dialogues
    :param SLOTS: all possible slots from the ontology
    :return: data dictionary, compatible with the one in the multiWOZ file
    """
    file_paths = []
    for _, __, filenames in os.walk(directory):
        file_paths.extend([join(directory, filename) for filename in filenames if filename != "schema.json"])
        break
    data = []
    max_resp_len = 0

    for file_path in file_paths:
        with open(file_path) as f:
            dialogue_dicts = json.load(f)
            for dialogue_dict in dialogue_dicts:
                temp_data, dialogue_length = process_dialogue(dialogue_dict, SLOTS)
                data.extend(temp_data)
                max_resp_len = dialogue_length if dialogue_length > max_resp_len else max_resp_len
    return data, max_resp_len


def prepare_data_seq(training, batch_size=100):
    # TODO Make it os path related
    ontology_dir = "/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/ontology.json"
    train_dialogues_dir = "/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/train"
    test_dialogues_dir = "/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/test"
    dev_dialogues_dir = "/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/dev"

    ontology, domains, ALL_SLOTS = read_ontology_file(ontology_dir)
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    if training:
        pair_train, train_max_len = process_all_dialogues(train_dialogues_dir)
        pair_test, test_max_len = process_all_dialogues(test_dialogues_dir)
        pair_dev, dev_max_len = process_all_dialogues(test_dialogues_dir)




SLOTS = read_ontology_file("/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/ontology.json")[2]
directory = "/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/train"
data = process_all_dialogues(directory, SLOTS)

# with open("/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/train/dialogues_123.json") as f:
#     dialogue_dict = json.load(f)[6]
#     data = process_dialogue(dialogue_dict, list(SLOTS[2]))

print()
