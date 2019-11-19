import os
from os.path import dirname, join
import json

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
            [domains.append(service["service_name"].split("_")[0].lower()) for service in dials if
             service["service_name"] not in domains]
            domain_slot_values = {
                str(service["service_name"].split("_")[0].lower() + "-" + slot["name"]): slot["possible_values"]
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
        assert len(user_turn["frames"]) == 1
        user_state = user_turn["frames"][0]["state"]
        turn_domain = user_turn["frames"][0]["service"].split("_")[0].lower()
        turn_belief = ["-".join([turn_domain, slot, value]) for slot, values in user_state["slot_values"].items() for
                       value in values]
        turn_utterance = system_turn["utterance"] + " ; " + user_turn["utterance"]

        slots_to_idx = {slot: SLOTS.index("-".join([turn_domain, slot])) for slot in user_state["slot_values"].keys()}

        gating_label = [2] * len(SLOTS)
        generate_y = ['none'] * len(SLOTS)

        for slot, idx in slots_to_idx.items():
            if user_state["slot_values"][slot] == "dontcare":
                gating_label[idx] = 1
                generate_y[idx] = "dontcare"
            elif user_state["slot_values"][slot] == "No":
                gating_label[idx] = 2
                generate_y[idx] = "none"
            else:
                gating_label[idx] = 0
                generate_y[idx] = user_state["slot_values"][slot]

        return {'turn_domain': turn_domain,
                'turn_utterance': turn_utterance,
                'turn_belief': turn_belief,
                'generate_y': generate_y,
                'gating_label': gating_label}

    idx = dialogue_dict["dialogue_id"]
    domains = [service.split("_")[0].lower() for service in dialogue_dict["services"]]
    initial_turn = [{"speaker": "SYSTEM",
                     "utterance": ""}]
    dialogue_dict["turns"] = initial_turn + dialogue_dict["turns"]
    dialogue_history = " ; "
    turn_id = 0
    data = []
    assert initial_turn[-1]["speaker"] == "SYSTEM"

    for system_turn, user_turn in pairwise(dialogue_dict["turns"]):
        print(turn_id)
        new_dict = combine_turns(user_turn, system_turn)
        dialogue_history = dialogue_history + new_dict["turn_utterance"]
        new_dict["dialog_history"] = dialogue_history + new_dict["turn_utterance"]
        new_dict["ID"] = idx
        new_dict["turn_id"] = turn_id
        turn_id += 1
        new_dict["domains"] = domains
        data.append(new_dict)

    return data


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    return zip(*[iter(iterable)] * 2)


SLOTS = read_ontology_file("/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/ontology.json")
with open("/Users/khasbulat.kerimov/Projects/trade-dst/data/schema/train/dialogues_123.json") as f:
    dialogue_dict = json.load(f)[0]
    data = process_dialogue(dialogue_dict, list(SLOTS[2]))
    print()
