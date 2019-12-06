import os
from os.path import dirname, join
import json
from utils.utils_dataset_processor import Lang, DialogProcessor
from utils.config import *


class SchemaProcessor(DialogProcessor):
    def __init__(self, training, sequicity):
        super().__init__(training, sequicity)
        self.path_train = 'data/schema/train'
        self.path_test = "data/schema/test"
        self.path_dev = "data/schema/dev"
        self.dataset = 'schema'

        ontology_dir = "data/schema/ontology.json"
        self.ontology, self.domains, self.ALL_SLOTS = read_ontology_file(ontology_dir)

        self.souce_dir = dirname(dirname(__file__))
        self.dataset_dir = join(self.souce_dir, 'data/schema/')
        self.dirnames = {"dev": join(self.dataset_dir, 'dev/'),
                        "test": join(self.dataset_dir, 'test/'),
                        "train": join(self.dataset_dir, "train/")}

        self.file_paths = []
        for phase, folder in self.dirnames.items():
            for _, __, filenames in os.walk(folder):
                self.file_paths.extend([join(folder, filename) for filename in filenames])
                break

    def process_dialogue(self, dialogue_dict, dataset, seen_slots):
        """
        Transform SCHEMA DSTC8 dialogue dictionary into a data dictionary, compatible with the Trade DST algorithm
        :param dialogue_dict: dictionary of a dialogue in DSTC8 format
        :param SLOTS: list of all possible slots
        :return: data dict
        """
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
            new_dict = self.combine_turns(user_turn, system_turn, dataset, seen_slots)
            dialogue_history = dialogue_history + new_dict["turn_uttr"]
            new_dict["dialog_history"] = dialogue_history.strip()
            new_dict["turn_uttr"] = new_dict["turn_uttr"].strip()
            new_dict["ID"] = idx
            new_dict["turn_id"] = turn_id
            turn_id += 1
            new_dict["domains"] = domains
            data.append(new_dict)

        assert len(data) > 1
        dialog_length = len(data[-1]["dialog_history"])
        return data, dialog_length


    def read_langs(self, directory, dataset, max_line=None):
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
        seen_slots = []

        # Process all dialog files
        for file_path in file_paths:
            with open(file_path) as f:
                dialogue_dicts = json.load(f)
                for dialogue_dict in dialogue_dicts:
                    temp_data, dialogue_length = self.process_dialogue(dialogue_dict, dataset, seen_slots)
                    data.extend(temp_data)
                    max_resp_len = dialogue_length if dialogue_length > max_resp_len else max_resp_len

        # Trimming repeating domain_slot_pairs
        for turn in data:
            gating_label = [2] * len(seen_slots)
            generate_y = ['none'] * len(seen_slots)
            for elem in turn["turn_belief"]:
                [domain, slot, value] = elem.split("-", 2)
                slot_idx = seen_slots.index("-".join([domain, slot]))

                if value == "dontcare":
                    gating_label[slot_idx] = 1
                    generate_y[slot_idx] = "dontcare"
                elif value == "No":
                    gating_label[slot_idx] = 2
                    generate_y[slot_idx] = "none"
                else:
                    gating_label[slot_idx] = 0
                    generate_y[slot_idx] = value

            turn["generate_y"] = generate_y
            turn["gating_label"] = gating_label

        return data, max_resp_len, seen_slots

    def combine_turns(self, user_turn, system_turn, dataset, seen_slots):
        assert user_turn["speaker"] == "USER" and system_turn["speaker"] == "SYSTEM"
        turn_domain = get_domain_name(user_turn["frames"][-1]["service"])
        turn_belief = ["-".join([get_domain_name(frame["service"]), slot, value]) for frame in
                       user_turn["frames"] for
                       slot, values in
                       frame["state"]["slot_values"].items() for value in values]
        turn_utterance = system_turn["utterance"].lower() + " ; " + user_turn["utterance"].lower() + " "
        if (dataset == "train" or args["all_vocab"]) and self.training:
            self.lang.index_words(turn_utterance, 'utter')

        # Update seen slots
        for elem in turn_belief:
            [domain, slot, value] = elem.split("-", 2)
            domain_slot_pair = "-".join([domain, slot])
            if domain_slot_pair not in seen_slots:
                seen_slots.append(domain_slot_pair)
            if (dataset == "train" or args["all_vocab"]) and self.training:
                self.mem_lang.index_words(value, 'value')

        return {'turn_domain': turn_domain,
                'turn_uttr': turn_utterance,
                'turn_belief': turn_belief}


    def write_ontology_file(self):
        schema_files = []
        [schema_files.append(file) for file in self.file_paths]
        domains = []
        for file in schema_files:
            with open(file) as f:
                dials = json.load(f)
                [domains.append(get_domain_name(service["service_name"])) for service in dials if
                 service["service_name"] not in domains]
                domain_slot_values = {
                    str(get_domain_name(service["service_name"].split("_")) + "-" + slot["name"]): slot["possible_values"]
                    for service in dials for slot in service["slots"]}
        with open(join(self.dataset_dir, 'ontology.json'), 'w') as j:
            json.dump(domain_slot_values, j, indent=4)


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


