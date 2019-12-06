from create_data import createData, divideData
from utils.utils_multiWOZ_DST import prepare_data_seq_woz
from utils.config import *
from models.TRADE import *

args['decoder'] = "TRADE"
args['batch'] = 32
args['drop'] = 0.2
args['learn'] = 0.001
args['load_embedding'] = 1
args['parallel_decode'] = False

train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq_woz(True)


# Encoder Forward propagation
encoder = EncoderRNN(lang[0].n_words, int(args['hidden']), args['drop'])

it = iter(train)
data_batch = next(it)
story_size = data_batch['context'].size()
rand_mask = np.ones(story_size)
bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-0.2)[0]
rand_mask = rand_mask * bi_mask
rand_mask = torch.Tensor(rand_mask)
story = data_batch['context'] * rand_mask.long()

output, hidden_state = encoder.forward(story.transpose(0, 1), data_batch['context_len'])


# Decoder Forward propagation

batch_size = len(data_batch['context_len'])
copy_list = data_batch['context_plain']
max_res_len = data_batch['generate_y'].size(2)

decoder = Generator(lang[0], encoder.embedding, lang[0].n_words, int(args['hidden']), args['drop'], SLOTS_LIST[0], len(gating_dict))

all_point_outputs, all_gate_outputs, words_point_out, words_class_out = decoder.forward(batch_size,


                                                                                        hidden_state,
                                                                                        output,
                                                                                        data_batch['context_len'],
                                                                                        data_batch['context'],
                                                                                        max_res_len,
                                                                                        data_batch['generate_y'],
                                                                                        False,
                                                                                        SLOTS_LIST[1]
                                                                                        )


print()

