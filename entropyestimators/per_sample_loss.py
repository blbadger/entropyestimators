import torch
from einops import rearrange
import transformers
import shutil
import os
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from prettytable import PrettyTable
from dotenv import load_dotenv
from safetensors.torch import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bidirectional_model_trainer import ReverseTransformer, OutsideInMaskModel, BidirectionalTransformer

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')
device = 'cuda' if torch.cuda.is_available else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 512
n_hidden_layers = 16
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': n_hidden_layers,
	'num_attention_heads': 4,
	'vocab_size':  len(tokenizer)
}

configuration = LlamaConfig(**llama_config_kwargs)

# Initialize a reverse model tainer
model = LlamaForCausalLM(configuration)
reverse_model = ReverseTransformer(model).to(device)
forward_model = LlamaForCausalLM(configuration).to(device)

causal_model = LlamaForCausalLM(configuration)
outside_in_model = OutsideInMaskModel(causal_model, loss_direction='both', double_heads=False).to(device).to(torch.float16)

tokenized_length = 512
dim = 512
n_hidden_layers = 8
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': n_hidden_layers,
	'num_attention_heads': 4,
	'vocab_size':  len(tokenizer)
}

configuration = LlamaConfig(**llama_config_kwargs)

forward_m = LlamaModel(configuration)
reverse_m = LlamaModel(configuration)
bidirectional_model = BidirectionalTransformer(n_vocab, dim, forward_m, reverse_m, bidirectional_model=None, tokenized_length=tokenized_length).to(device)

load_model(reverse_model, '/home/bbadger/Desktop/reverse_model.safetensors')
load_model(forward_model, '/home/bbadger/Desktop/forward_model.safetensors')
load_model(outside_in_model, '/home/bbadger/Desktop/outside_in_model.safetensors')
load_model(bidirectional_model, '/home/bbadger/Desktop/bidirectional_model.safetensors')


def half_input(sample):
	half_length = len(sample['input_ids']) // 2
	sample['input_ids'] = sample['input_ids'][half_length:]
	sample['attention_mask'] = sample['attention_mask'][half_length:]
	return sample

test_path =  f"{data_root}/fineweb-edu-tokenized-test-c512-8k"
test_dataset =  load_from_disk(test_path).filter(lambda x: x['input_ids'][-1] != tokenizer.pad_token_id).take(300) # we don't want pad tokens 
# test_path =  f"{data_root}/fineweb-edu-tokenized-test-c1024"
# test_dataset = load_from_disk(test_path).map(half_input).filter(lambda x: x['input_ids'][-1] != tokenizer.pad_token_id).take(3000) # we don't want pad tokens 
cel = nn.CrossEntropyLoss(reduction='none')

def outside_in_loss(logits, labels):
	half_length = logits.shape[-1] // 2 # b e t
	right_shift_logits = logits[..., :-1]
	left_shift_logits = logits[..., 1:]
	right_shift_labels = labels[..., 1:]
	left_shift_labels = labels[..., :-1]
	forward_loss = cel(right_shift_logits[..., :half_length], right_shift_labels[..., :half_length])
	reverse_loss = cel(left_shift_logits[..., half_length:], left_shift_labels[..., half_length:])
	all_loss = torch.concat((forward_loss, reverse_loss), dim=-1)
	return all_loss

@torch.no_grad()
def middle_outputs(forward_model, reverse_model, input_ids, attention_mask, labels):
	half_length = len(input_ids[0]) // 2
	forward_attention_mask = attention_mask.clone()
	forward_attention_mask[:, :half_length] = torch.zeros((batch_size, half_length)).to(attention_mask.dtype).to(attention_mask.device)
	forward_outs = forward_model(input_ids, attention_mask=forward_attention_mask).logits
	_, reverse_outs = reverse_model(input_ids, attention_mask=attention_mask) # returns logits
	forward_outs = rearrange(forward_outs, 'b t e -> b e t')
	# outputs are pre-shifted
	reverse_outs = torch.flip(reverse_outs[..., half_length-1:-1], dims=[-1])
	forward_outs = forward_outs[..., half_length-1:-1]
	middle_outs = torch.cat((reverse_outs, forward_outs), dim=-1)
	return middle_outs

@torch.no_grad()
def get_outputs(forward_model, reverse_model, outside_in_model, bidirectional_model, attention_mask, input_ids):
	forward_outs = forward_model(input_ids, attention_mask=attention_mask).logits
	_, reverse_outs = reverse_model(input_ids, attention_mask=attention_mask) # returns logits
	_, outside_in_outs = outside_in_model(input_ids, attention_mask=attention_mask)
	_, bidirectional_outs = bidirectional_model(input_ids, attention_mask=attention_mask)
	return forward_outs, reverse_outs, outside_in_outs, bidirectional_outs

batch_size = 32
test_length = len(test_dataset)
index = 0
losses_per_sample = []
batches = test_length // batch_size

all_midout_losses = []
for i in tqdm(range(batches)):
	index = i * batch_size
	samples = test_dataset[index:index+batch_size]
	labels = torch.tensor(samples['input_ids'])
	labels = torch.where(labels==tokenizer.pad_token_id, -100, labels).to(device)

	attention_mask = torch.where(labels==tokenizer.pad_token_id, 0, 1).to(device)	
	input_ids = torch.tensor(samples['input_ids']).to(device)

	middle_outs = middle_outputs(forward_model, reverse_model, input_ids, attention_mask, labels)
	losses = cel(middle_outs, labels)
	all_midout_losses.append(losses)

print (sum(torch.mean(m) for m in all_midout_losses) / len(all_midout_losses))

all_midout_losses = torch.cat(all_midout_losses, dim=0).cpu()
df = pd.DataFrame()
indices = torch.arange(512).cpu().numpy()
full_repeat_indices = np.tile(indices, batches*batch_size) 
df['indices'] = full_repeat_indices
df['loss'] = all_midout_losses.flatten()
directions = ['midout']*len(all_midout_losses.flatten()) 
df['direction'] = directions
sns.lineplot(data=df, x='indices', y='loss', hue='direction')
plt.show()
plt.close()


all_min_losses = []
all_forward_losses = []
all_reverse_losses = []
outside_in_losses = []
all_oin_losses = []
all_bidirectional_losses = []

for i in tqdm(range(batches)):
	index = i * batch_size
	samples = test_dataset[index:index+batch_size]
	labels = torch.tensor(samples['input_ids'])
	labels = torch.where(labels==tokenizer.pad_token_id, -100, labels).to(device)
	reverse_labels = torch.flip(labels.clone(), dims=[1])

	attention_mask = torch.where(labels==tokenizer.pad_token_id, 0, 1).to(device)	
	input_ids = torch.tensor(samples['input_ids']).to(device)

	forward_outs, reverse_outs, outside_in_outs, bidirectional_outs = get_outputs(
		forward_model, 
		reverse_model, 
		outside_in_model, 
		bidirectional_model, 
		attention_mask, 
		input_ids
	)

	oin_losses = outside_in_loss(outside_in_outs, labels)

	forward_outs = rearrange(forward_outs, 'b t e -> b e t')
	forward_outs = forward_outs[..., :-1]

	reverse_outs = reverse_outs[..., :-1]
	shift_labels = labels[..., 1:]
	reverse_labels = reverse_labels[..., 1:]

	forward_losses = cel(forward_outs, shift_labels)
	reverse_losses = cel(reverse_outs, reverse_labels)

	all_forward_losses.append(forward_losses)
	all_reverse_losses.append(torch.flip(reverse_losses.clone(), dims=[1]))
	all_oin_losses.append(oin_losses.clone())

	bidirectional_outs = rearrange(bidirectional_outs, 'b t e -> b e t')
	bidirectional_losses = cel(bidirectional_outs, labels)
	all_bidirectional_losses.append(bidirectional_losses) # to match other shapes

	# compute average loss, ignoring pad
	n_nonpad_tokens = torch.tensor([torch.sum(label != -100) for label in shift_labels]).to(device)
	min_losses = torch.min(forward_losses, reverse_losses)
	forward_losses = torch.sum(forward_losses, dim=-1) / n_nonpad_tokens
	reverse_losses = torch.sum(reverse_losses, dim=-1) / n_nonpad_tokens
	all_min_losses.append(torch.sum(min_losses, dim=-1) / n_nonpad_tokens)
	losses_per_sample += list(zip(forward_losses, reverse_losses))

	outside_in_losses.append(torch.sum(oin_losses, dim=-1) / n_nonpad_tokens)

min_losses_per_sample = [min(pair) for pair in losses_per_sample]
average_forward_loss = sum(l[0] for l in losses_per_sample) / len(losses_per_sample)
average_reverse_loss = sum(l[1] for l in losses_per_sample) / len(losses_per_sample)
average_min_loss = sum(min_losses_per_sample) / len(min_losses_per_sample)
average_per_token_mins = sum(torch.mean(m) for m in all_min_losses) / len(all_min_losses)
average_outside_in_loss = sum(torch.nanmean(m) for m in outside_in_losses) / len(all_min_losses)

print (f'Average forward loss: {average_forward_loss}\
 \n Average Reverse loss: {average_reverse_loss},\
 \n Average Min Loss per Sample: {average_min_loss},\
 \n Average Min Loss per Element: {average_per_token_mins}\
 \n Average Outside In Losses: {average_outside_in_loss}')

all_forward_losses = torch.cat(all_forward_losses, dim=0).cpu()
all_reverse_losses = torch.cat(all_reverse_losses, dim=0).cpu()
all_oin_losses = torch.cat(all_oin_losses, dim=0).cpu()
all_bidirectional_losses = torch.cat(all_bidirectional_losses, dim=0).cpu()

indices = torch.arange(511).cpu().numpy()
forward_positionals = torch.mean(all_forward_losses, dim=0)
reverse_positionals = torch.mean(all_reverse_losses, dim=0)

repeat_indices = np.tile(indices, batches*batch_size) # repeat indices samples number of times
print (torch.min(all_forward_losses))


df = pd.DataFrame()
indices = torch.arange(512).cpu().numpy()
full_repeat_indices = np.tile(indices, batches*batch_size) 
df['indices'] = full_repeat_indices
df['loss'] = all_bidirectional_losses.flatten()
directions = ['bidirectional']*len(all_bidirectional_losses.flatten()) 
df['direction'] = directions
sns.lineplot(data=df, x='indices', y='loss', hue='direction')
plt.show()
plt.close()


df = pd.DataFrame()
df['indices'] = np.concatenate((repeat_indices, full_repeat_indices), axis=0)
df['loss'] = torch.cat((all_oin_losses.flatten(), all_bidirectional_losses.flatten()), dim=0)
directions = ['outside in']*len(all_forward_losses.flatten()) + ['bidirectional']*len(all_bidirectional_losses.flatten()) 
df['direction'] = directions
sns.lineplot(data=df, x='indices', y='loss', hue='direction')
plt.show()
plt.close()

# df = pd.DataFrame()
# df['indices'] = np.concatenate((repeat_indices, repeat_indices), axis=0)
# df['loss'] = torch.cat((all_forward_losses.flatten(), all_reverse_losses.flatten()), dim=0)
# directions = ['forward']*len(all_forward_losses.flatten()) + ['reverse']*len(all_forward_losses.flatten()) 
# df['direction'] = directions
# sns.lineplot(data=df, x='indices', y='loss', hue='direction')
# plt.show()
# plt.close()

# df = pd.DataFrame()
# df['indices'] = repeat_indices
# df['loss'] = torch.min(all_forward_losses.flatten(), all_reverse_losses.flatten())
# directions = ['minimum']*len(all_forward_losses.flatten()) 
# df['direction'] = directions
# sns.lineplot(data=df, x='indices', y='loss', hue='direction')
# plt.show()
# plt.close()

# plt.plot(indices, forward_positionals, label='forward')
# plt.plot(indices, reverse_positionals, label='reverse')
# plt.plot(indices, forward_positionals - reverse_positionals, label='differential')
# plt.legend()
# plt.savefig("plot.png", dpi=350, bbox_inches="tight")
# plt.close()



