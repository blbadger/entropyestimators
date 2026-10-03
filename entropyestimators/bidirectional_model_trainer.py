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


class ReverseTransformer(nn.Module):

	def __init__(self, n_vocab, dim, model):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.lm_head = nn.Linear(dim, n_vocab)
		self.cel = nn.CrossEntropyLoss()
		self.model = model # a clm

	def forward(self, input_ids, labels=None, attention_mask=None):
		reversed_ids = torch.flip(input_ids.clone(), dims=[1])
		if attention_mask is not None:
			attn_mask = torch.flip(attention_mask.clone(), dims=[1])
		if labels is not None:
			labels = torch.flip(labels, dims=[1])
		model_output = self.model(input_ids=reversed_ids, attention_mask=attn_mask).logits
		logits = rearrange(logits, 'b t e -> b e t')

		if labels is not None:
			shift_logits = logits[..., :-1]
			shift_labels = labels[..., 1:]
			loss = self.cel(shift_logits, shift_labels)
		else:
			loss = 0
		return loss, logits

class BidirectionalTransformer(nn.Module):

	def __init__(self, n_vocab, dim, forward_model, reverse_model, last_loss_only=False):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.unreduced_cel = nn.CrossEntropyLoss(reduction='none')
		self.last_loss_only = last_loss_only
		self.tokenized_length = tokenized_length
		self.forward_model = forward_model
		self.reverse_model = reverse_model
		

	def forward(self, input_ids, labels=None, attention_mask=None):
		x = input_ids
		x = self.wte(x) # unified token embedding 
		y = torch.flip(x.clone(), dims=[1]) # reversed in token dim
		y_attn_mask = None
		if attention_mask is not None:
			y_attn_mask = torch.flip(attention_mask.clone(), dims=[1])
		
		forward = self.forward_model(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
		reverse = self.reverse_model(inputs_embeds=y, attention_mask=y_attn_mask).last_hidden_state
		pad = torch.zeros(x.shape[0], 1, x.shape[2]).to(input_ids.device)

		reverse = torch.cat([torch.flip(reverse, dims=[1])[..., 1:, :], pad], dim=1) # right pad reverse
		forward = torch.cat([pad, forward[..., :-1, :]], dim=1) # left pad forward

		output = self.lm_head(forward + reverse) # linear combination of f and r modules
		logits = rearrange(output, 'b t e -> b e t')
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		if self.last_loss_only:
			loss = self.unreduced_cel(logits, labels)[:, -1] # last token loss from all batch elements
		else:
			loss = self.cel(logits, labels)
		return loss, output

class OutsideInTransformer(nn.Module):

	def __init__(self, n_vocab, dim, forward_model, reverse_model):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.lm_head_f = nn.Linear(dim, n_vocab, bias=False)
		self.lm_head_r = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.unreduced_cel = nn.CrossEntropyLoss(reduction='none')
		self.tokenized_length = tokenized_length
		self.forward_model = forward_model
		self.reverse_model = reverse_model
		

	def forward(self, input_ids, labels=None, attention_mask=None):
		x = input_ids
		x = self.wte(x) # unified token embedding 
		y = torch.flip(x.clone(), dims=[1]) # reversed in token dim
		y_attn_mask = None
		if attention_mask is not None:
			y_attn_mask = torch.flip(attention_mask.clone(), dims=[1])
		
		# separate f/r modules not necessary as t_n+1 not in f_n or r_-n
		forward = self.forward_model(inputs_embeds=x+y, attention_mask=attention_mask).last_hidden_state
		reverse = self.reverse_model(inputs_embeds=x+y, attention_mask=y_attn_mask).last_hidden_state

		forward_output = self.lm_head_f(forward)
		reverse_output = self.lm_head_r(reverse)
		forward_logits = rearrange(forward_output, 'b t e -> b e t')
		reverse_logits = rearrange(reverse_output, 'b t e -> b e t')
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		if self.last_loss_only:
			loss_f = self.unreduced_cel(forward_logits, labels)[:, :input_ids.shape[1]//2] # first half of tokens are by head on forward modules
			loss_r = self.unreduced_cel(revers_logits, labels)[:, input_ids.shape[1]//2:] # second half are predicted by head on reverse modules
		return loss, output

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 1024
dim = 512
n_hidden_layers = 8 # half of the CLM equivalent
			
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': n_hidden_layers,
	'num_attention_heads': 4,
	'vocab_size': 4096
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
forward_model = LlamaModel(configuration)
reverse_model = LlamaModel(configuration)
model = BidirectionalTransformer(n_vocab, dim, forward_model, reverse_model)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
test_path =  f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

#map_dataset(train_path, test_path)
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

global_batch_size = 128
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_bidirectional\
_d{dim}\
_n{n_hidden_layers}\
_c{tokenized_length}_b{batch_size}x{n_devices}'

print (f"training model, saving to {output_dir}")

# train unique num_models, storing outputs from each
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=500,
	eval_steps=5000,
	logging_steps=500,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=200000,
	save_strategy='steps',
	save_steps=10000,
	torch_compile=True,
	report_to='none'
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

model.train()
trainer.train()

# evaluate last token prediction accuracy
print ('evaluating last token loss only')
model.last_loss_only = True
model.eval()
trainer.evaluate()
