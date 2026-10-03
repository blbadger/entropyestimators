import torch
from einops import rearrange
import transformers
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from transformers import BertConfig, BertForMaskedLM, ModernBertConfig, ModernBertForMaskedLM
from prettytable import PrettyTable
import shutil
from dotenv import load_dotenv
import os
import pathlib

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.mask_token_id = len(tokenizer) - 1
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 512
n_hidden_layers = 16
			
bert_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': n_hidden_layers,
	'num_attention_heads': 4,
	'vocab_size': len(tokenizer)
}

# Initializing a Bert model
configuration = BertConfig(**bert_config_kwargs)
model = BertForMaskedLM(configuration)

configuration = ModernBertConfig(**bert_config_kwargs)
model = ModernBertForMaskedLM(configuration)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
test_path =  f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

#map_dataset(train_path, test_path)
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

# get number of devices (assumes that all visible devices are used for training)
global_batch_size=128
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_bert_0.15mlm\
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
	torch_compile=False,
	report_to='none'
)

mlm_probs = 0.15 if isinstance(model, BertForMaskedLM) else 0.3

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=mlm_probs),
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

model.train()
trainer.train()
