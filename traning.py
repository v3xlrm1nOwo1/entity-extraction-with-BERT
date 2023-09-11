import numpy as np

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import prepare_dataset
import dataset
import opendelta
import engine
from bert_model import EntityModel

device = config.DEVICE

data, num_pos, num_tag = prepare_dataset.process_data(config.DATASET_PATH)

train_data_loader = dataset.create_data_loader(data['train_sentences'], data['train_pos'], data['train_tag'], tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
valid_data_loader = dataset.create_data_loader(data['valid_sentences'], data['valid_pos'], data['valid_tag'], tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.VALID_BATCH_SIZE, shuffle=False) 
test_data_loader = dataset.create_data_loader(data['test_sentences'], data['test_pos'], data['test_tag'], tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.TEST_BATCH_SIZE, include_row_text=False, shuffle=False)

model = EntityModel(num_tag=num_tag, num_pos=num_pos)
model.to(device)

delta_model = opendelta.AdapterModel(model)
delta_model.freeze_module(exclude=['deltas', 'classifier'])
delta_model.log()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_parameters = [
    {
        'params': [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        'weight_decay': 0.001,
    },
    {
        'params': [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        'weight_decay': 0.0,
    },
]

num_train_steps = int(len(data['train_sentences']) / config.TRAIN_BATCH_SIZE * config.NUM_EPOCHS)
optimizer = AdamW(optimizer_parameters, lr=config.LEARN_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)


best_loss = np.inf
for epoch in range(config.NUM_EPOCHS):
    train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
    
    valid_loss = engine.eval_fn(valid_data_loader, model, device)
    
    print('=' * 50)
    print(f'Train Loss = {train_loss} - Valid Loss = {valid_loss}')
    print('=' * 50)
    
    if valid_loss < best_loss:
        delta_model.save_finetuned('bert_model_state')
        best_loss = valid_loss
        
        