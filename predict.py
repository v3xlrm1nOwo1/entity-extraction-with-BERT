from tqdm import tqdm

import joblib
import torch

import config
import dataset
import opendelta
import prepare_dataset
from bert_model import EntityModel


if __name__ == '__main__':

    meta_data = joblib.load('meta.bin')
    enc_pos = meta_data['enc_pos']
    enc_tag = meta_data['enc_tag']

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    data = prepare_dataset.process_data(config.DATASET_PATH)[0]
    test_data_loader = dataset.create_data_loader(data['test_sentences'], data['test_pos'], data['test_tag'], tokenizer=config.TOKENIZER, max_len=config.MAX_LEN, batch_size=config.TEST_BATCH_SIZE, include_row_text=False, shuffle=False)

    device = config.DEVICE
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)

    model.to(device)
    delta_model = opendelta.AutoDeltaModel.from_finetuned('/content/bert_model_state', backbone_model=model)
    
    final_loss = 0
    with torch.no_grad():
        for d in tqdm(test_data_loader, total=len(test_data_loader)):
            for k, v in d.items():
                d[k] = v.to(device)
                
            tag, pos, loss = model(**d)
            
            final_loss += loss.item()

    print(final_loss / len(test_data_loader))