import config
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class EntityDataset(Dataset):
    def __init__(self, texts: pd.DataFrame, pos: pd.DataFrame, tags: pd.DataFrame, tokenizer: config.TOKENIZER, max_len: int = config.MAX_LEN, include_row_text=False):
        self.texts = texts
        self.pos = pos
        self.tags = tags
        self.tokenizer = tokenizer
        self.include_row_text = include_row_text
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        label_pos = []
        label_tag =[]

        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(
                s,
                add_special_tokens=False
            )

            input_len = len(inputs)
            ids.extend(inputs)
            label_pos.extend([pos[i]] * input_len)
            label_tag.extend([tags[i]] * input_len)

        ids = ids[: self.max_len - 2]
        label_pos = label_pos[: self.max_len - 2]
        label_tag = label_tag[: self.max_len - 2]

        ids = [101] + ids + [102]
        label_pos = [0] + label_pos + [0]
        label_tag = [0] + label_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        label_pos = label_pos + ([0] * padding_len)
        label_tag = label_tag + ([0] * padding_len)

        output = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label_pos': torch.tensor(label_pos, dtype=torch.long),
            'label_tag': torch.tensor(label_tag, dtype=torch.long),
        }
        
        if self.include_row_text:
            output['sentences'] = text
            
        return output
        
     

def create_data_loader(texts, pos, tags, tokenizer, max_len=config.MAX_LEN, batch_size=config.TRAIN_BATCH_SIZE, include_row_text=False, shuffle=False):
  ds = EntityDataset(
    texts=texts,
    pos=pos,
    tags=tags,
    tokenizer=tokenizer,
    max_len=max_len,
    include_row_text=include_row_text
  )

  return DataLoader(
    ds,
    shuffle=shuffle,
    batch_size=batch_size,
    num_workers=4,
  )   

