import config
import torch
import transformers
import torch.nn as nn

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    
    loss = lfn(active_logits, active_labels)
    return loss



class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.CHECKPOINT, return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(self.model.config.hidden_size, self.num_tag)
        self.out_pos = nn.Linear(self.model.config.hidden_size, self.num_pos)
    
    def forward(self, input_ids, attention_mask, token_type_ids, label_pos, label_tag):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        bo_tag = self.bert_drop_1(output)
        bo_pos = self.bert_drop_2(output)

        output_tag = self.out_tag(bo_tag)
        output_pos = self.out_pos(bo_pos)

        loss_tag = loss_fn(output_tag, label_tag, attention_mask, self.num_tag)
        loss_pos = loss_fn(output_pos, label_pos, attention_mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return output_tag, output_pos, loss
