import transformers
import torch.nn as nn


class BERTBasedUncased(nn.Module):
    def __init__(self):
        super(BerertBasedUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(786,1)

# Gives you a avector of 768 based on how the BERTBasedUncased model
    def forward(self,ids,mask,token_type_ids):
        _,o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output