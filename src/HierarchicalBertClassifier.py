
from torch import nn
from transformers import BertModel

class HierarchicalBertClassifier(nn.Module):
    def __init__(self, num_categories_per_level):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        # Create separate classifiers for each level
        self.classifiers = nn.ModuleList([
            nn.Linear(768, num_cats) for num_cats in num_categories_per_level
        ])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each level
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits