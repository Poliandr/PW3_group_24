import torch
from torch import nn
from transformers import AutoModel

class SimplifiedImprovedModel(nn.Module):
    def __init__(self, pretrained_model, hidden_dim, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)  # Добавлен Dropout для регуляризации
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        features = self.dropout(features)  # Применение Dropout
        output = self.fc(features)
        return output

# Устройство для обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация модели
model = SimplifiedImprovedModel(
    pretrained_model="ai-forever/ruBert-base",
    hidden_dim=768,
    num_classes=len(labels)
)
model = model.to(device)
