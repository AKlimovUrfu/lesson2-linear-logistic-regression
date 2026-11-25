import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("=== ЗАДАНИЕ 1: Модификация моделей ===\n")

# ==========================================
# 1.1 Линейная регрессия с L1/L2 и Early Stopping
# ==========================================
class LinearRegRegularized(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.l1_lambda = 0.01
        self.l2_lambda = 0.01

    def forward(self, x):
        return self.linear(x)

    def get_l1_penalty(self):
        return self.l1_lambda * torch.sum(torch.abs(self.linear.weight))

    def get_l2_penalty(self):
        return self.l2_lambda * torch.sum(self.linear.weight ** 2)

def train_with_early_stopping(X_train, y_train, X_val, y_val, epochs=100, lr=0.01, patience=5):
    model = LinearRegRegularized(X_train.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        
        # Loss + L1 + L2 regularization
        loss = criterion(predictions, y_train) + model.get_l1_penalty() + model.get_l2_penalty()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping сработал на эпохе {epoch+1}")
            break
            
    model.load_state_dict(best_weights)
    print(f"✅ Линейная регрессия: Лучшая валидационная ошибка: {best_val_loss:.4f}\n")
    return model

# ==========================================
# 1.2 Логистическая регрессия (Multiclass + метрики)
# ==========================================
class LogisticRegMulticlass(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def evaluate_multiclass(model, X, y, num_classes):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
    y_np = y.numpy()
    preds_np = preds.numpy()
    probs_np = probs.numpy()
    
    try:
        roc_auc = roc_auc_score(y_np, probs_np, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = 0.0 
        
    precision = precision_score(y_np, preds_np, average='macro', zero_division=0)
    recall = recall_score(y_np, preds_np, average='macro', zero_division=0)
    f1 = f1_score(y_np, preds_np, average='macro', zero_division=0)
    
    print("✅ Логистическая регрессия (Multiclass):")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    cm = confusion_matrix(y_np, preds_np)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Multiclass)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('plots/confusion_matrix.png')
    print("📊 Confusion matrix сохранена в plots/confusion_matrix.png")

# Тестовый запуск на синтетических данных
if __name__ == "__main__":
    X_t = torch.randn(100, 5)
    y_t_reg = torch.randn(100, 1)
    X_val_t = torch.randn(20, 5)
    y_val_t = torch.randn(20, 1)
    
    train_with_early_stopping(X_t, y_t_reg, X_val_t, y_val_t)
    
    y_t_clf = torch.randint(0, 3, (100,))
    model_log = LogisticRegMulticlass(5, 3)
    evaluate_multiclass(model_log, X_t, y_t_clf, 3)
