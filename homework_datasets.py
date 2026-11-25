import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression

print("=== ЗАДАНИЕ 2: Работа с датасетами ===\n")

# ==========================================
# 2.1 Кастомный Dataset класс для CSV
# ==========================================
class CustomCSVDataset(Dataset):
    def __init__(self, csv_path, target_col, task_type='classification'):
        self.df = pd.read_csv(csv_path)
        self.task_type = task_type
        self.target_col = target_col
        
        self.y = self.df.pop(target_col).values
        self.df = pd.get_dummies(self.df, drop_first=True)
        self.X = self.df.values.astype(np.float32)
        
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        if self.task_type == 'classification':
            self.y = self.y.astype(np.int64)
        else:
            self.y = self.y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ==========================================
# 2.2 Эксперименты с датасетами
# ==========================================
def run_dataset_experiment():
    # Генерируем и сохраняем CSV для классификации
    X_clf, y_clf = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    df_clf = pd.DataFrame(X_clf, columns=[f'feat_{i}' for i in range(10)])
    df_clf['target'] = y_clf
    df_clf.to_csv('data/dummy_classification.csv', index=False)
    
    # Генерируем и сохраняем CSV для регрессии
    X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    df_reg = pd.DataFrame(X_reg, columns=[f'feat_{i}' for i in range(5)])
    df_reg['target'] = y_reg
    df_reg.to_csv('data/dummy_regression.csv', index=False)

    print("✅ Датасеты созданы в папке data/")

    # Загрузка и обучение на классификации
    dataset_clf = CustomCSVDataset('data/dummy_classification.csv', target_col='target', task_type='classification')
    train_size = int(0.8 * len(dataset_clf))
    val_size = len(dataset_clf) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_clf, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = nn.Linear(10, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(5):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            
    print(" Обучение на кастомном CSV-датасете успешно завершено!")

if __name__ == "__main__":
    run_dataset_experiment()
