import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

print("=== ЗАДАНИЕ 3: Эксперименты и анализ ===\n")

# ==========================================
# 3.1 Исследование гиперпараметров
# ==========================================
def experiment_hyperparameters():
    X = torch.randn(500, 5)
    y = torch.randn(500, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    learning_rates = [0.001, 0.01, 0.1]
    optimizers = {'SGD': optim.SGD, 'Adam': optim.Adam}
    
    results = {}
    plt.figure(figsize=(10, 6))
    
    for opt_name, opt_class in optimizers.items():
        for lr in learning_rates:
            model = nn.Linear(5, 1)
            optimizer = opt_class(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            losses = []
            for epoch in range(30):
                epoch_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(loader))
            
            label = f"{opt_name} (lr={lr})"
            results[label] = losses[-1]
            plt.plot(losses, label=label, linewidth=2)
            
    plt.title('Сравнение оптимизаторов и learning rate')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/hyperparams_comparison.png', dpi=150)
    print("📊 График сохранен в plots/hyperparams_comparison.png")

# ==========================================
# 3.2 Feature Engineering
# ==========================================
def experiment_feature_engineering():
    X_base = np.random.randn(500, 2)
    # Нелинейная зависимость: y = 3*x1^2 + 2*x2 + шум
    y = 3 * X_base[:, 0]**2 + 2 * X_base[:, 1] + np.random.randn(500) * 0.5
    
    # Базовая линейная модель
    model_base = nn.Linear(2, 1)
    X_t_base = torch.tensor(X_base, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    opt_base = optim.SGD(model_base.parameters(), lr=0.01)
    for _ in range(100):
        opt_base.zero_grad()
        loss = nn.MSELoss()(model_base(X_t_base), y_t)
        loss.backward()
        opt_base.step()
    loss_base = nn.MSELoss()(model_base(X_t_base), y_t).item()
    
    # Feature Engineering: Полиномиальные признаки
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    
    model_poly = nn.Linear(X_poly.shape[1], 1)
    X_t_poly = torch.tensor(X_poly, dtype=torch.float32)
    
    opt_poly = optim.SGD(model_poly.parameters(), lr=0.01)
    for _ in range(100):
        opt_poly.zero_grad()
        loss = nn.MSELoss()(model_poly(X_t_poly), y_t)
        loss.backward()
        opt_poly.step()
    loss_poly = nn.MSELoss()(model_poly(X_t_poly), y_t).item()
    
    print("\n Feature Engineering:")
    print(f"  Ошибка базовой линейной модели: {loss_base:.4f}")
    print(f"  Ошибка модели с полиномиальными признаками: {loss_poly:.4f}")
    print("  Вывод: Добавление нелинейных признаков значительно улучшило качество!")

if __name__ == "__main__":
    experiment_hyperparameters()
    experiment_feature_engineering()
