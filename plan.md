# Промпт для агента по улучшению Jupyter Notebook

---

## Промпт

```
Ты — опытный ML-инженер. Тебе нужно проанализировать и улучшить Jupyter Notebook для бинарной классификации сагиттальных МРТ-снимков. Модель сейчас не обучается — метрики на уровне случайного угадывания. Ниже описаны выявленные проблемы, гипотезы и план действий. Действуй последовательно, вноси изменения и после каждого шага проверяй результат.

---

## ТЕКУЩЕЕ СОСТОЯНИЕ МОДЕЛИ

Задача: бинарная классификация сагиттальных снимков (central vs non-central).
Обучение проведено ~50 эпох. Результаты:
- Val Accuracy: ~50-60% (уровень случайного угадывания)
- Val AUC-ROC: колеблется 0.2–0.8, среднее ~0.4–0.5 (хуже случайного)
- Val F1: 0–0.4, крайне нестабилен
- Loss: train медленно падает (0.084→0.060), val loss НЕ падает и даже растёт
- Sensitivity и Specificity: на валидации периодически падают до 0.0 — модель коллапсирует в тривиальное предсказание одного класса
- Все кривые сильно осциллируют (зубчатые графики)
- Снижение learning rate (scheduler) не помогает стабилизировать обучение

---

## ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ

### Проблема 1: Модель не обобщает
- Val loss не снижается при снижении train loss
- Val accuracy на уровне случая
- AUC < 0.5 на некоторых эпохах (модель инвертирует предсказания)

### Проблема 2: Сильная нестабильность обучения
- Все метрики (loss, accuracy, sensitivity, specificity, F1, AUC) хаотично скачут от эпохи к эпохе
- Это указывает на: слишком высокий learning rate, маленький batch size, зашумлённые данные или неудачную функцию потерь

### Проблема 3: Коллапс классов
- Sensitivity и Specificity попеременно падают до 0.0
- Модель переключается между тривиальными решениями: "всё класс 0" ↔ "всё класс 1"
- Вероятная причина: дисбаланс классов, отсутствие взвешивания в loss

### Проблема 4: Подозрение на проблемы с данными
- Диапазон loss подозрительно узкий (0.06–0.085) — возможно, используется некорректная функция потерь или неправильная нормализация
- Возможны ошибки в разметке
- Возможно, данных слишком мало для выбранной архитектуры

### Проблема 5: Learning rate schedule неэффективен
- Два снижения LR видны на графике, но ни одно не привело к улучшению
- Начальный LR (~1e-4) может быть неоптимальным

---

## ПЛАН ДЕЙСТВИЙ (выполняй последовательно)

### Шаг 0: Диагностика данных
Перед изменением модели ОБЯЗАТЕЛЬНО проверь:

1. **Распределение классов**:
   - Выведи количество примеров каждого класса в train и val
   - Посчитай соотношение классов (ratio)
   - Если дисбаланс > 1:3, это критическая проблема

2. **Проверка разметки**:
   - Выведи 10-20 случайных примеров с их метками
   - Визуализируй несколько снимков каждого класса — убедись, что метки корректны
   - Проверь, нет ли дубликатов между train и val (data leakage)

3. **Проверка препроцессинга**:
   - Выведи min, max, mean, std значений пикселей после препроцессинга
   - Убедись, что нормализация соответствует ожиданиям модели (ImageNet stats или [0,1])
   - Проверь размеры входных изображений

4. **Проверка DataLoader**:
   - Загрузи один батч, выведи shapes, dtype, диапазон значений
   - Убедись, что метки — это 0 и 1 (не one-hot, если используется BCEWithLogitsLoss)
   - Проверь, что аугментации не искажают данные критически

```python
# Код для диагностики — вставь и выполни
print("=== Распределение классов ===")
# train_labels = [label for _, label in train_dataset]
# val_labels = [label for _, label in val_dataset]
# print(f"Train: {Counter(train_labels)}")
# print(f"Val: {Counter(val_labels)}")

print("\n=== Проверка батча ===")
# batch = next(iter(train_loader))
# images, labels = batch
# print(f"Images shape: {images.shape}, dtype: {images.dtype}")
# print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
# print(f"Labels unique: {labels.unique()}")
# print(f"Pixel range: [{images.min():.3f}, {images.max():.3f}]")
# print(f"Pixel mean: {images.mean():.3f}, std: {images.std():.3f}")
```

### Шаг 1: Исправление дисбаланса классов

Если обнаружен дисбаланс:

**Вариант A — Взвешенная функция потерь:**
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(train_labels))
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Для BCEWithLogitsLoss:
pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Для CrossEntropyLoss:
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Вариант B — Weighted Random Sampler:**
```python
from torch.utils.data import WeightedRandomSampler

class_counts = [count_class_0, count_class_1]
sample_weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
```

**Вариант C — Focal Loss (при сильном дисбалансе):**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

### Шаг 2: Стабилизация обучения

**2.1. Увеличить batch size:**
- Текущий batch size скорее всего маленький (8-16)
- Увеличь до 32-64 (или максимум, что влезает в GPU)
- Если GPU памяти не хватает, используй gradient accumulation:

```python
accumulation_steps = 4  # Эффективный batch = batch_size * accumulation_steps
optimizer.zero_grad()
for i, (images, labels) in enumerate(train_loader):
    loss = criterion(model(images), labels)
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**2.2. Уменьшить learning rate:**
```python
# Попробуй начальный LR = 1e-5 вместо 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
```

**2.3. Добавить warmup:**
```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-7)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
```

**2.4. Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Шаг 3: Проверка и исправление функции потерь

Диапазон loss 0.06–0.085 подозрительно узкий для BCE. Проверь:

```python
# Убедись, что используется правильная loss функция
# Если выход модели — logits (без sigmoid), используй:
criterion = nn.BCEWithLogitsLoss()

# Если выход модели — после sigmoid, используй:
criterion = nn.BCELoss()

# ЧАСТАЯ ОШИБКА: sigmoid применён дважды — в модели и в loss
# Проверь последний слой модели!
print(model)  # Посмотри, есть ли Sigmoid в конце

# Тест: прогони один батч и проверь выходы
with torch.no_grad():
    outputs = model(images.to(device))
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"Output shape: {outputs.shape}")
    # Если выходы уже в [0,1] и ты используешь BCEWithLogitsLoss — это ошибка!
    # Если выходы в [-inf, +inf] и ты используешь BCELoss — это тоже ошибка!
```

### Шаг 4: Регуляризация

```python
# Добавь/увеличь dropout
# В модели перед последним линейным слоем:
self.dropout = nn.Dropout(p=0.5)

# Weight decay в оптимизаторе (если ещё нет):
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
```

**Аугментации (если их мало):**
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Валидационные — БЕЗ аугментаций!
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Шаг 5: Transfer Learning

Если модель обучается с нуля — это может быть ключевой проблемой при малом количестве данных:

```python
import torchvision.models as models

# Используй предобученную модель
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Заморозь backbone на первые эпохи
for param in model.parameters():
    param.requires_grad = False

# Замени классификатор
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 1)  # 1 выход для бинарной классификации
)

# Разморозь через 5-10 эпох:
# for param in model.parameters():
#     param.requires_grad = True
# И уменьши LR при разморозке
```

### Шаг 6: Правильный training loop с Early Stopping

```python
best_val_auc = 0.0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_auc = roc_auc_score(all_labels, all_preds)
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✓ New best AUC: {val_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    scheduler.step()
```

### Шаг 7: Быстрый Sanity Check

Перед полным обучением убедись, что модель МОЖЕТ переобучиться на маленькой подвыборке:

```python
# Overfit test: модель должна достичь ~100% accuracy на 1-2 батчах
small_dataset = torch.utils.data.Subset(train_dataset, range(32))
small_loader = DataLoader(small_dataset, batch_size=32, shuffle=True)

model_test = create_model()  # Свежая модель
optimizer_test = torch.optim.Adam(model_test.parameters(), lr=1e-3)

for epoch in range(100):
    for images, labels in small_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer_test.zero_grad()
        loss = criterion(model_test(images), labels)
        loss.backward()
        optimizer_test.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Если loss не падает до ~0 — проблема в архитектуре или данных!
```

---

## ПРИОРИТЕТ ДЕЙСТВИЙ

1. 🔴 **Шаг 0** — Диагностика (ОБЯЗАТЕЛЬНО ПЕРВЫМ)
2. 🔴 **Шаг 7** — Sanity check (быстро покажет, жива ли модель)
3. 🔴 **Шаг 3** — Проверка loss (частая причина "странных" значений)
4. 🟡 **Шаг 1** — Дисбаланс классов
5. 🟡 **Шаг 2** — Стабилизация (LR, batch size)
6. 🟢 **Шаг 5** — Transfer learning
7. 🟢 **Шаг 4** — Регуляризация
8. 🟢 **Шаг 6** — Training loop + early stopping

После каждого шага запускай обучение на 10-15 эпох и сравнивай кривые с текущими. Жду отчёт по каждому шагу.
```
