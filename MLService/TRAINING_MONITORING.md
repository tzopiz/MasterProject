# Команды для мониторинга обучения TMJ Detector

## 📊 Основные команды:

### 1. Посмотреть текущий статус (последние 50 строк):
```bash
tail -50 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

### 2. Следить за обучением в реальном времени:
```bash
tail -f /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```
(Нажми Ctrl+C чтобы выйти)

### 3. Посмотреть последнюю эпоху:
```bash
tail -20 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

### 4. Поиск по ключевым словам (например, "best"):
```bash
grep "best" /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

### 5. Посмотреть только эпохи:
```bash
grep "Epoch" /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

### 6. Посмотреть validation MAE:
```bash
grep "Val   Loss" /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

---

## 🔍 Что искать в логе:

### **Формат вывода одной эпохи:**
```
Epoch 10/200
  Train Loss: 0.0234, MAE: 45.32 px
  Val   Loss: 0.0198, MAE: 38.45 px
  Val MAE - Left: 36.21, Right: 40.69
  Val MAE - Z: 15.32, Y: 22.45, X: 18.67
  LR: 0.000100
  ✅ Saved best model (MAE: 38.45 px)
```

### **Ключевые метрики:**
- **Val MAE** - главная метрика (чем меньше, тем лучше)
- **< 30 px** - отлично! ✅
- **30-50 px** - хорошо ✅
- **> 50 px** - работает, но неточно ⚠️

---

## ⚡ Быстрые проверки:

### Узнать текущую эпоху:
```bash
tail -50 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | grep "Epoch"
```

### Узнать лучший результат:
```bash
grep "Saved best model" /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | tail -1
```

### Проверить завершилось ли обучение:
```bash
tail -10 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | grep -E "(COMPLETE|Early stopping)"
```

### Посмотреть финальный результат:
```bash
tail -20 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | grep -A 5 "TRAINING COMPLETE"
```

---

## 📁 Проверить созданные файлы:

### Посмотреть директорию эксперимента:
```bash
ls -la experiments/detector_*/
```

### Проверить что модель сохранилась:
```bash
ls -lh experiments/detector_*/best_model.pth
```

### Посмотреть конфиг обучения:
```bash
cat experiments/detector_*/config.json
```

---

## 🎯 Рекомендованный workflow:

### **Каждые 5-10 минут:**
```bash
tail -30 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

### **Когда хочешь увидеть весь процесс:**
```bash
cat /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | less
```
(Используй стрелки для навигации, Q для выхода)

---

## 🚨 Если что-то пошло не так:

### Проверить есть ли ошибки:
```bash
grep -i "error\|traceback\|failed" /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

### Посмотреть последние 100 строк (для debug):
```bash
tail -100 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

---

## 💡 Короткие алиасы (опционально):

Добавь в `~/.zshrc` или `~/.bashrc`:

```bash
# Алиасы для мониторинга обучения
alias train-status='tail -50 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt'
alias train-watch='tail -f /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt'
alias train-best='grep "Saved best model" /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | tail -1'
alias train-done='tail -20 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | grep -E "COMPLETE|Early stopping"'
```

После добавления:
```bash
source ~/.zshrc  # или ~/.bashrc
train-status     # Посмотреть статус
```

---

## 📊 Пример хорошего лога:

```
2025-11-24 01:40:00,123 - INFO - Starting training...

Epoch 1/200
  Train Loss: 0.1234, MAE: 120.45 px
  Val   Loss: 0.1156, MAE: 115.32 px
  ✅ Saved best model (MAE: 115.32 px)

Epoch 10/200
  Train Loss: 0.0456, MAE: 65.23 px
  Val   Loss: 0.0421, MAE: 58.67 px
  ✅ Saved best model (MAE: 58.67 px)

Epoch 50/200
  Train Loss: 0.0123, MAE: 28.45 px
  Val   Loss: 0.0134, MAE: 32.15 px
  ✅ Saved best model (MAE: 32.15 px)

Early stopping after 90 epochs

TRAINING COMPLETE
Best validation MAE: 32.15 pixels
Model saved to: experiments/detector_20251124_013648/best_model.pth
```

---

## ✅ Быстрая проверка прямо сейчас:

Запусти это чтобы увидеть текущий статус:

```bash
tail -30 /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt
```

Или это для best result:

```bash
grep "Saved best" /Users/tzopiz/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/4.txt | tail -1
```

