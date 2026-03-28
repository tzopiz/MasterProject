# Экспорт презентации в PDF или PPTX

## Быстрый способ — PDF из браузера (без установки)

1. Откройте `presentation.html` в Chrome или Safari (двойной клик по файлу или перетащите в окно браузера).
2. Нажмите **Cmd+P** (Mac) или **Ctrl+P** (Windows/Linux) — печать.
3. В качестве принтера выберите **«Сохранить как PDF»** / **«Save as PDF»**.
4. Включите опцию **«Фоновая графика»** / **«Background graphics»**, чтобы сохранились фон и градиенты.
5. Сохраните файл.

Готовый PDF будет соответствовать тому, как выглядит страница в браузере.

---

## Автоматический способ — скрипты

### PDF

```bash
cd /Users/tzopiz/Developer/MasterProject
python3 -m venv .venv_export
source .venv_export/bin/activate   # Windows: .venv_export\Scripts\activate
pip install playwright python-pptx
playwright install chromium
python3 export_presentation_pdf.py
```

Результат: `presentation.pdf` в корне проекта.

### PPTX (каждая секция — слайд)

```bash
source .venv_export/bin/activate
python3 export_presentation_pptx.py
```

Результат: `presentation.pptx` в корне проекта (5 слайдов: титул, проблема/решение, возможности, метрики, рынок и CTA).

---

## Только PDF через скрипт (если уже есть venv)

```bash
pip install playwright
playwright install chromium
python3 export_presentation_pdf.py
```
