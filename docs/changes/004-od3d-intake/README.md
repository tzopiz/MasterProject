# 004 — Проверяемый вход TMJ-OD3D

Статус: in-progress. Пользователь поручил автономную реализацию до запуска обучения.

## Дизайн и приёмка

[Авторский README](https://github.com/ZiTingW/TMJ-OD3D/blob/a4bce89c89b7175e330193f04a282887b5e3f8b0/README.md)
связан с тем же DOI и определяет 0=normal, 1=cortical erosion,
2=subchondral sclerosis, 3=subchondral cystic changes, 4=condylar flattening,
5=osteophyte, 6=other. Это отдельная онтология.
CSV использует |, JSON — comma. 0 с ненулевым кодом недопустим.
Пустые метки неизвестны. Сохранить исходные сочетания для будущего multilabel.
Авторский код предшествует V2; CSV/JSON должны подтвердить применимость схемы
при фактическом импорте, несоответствия не исправляются предположениями.

JSON содержит root shapes и/или frames/shapes, points в x,y и label
codes-side-start-end. Диапазон включительно по InstanceNumber.
Прямоугольник min/max двух точек. Проверять ожидаемую сторону,
конечные координаты и ненулевую площадь, известные коды и порядок диапазона.
Пустой JSON не означает норму. Существующий position intake не менять.
Исключения содержат фиксированные коды без patient rows/paths.

## План реализации

Применяется writing-plans; targeted behavioral tests.

1. training/tmj_od3d_inputs.py: константы источника/codebook;
   parse_codes(value, separator) → tuple[int,...] либо None для пустого;
   load_metadata(path) → ID→{left:codes,right:codes,slice_min:int,slice_max:int,slice_count:int};
   parse_annotation(path, expected_side) → list[dict] с codes, side,
   instance_start, instance_end, bbox_xyxy, reference_basename, width, height.
2. tests/test_tmj_od3d_inputs.py: норма/комбинации/пустое,
   неизвестные и конфликт0, дубликатыID, side mismatch, bounds,
   обратная диагональ, несколько shapes, root/frame, basename без пути.
3. Проверить на приватном реальном CSV/JSON без вывода строк.
4. Обновить связи требований и выполнить check_specs.

## Последующие зависимости

Отдельная подготовка изображений сохраняет авторские ROI как oracle_roi.
Доступные срезы — bag of 2D slices, без предположения о непрерывном 3D affine.
Выбор одинакового числа срезов по одинаковому правилу для всех кодов;
patient split, train/eval и cloud launch — следующие проверяемые изменения.

## Результаты

Реализованы только строгие входные функции `parse_codes`, `load_metadata`,
`parse_annotation`; существующий position intake не изменён. Исходники:
[tmj_od3d_inputs.py](../../../MLService/training/tmj_od3d_inputs.py),
[синтетические тесты](../../../MLService/tests/test_tmj_od3d_inputs.py).
CSV `selected_slice_min/max` сохраняются как исходные числовые части имён
файлов и не подменяют JSON InstanceNumber. Для sparse selection допускается
`1 <= slice_count <= slice_max - slice_min + 1`; наличие файлов и соответствие
номеров проверяет следующий этап подготовки изображений.

TDD: первоначальные 56 сценариев отказали из-за отсутствующего модуля;
отдельный regression воспроизвёл OverflowError для чрезмерной целочисленной
координаты, после исправления она отклоняется безопасным фиксированным кодом.
После форматирования: **58 passed** командой
`cloud-pins-venv/bin/python -m pytest -p no:cacheprovider tests/test_tmj_od3d_inputs.py -q`
из MLService. Runtime: существующий Python 3.12 в
`MLService/docs/ai/tasks/GH-85/cloud-pins-venv`. Ruff check/format --check
двух новых файлов прошли; использован установленный Ruff 0.15.10.

Приватный реальный CSV разобран без вывода идентификаторов и строк:
1043 пациента, 2086 сторон; 11 неизвестных меток, 729 авторских normal,
1346 abnormal, 517 сочетаний. Два доступных реальных JSON (L/R) разобраны,
каждый содержит один прямоугольник. Это проверка этих входов, а не всего
архива, независимости пациентов или анатомической пригодности ROI.

Полный suite пока не подтверждён: первая попытка остановилась на collection
конкурентно создаваемого osseous trainer, повторная была прервана по поручению
координатора после 165 passed. Итоговую совместную проверку проводит
координатор после завершения зависимых изменений. Обучение, paid jobs,
загрузка полного архива и публикация данных не выполнялись.
