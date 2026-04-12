# iOS App

SwiftUI-клиент **AI Doctor** для загрузки КЛКТ и просмотра результатов анализа ВНЧС.

## Содержимое

- **`MasterDoctor/`** — Xcode-проект, модули `MainFeatures`, `Foundation`, `CommonCore` и т.д.
- **`ModularAppArchitecture.txt`** — устройство модулей и зависимостей (UTF-8; в каталоге только один `README.md` и файлы не `.md`).

## Запуск

Откройте в Xcode:

`iOSApp/MasterDoctor/MasterDoctor.xcodeproj`

Базовый URL Backend и пути API задаются в коде (например, `AnalysisEndpoint`).

## Связка с репозиторием

Корневой обзор и цепочка ML → Backend → iOS: [README.md](../README.md).
