# Modular App Architecture (MAA)

> Паттерн архитектуры для iOS приложений с чётким разделением слоёв и однонаправленным потоком данных.

---

## Обзор архитектуры

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌───────────┐    ┌───────────┐    ┌─────────────────────────┐  │
│  │   View    │◄───│ ViewStore │◄───│         Store           │  │
│  └───────────┘    └───────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Domain Layer                              │
│  ┌───────────────┐  ┌────────────────────┐  ┌────────────────┐  │
│  │   Provider    │  │   Mutable Models   │  │Immutable Models│  │
│  └───────────────┘  └────────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Service-Access Layer (SAL)                    │
│  ┌───────────┐    ┌────────────────────┐    ┌────────────────┐  │
│  │  Client   │    │ DTOToDomainMapper  │    │DomainToDTOMapper│  │
│  └───────────┘    └────────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data-Access Layer                            │
│                      ┌───────────┐                               │
│                      │  Storage  │                               │
│                      └───────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Service-Access Layer (SAL)

### Client

Осуществляет сетевые запросы к бекенду.

**Нейминг:** `{EndpointName}Client`

**Примеры:**
- `AuthClient` — для авторизации
- `ProfileClient` — для работы с профилем пользователя
- `WorkoutClient` — может работать с несколькими ендпоинтами тренировок

**Правила:**
- Основные интерфейсы клиентов **не работают** напрямую с `Promise`/`Future`
- Могут расширяться до асинхронных интерфейсов через extensions
- Клиент не управляет состоянием асинхронных операций — только комбинирует входящие параметры, запрос и колбеки

```swift
protocol WorkoutClientProtocol {
    func fetchWorkout(id: String, completion: @escaping (Result<Workout, Error>) -> Void)
    func saveWorkout(_ request: SaveWorkoutRequest, completion: @escaping (Result<Workout, Error>) -> Void)
}

// Extension для async/await
extension WorkoutClientProtocol {
    func fetchWorkout(id: String) async throws -> Workout {
        try await withCheckedThrowingContinuation { continuation in
            fetchWorkout(id: id) { result in
                continuation.resume(with: result)
            }
        }
    }
}
```

### DTOToDomainMapper

Конвертирует DTO (Data Transfer Object) из сети в доменные модели.

**Нейминг:** `{DTOName}ToDomainMapper`

```swift
struct ExerciseDTOToDomainMapper {
    func map(_ dto: ExerciseDTO) -> Exercise {
        Exercise(
            name: dto.exercise_name,
            muscleGroup: MuscleGroup(rawValue: dto.muscle_group) ?? .other,
            equipment: Equipment(rawValue: dto.equipment_type)
        )
    }
}
```

### DomainToDTOMapper

Конвертирует доменные модели в DTO для отправки на сервер.

**Нейминг:** `{DomainName}ToDTOMapper`

```swift
struct ExerciseToDTOMapper {
    func map(_ domain: Exercise) -> ExerciseDTO {
        ExerciseDTO(
            exercise_name: domain.name,
            muscle_group: domain.muscleGroup.rawValue,
            equipment_type: domain.equipment?.rawValue
        )
    }
}
```

---

## Data-Access Layer

### Storage

Используется для сохранения и чтения локальных данных.

**Нейминг:** `{DomainName}Storage`

**Примеры:**
- `SessionStorage` — хранение сессии пользователя
- `SettingsStorage` — настройки приложения
- `WorkoutHistoryStorage` — история тренировок

```swift
protocol SessionStorageProtocol {
    var currentSession: Session? { get }
    func save(_ session: Session)
    func clear()
}
```

---

## Domain Layer

Содержит данные предметной области: структуры данных и базовое поведение.

**Ключевые принципы:**
- Не имеет связи с UI
- Не занимается вопросами представления
- Может взаимодействовать только с объектами Domain слоя и SAL
- Все важные данные должны быть в Domain, чтобы можно было полностью восстановить состояние приложения

### Provider

Инкапсулирует бизнес-логику и оперирует моделями с постоянным состоянием.

**Нейминг:** `{DomainName}Provider`

**Характеристики:**
- Переиспользуется между несколькими представлениями
- Шарит состояние между компонентами
- Содержит логику реактивного обновления данных

```swift
final class WorkoutProgressProvider {
    private let workoutClient: WorkoutClientProtocol
    
    @Published private(set) var progress: WorkoutProgress?
    
    // При изменении набора упражнений или интенсивности — автоматически
    // пересчитывает калории и длительность тренировки
    func updateExercises(_ exercises: [Exercise], intensity: Intensity) {
        // ...
    }
}
```

### Immutable Models

Переносят данные без изменения. Являются value types.

```swift
struct Exercise: Equatable, Hashable {
    let name: String
    let muscleGroup: MuscleGroup
    let equipment: Equipment?
}

struct WorkoutProgress: Equatable {
    let duration: TimeInterval
    let caloriesBurned: Int
    let completedSets: Int
}
```

### Mutable Models

Хранят изменяемое состояние, но не содержат бизнес-логики.

> ⚠️ Использование таких моделей следует избегать

```swift
// Пример: состояние редактируемой тренировки
class WorkoutDraft {
    var selectedExercises: [Exercise]
    var restDuration: TimeInterval
    var intensity: Intensity?
    var scheduledDate: Date?
}
```

---

## Presentation Layer

### Паттерн MAA (Modular App Architecture)

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌─────────┐         ┌───────────┐         ┌─────────┐     │
│   │  View   │ ◄─────  │ ViewStore │ ◄─────  │  Store  │     │
│   └────┬────┘         └─────┬─────┘         └────┬────┘     │
│        │                    │                    │          │
│        │ ViewEvent          │                    │ Event    │
│        ▼                    │                    ▼          │
│   ┌─────────┐               │              ┌─────────┐      │
│   │ViewState│               │              │  State  │      │
│   └─────────┘               │              └─────────┘      │
│                             │                               │
│              ┌──────────────┴──────────────┐                │
│              │      Converters (pure)      │                │
│              │  • StateToViewState         │                │
│              │  • ViewEventToEvent         │                │
│              └─────────────────────────────┘                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Store

Фасад бизнес-логики для сцены.

**Ответственности:**
- Собирает State из доменных моделей (Provider)
- Принимает Event — события в терминах бизнес-операций
- Содержит сильную ссылку на Router для навигации

```swift
final class WorkoutStore: ObservableObject {
    @Published private(set) var state: WorkoutState
    
    private let progressProvider: WorkoutProgressProvider
    private let workoutClient: WorkoutClientProtocol
    private weak var router: WorkoutRouterProtocol?
    
    func send(_ event: WorkoutEvent) {
        switch event {
        case .selectIntensity(let intensity):
            state.selectedIntensity = intensity
            progressProvider.recalculateCalories(for: intensity)
            
        case .startWorkout:
            router?.showActiveWorkout(for: state.workoutDraft)
            
        case .addExercise:
            router?.showExercisePicker()
        }
    }
}
```

### State

Снапшот доменных моделей, влияющих на сцену.

```swift
struct WorkoutState: Equatable {
    var selectedExercises: [Exercise]
    var selectedIntensity: Intensity?
    var availableIntensities: [Intensity]
    var workoutProgress: WorkoutProgress?
    var isLoading: Bool
}
```

### ViewStore

Посредник между View и Store. Type-erasure обёртка.

**Характеристики:**
- `final class`
- Принимает Store и два конвертера
- Конвертеры являются **чистыми функциями**

```swift
final class ViewStore<ViewState, ViewEvent> {
    let state: Published<ViewState>.Publisher
    
    private let store: AnyObject
    private let stateConverter: (Any) -> ViewState
    private let eventConverter: (ViewEvent) -> Any
    
    init<State, Event>(
        store: Store<State, Event>,
        stateConverter: @escaping (State) -> ViewState,
        eventConverter: @escaping (ViewEvent) -> Event
    ) {
        // ...
    }
    
    func send(_ event: ViewEvent) {
        // Конвертирует ViewEvent -> Event и отправляет в Store
    }
}
```

### ViewState

Готовые к отображению данные. Конфигурация View.

```swift
struct WorkoutViewState: Equatable {
    let exercisesList: [ExerciseRowViewModel]
    let intensityItems: [IntensityItemViewModel]
    let caloriesText: String?
    let durationText: String?
    let startButtonTitle: String
    let isStartButtonEnabled: Bool
}

struct IntensityItemViewModel: Identifiable, Equatable {
    let id: String
    let title: String
    let description: String
    let iconName: String
    let isSelected: Bool
}
```

### ViewEvent

События в терминах View.

```swift
enum WorkoutViewEvent {
    case addExerciseTapped
    case exerciseRemoved(id: String)
    case intensitySelected(id: String)
    case startButtonTapped
}
```

### View

Отвечает за отрисовку UI.

**Типы View:**

#### 1. "Глупая" View
Не имеет собственного ViewStore, получает состояние извне.

```swift
// SwiftUI
struct IntensityCell: View {
    let viewModel: IntensityItemViewModel
    let onTap: () -> Void
    
    var body: some View {
        // ...
    }
}

// UIKit
class IntensityCellView: UIView {
    func configure(with viewModel: IntensityItemViewModel) {
        // ...
    }
}
```

#### 2. "Умная" View
Создаётся с собственным ViewStore.

```swift
struct WorkoutBuilderView: View {
    @StateObject private var viewStore: ViewStore<WorkoutViewState, WorkoutViewEvent>
    
    var body: some View {
        // ...
    }
}
```

---

## Конвертеры

### State → ViewState

Подготавливает данные для отображения.

```swift
struct WorkoutStateToViewStateConverter {
    private let durationFormatter: DurationFormatter
    
    func convert(_ state: WorkoutState) -> WorkoutViewState {
        WorkoutViewState(
            exercisesList: state.selectedExercises.map { exercise in
                ExerciseRowViewModel(
                    id: exercise.id,
                    name: exercise.name,
                    muscleGroupIcon: exercise.muscleGroup.iconName
                )
            },
            intensityItems: state.availableIntensities.map { intensity in
                IntensityItemViewModel(
                    id: intensity.id,
                    title: intensity.name,
                    description: intensity.description,
                    iconName: intensity.iconName,
                    isSelected: intensity.id == state.selectedIntensity?.id
                )
            },
            caloriesText: state.workoutProgress.map { 
                "\($0.caloriesBurned) kcal" 
            },
            durationText: state.workoutProgress.map {
                durationFormatter.format($0.duration)
            },
            startButtonTitle: "Начать тренировку",
            isStartButtonEnabled: state.selectedIntensity != nil 
                && !state.selectedExercises.isEmpty
        )
    }
}
```

### ViewEvent → Event

Преобразует UI события в бизнес-события.

```swift
struct WorkoutViewEventToEventConverter {
    func convert(_ viewEvent: WorkoutViewEvent, state: WorkoutState) -> WorkoutEvent? {
        switch viewEvent {
        case .addExerciseTapped:
            return .addExercise
            
        case .exerciseRemoved(let id):
            guard let exercise = state.selectedExercises.first(where: { $0.id == id }) else {
                return nil
            }
            return .removeExercise(exercise)
            
        case .intensitySelected(let id):
            guard let intensity = state.availableIntensities.first(where: { $0.id == id }) else {
                return nil
            }
            return .selectIntensity(intensity)
            
        case .startButtonTapped:
            return .startWorkout
        }
    }
}
```

### Правила типизации

```swift
// ✅ Хорошо — ViewEvent может быть typealias для Event
typealias WorkoutViewEvent = WorkoutEvent

// ❌ Плохо — Event не должен быть typealias для ViewEvent
typealias WorkoutEvent = WorkoutViewEvent

// ✅ Хорошо — для простых случаев State может быть ViewState
typealias WorkoutViewState = WorkoutState // только если нет преобразований

// ❌ Плохо — State не должен быть typealias для ViewState
typealias WorkoutState = WorkoutViewState
```

---

## Scene (Сцена)

Единица декомпозиции UI, представляющая часть экранного пространства.

**Примеры:**
- Экран
- Модальное окно
- Ячейка таблицы (в сложных случаях)

### Структура файлов сцены

```
WorkoutScene/
├── WorkoutDependencyContainer.swift    # DI контейнер
├── WorkoutScene.swift                  # State, Event, ViewState, ViewEvent
├── WorkoutSceneBuilder.swift           # Сборка сцены
├── WorkoutStore.swift                  # Бизнес-логика
├── WorkoutView.swift                   # UI (SwiftUI)
└── WorkoutViewController.swift         # UI (UIKit)
```

### SceneBuilder

```swift
struct WorkoutSceneBuilder {
    private let container: WorkoutDependencyContainer
    
    func build() -> WorkoutViewController {
        // 1. Создать Store
        let store = WorkoutStore(
            progressProvider: container.progressProvider,
            workoutClient: container.workoutClient,
            router: container.router
        )
        
        // 2. Создать ViewStore с конвертерами
        let viewStore = ViewStore(
            store: store,
            stateConverter: WorkoutStateToViewStateConverter().convert,
            eventConverter: WorkoutViewEventToEventConverter().convert
        )
        
        // 3. Создать ViewController
        return WorkoutViewController(viewStore: viewStore)
    }
}
```

---

## Dependency Injection

### DependencyContainer

Предоставляет зависимости и управляет их жизненным циклом.

```swift
final class WorkoutDependencyContainer {
    private let appContainer: AppDependencyContainer
    
    // Shared (singleton within container)
    lazy var progressProvider: WorkoutProgressProvider = {
        WorkoutProgressProvider(workoutClient: appContainer.workoutClient)
    }()
    
    // Factory (new instance each time)
    func makeWorkoutClient() -> WorkoutClientProtocol {
        WorkoutClient(networkService: appContainer.networkService)
    }
    
    var router: WorkoutRouterProtocol {
        WorkoutRouter(container: self)
    }
}
```

---

## Навигация

### Фреймворк BaseRouting

Базовый класс `BaseRouter` предоставляет фундаментальные возможности навигации:

1. Позволяет осуществлять `push` / `pop` / `present` / `dismiss` транзишены
2. Поддерживает стандартный `UINavigationController` и кастомные navigation controller'ы

### Базовый сценарий навигации

Все навигационные действия сцена должна осуществлять через роутер, закрытый протоколом `{SceneName}Routing`.

**Важно:** Роутер флоу не должен делать предположений о контексте, в котором запускается флоу. Это нужно для того, чтобы другие роутеры / обработчики диплинков могли запускать флоу в разных контекстах.

```swift
// Протокол навигации для экрана подписок
protocol SubscriptionPlansRouting {
    func routeBack()
}

// Реализация роутера
final class SubscriptionPlansRouter {
    let baseRouter: BaseRouter
    
    init(baseRouter: BaseRouter) {
        self.baseRouter = baseRouter
    }
    
    func routeToSubscriptionPlans() {
        let viewController = SubscriptionPlansSceneBuilder.build(
            router: self
            // ...
        )
        baseRouter.presentViewController(viewController)
    }
}

extension SubscriptionPlansRouter: SubscriptionPlansRouting {
    func routeBack() {
        baseRouter.dismissViewController()
    }
}
```

### Флоу с несколькими сценами

Когда флоу содержит 2+ экранов, есть два варианта организации роутеров.

#### Вариант 1: Один роутер на весь флоу

```swift
protocol SubscriptionPlansRouting {
    func routeToDetails(of plan: SubscriptionPlan)
    func routeToPaymentSetup()
    func routeBackFromSubscriptionPlans()
}

extension SubscriptionPlansRouter: SubscriptionPlansRouting {
    func routeToDetails(of plan: SubscriptionPlan) {
        let viewController = PlanDetailsSceneBuilder.build(
            router: self,  // <- передали в качестве роутера self
            plan: plan
        )
        baseRouter.pushViewController(viewController)
    }
    
    func routeToPaymentSetup() {
        let viewController = PaymentSetupSceneBuilder.build(
            router: self  // <- передали в качестве роутера self
        )
        baseRouter.presentViewController(viewController)
    }
    
    func routeBackFromSubscriptionPlans() {
        baseRouter.dismissViewController()
    }
}

extension SubscriptionPlansRouter: PlanDetailsRouting {
    func routeBackFromPlanDetails() {
        baseRouter.popViewController()
    }
}

extension SubscriptionPlansRouter: PaymentSetupRouting {
    func routeBackFromPaymentSetup() {
        baseRouter.dismissViewController()
    }
}
```

> ⚠️ Для избежания коллизий имён для методов `back` добавляются суффиксы, указывающие откуда происходит переход.

#### Вариант 2: Отдельный роутер для вложенного флоу

```swift
final class PlanDetailsRouter {
    let baseRouter: BaseRouter
    var completionHandler: (() -> Void)?
    
    init(baseRouter: BaseRouter) {
        self.baseRouter = baseRouter
    }
    
    func routeToPlanDetails(plan: SubscriptionPlan) {
        let viewController = PlanDetailsSceneBuilder.build(
            router: self,
            plan: plan
        )
        baseRouter.pushViewController(viewController)
    }
}

extension PlanDetailsRouter: PlanDetailsRouting {
    func routeBack() {
        baseRouter.popViewController()
        completionHandler?()
    }
}

// Использование во внешнем роутере
extension SubscriptionPlansRouter: SubscriptionPlansRouting {
    func routeToDetails(of plan: SubscriptionPlan) {
        let detailsRouter = PlanDetailsRouter(baseRouter: baseRouter)
        detailsRouter.completionHandler = { [weak self] in
            // реагируем на завершение вложенного флоу
        }
        detailsRouter.routeToPlanDetails(plan: plan)
    }
}
```

**Когда выделять отдельный роутер:**
- Экран переиспользуется в разных флоу
- На экран есть диплинк (способ переиспользования в произвольном контексте)
- Роутер превышает ~500 строк кода

---

## Взаимодействие сцен через навигацию

Есть два способа взаимодействия сцен. Первый — предпочтительный.

> **Важно:** Роутер не участвует в потоках данных между сценами. Максимум — настраивает связи при создании сцены. Избегайте схем, где роутер является делегатом одной сцены и прокидывает данные в другую.

### 1. Взаимодействие через общую модель (Provider)

Обе сцены используют общую доменную модель. Они могут вносить изменения и реагировать на изменения по подписке.

```swift
// Общая модель для двух сцен
final class WorkoutPlanProvider: ObservableObject {
    @Published private(set) var currentPlan: WorkoutPlan?
    
    func updateExercises(_ exercises: [Exercise]) {
        currentPlan?.exercises = exercises
    }
}

// Обе сцены инжектят один и тот же инстанс WorkoutPlanProvider
```

Если редактирование происходит в "черновике" и не должно сразу влиять на основную модель:

```swift
// Модель редактирования, используется только двумя сценами
final class ExerciseEditingController: ObservableObject {
    @Published var draftExercises: [Exercise]
    
    func applyChanges(to provider: WorkoutPlanProvider) {
        provider.updateExercises(draftExercises)
    }
}
```

### 2. Получение результата через замыкание

Подходит для "глупых" view controller'ов, которые не знают как будет использоваться их output.

```swift
final class WorkoutRouter: WorkoutRouting {
    func showExercisePicker(selectionHandler: @escaping (Exercise) -> Void) {
        let pickerRouter = ExercisePickerRouter(
            baseRouter: baseRouter,
            container: container
        )
        let viewController = ExercisePickerSceneBuilder(
            router: pickerRouter,
            selectionHandler: selectionHandler
        ).build()
        baseRouter.pushViewController(viewController)
    }
}
```

---

## Flow & Router

### Flow

Пользовательский флоу — последовательность (дерево) переходов для решения задачи.

```swift
protocol WorkoutFlowProtocol {
    func start()
    func showExercisePicker(for muscleGroup: MuscleGroup)
    func showIntensityDetails(_ intensity: Intensity)
    func showActiveWorkout(workout: Workout)
    func complete(with result: WorkoutResult)
}
```

### Router

Осуществляет навигацию между сценами и их сборку.

```swift
final class WorkoutRouter: WorkoutRouterProtocol {
    private let baseRouter: BaseRouter
    private let container: WorkoutDependencyContainer
    
    init(baseRouter: BaseRouter, container: WorkoutDependencyContainer) {
        self.baseRouter = baseRouter
        self.container = container
    }
    
    func showExercisePicker() {
        let picker = ExercisePickerSceneBuilder(container: container).build()
        baseRouter.pushViewController(picker)
    }
    
    func showActiveWorkout(for workout: WorkoutDraft) {
        let activeWorkout = ActiveWorkoutSceneBuilder(
            container: container,
            workout: workout
        ).build()
        baseRouter.presentViewController(activeWorkout)
    }
}
```

### Схема навигации

```
┌─────────┐        ┌───────────┐        ┌─────────┐        ┌────────┐
│  View   │───────►│ ViewStore │───────►│  Store  │───────►│ Router │
└─────────┘        └───────────┘        └─────────┘        └────────┘
     │                                                          │
     │ ViewEvent                                      Navigation │
     │                                                          │
     ▼                                                          ▼
 User Tap                                              Push/Present
```

---

## Stateless роутеры vs Stateful координаторы

### Проблемы хранения состояния флоу в координаторе

В некоторых проектах координаторы хранят состояние флоу через enum:

```swift
// ❌ Проблемный подход
enum WorkoutFlowState {
    case exerciseList(ExerciseListViewController)
    case activeWorkout(ActiveWorkoutViewController)
    case summary(SummaryViewController)
}
```

**Проблемы такого подхода:**

1. **Плоская структура** — фиксируется только текущий экран, но не стек навигации. Непонятно откуда пришли и куда переходить по кнопке "назад"

2. **Противоречит вычислимой навигации** — при использовании `UINavigationController` мы получаем стек для возврата "из коробки"

3. **Сложность описания вложенных состояний** — для модальных окон приходится использовать `indirect enum`, что позволяет описывать невалидные состояния

### Преимущества stateless роутеров

1. **Не нужно поддерживать иерархию координаторов**
   
   При обработке диплинков можно просто перейти на нужный экран без обновления иерархии координаторов.

2. **Легко начать использовать в любой части приложения**
   
   Не требуется наличие родительского координатора.

3. **Переиспользование открытых экранов**
   
   При переходе по диплинку на уже открытый экран можно найти его и обновить контекст, вместо создания дублирующего экрана.

4. **Сцены становятся более самостоятельными**
   
   Роутер не участвует в формировании экранов, только в переходах.

### Рекомендация

```swift
// ✅ Хороший подход — использовать навигационный граф как источник правды
final class BaseRouter {
    private weak var navigationController: UINavigationController?
    
    func findViewController<T: UIViewController>(ofType type: T.Type) -> T? {
        navigationController?.viewControllers.first { $0 is T } as? T
    }
    
    func popToViewController<T: UIViewController>(ofType type: T.Type, animated: Bool = true) {
        guard let target = findViewController(ofType: type) else { return }
        navigationController?.popToViewController(target, animated: animated)
    }
}
```

---

## Важные правила навигации

### Избегайте retain cycle

Роутер **не должен** держать сильные ссылки на компоненты сцены (включая ViewController), так как сцена держит сильную ссылку на роутер.

```swift
// ❌ Плохо — retain cycle
final class BadRouter {
    var currentViewController: UIViewController?  // Strong reference!
}

// ✅ Хорошо — weak reference или отсутствие ссылки
final class GoodRouter {
    private weak var navigationController: UINavigationController?
}
```

### Делегирование внешних переходов

Для переходов, которые роутер не может произвести самостоятельно:

```swift
// Вариант 1: Замыкание
final class ChildRouter {
    var onFlowComplete: ((Result) -> Void)?
}

// Вариант 2: Протокол внешней навигации
protocol ChildExternalRouting: AnyObject {
    func childDidRequestParentTransition()
}

final class ChildRouter {
    weak var externalRouter: ChildExternalRouting?
}
```

---

## Композиция сцен

> ⚠️ Композиции сцен следует избегать

Для нагруженных сцен допустимо декомпозировать:

```swift
struct ParentState {
    var mainContent: MainContentState
    var childStore: ChildStore  // Дочерний Store в State родителя
}

struct ParentViewState {
    var mainContent: MainContentViewState
    var childViewStore: ViewStore<ChildViewState, ChildViewEvent>  // ViewStore в ViewState
}
```

---

## Взаимодействие между сценами

Состояния шарятся через общие доменные модели (Provider).

```swift
// Общий Provider между сценами
final class UserSessionProvider: ObservableObject {
    @Published private(set) var currentUser: User?
    
    // Используется в ProfileStore, WorkoutStore, SettingsStore...
}
```

---

## Best Practices

### ✅ Рекомендации

1. **Держите Store тонким** — вся логика в Provider
2. **Конвертеры — чистые функции** — без side effects и внешнего состояния
3. **View не знает о Domain** — только ViewState
4. **Один источник правды** — данные в Provider, не дублируйте
5. **Тестируйте конвертеры отдельно** — они pure functions

### ❌ Антипаттерны

1. Бизнес-логика во View
2. Сетевые запросы из View
3. Прямой доступ к Domain моделям из View
4. Mutable state в ViewState
5. Циклические зависимости между Store'ами

---

## Пример полной сцены

### ProfileScene.swift

```swift
// MARK: - State & Event

struct ProfileState: Equatable {
    var user: User?
    var workoutStats: WorkoutStats?
    var isLoading: Bool
    var error: Error?
    
    static func == (lhs: ProfileState, rhs: ProfileState) -> Bool {
        lhs.user == rhs.user && lhs.isLoading == rhs.isLoading
    }
}

enum ProfileEvent {
    case loadProfile
    case editProfile
    case logout
}

// MARK: - ViewState & ViewEvent

struct ProfileViewState: Equatable {
    let avatarURL: URL?
    let userName: String
    let totalWorkouts: String
    let totalCalories: String
    let isLoading: Bool
    let achievementBadges: [BadgeViewModel]
}

enum ProfileViewEvent {
    case viewAppeared
    case editButtonTapped
    case logoutButtonTapped
}
```

### ProfileStore.swift

```swift
final class ProfileStore: ObservableObject {
    @Published private(set) var state = ProfileState(user: nil, workoutStats: nil, isLoading: false, error: nil)
    
    private let userProvider: UserProvider
    private let authService: AuthServiceProtocol
    private weak var router: ProfileRouterProtocol?
    
    init(userProvider: UserProvider, authService: AuthServiceProtocol, router: ProfileRouterProtocol?) {
        self.userProvider = userProvider
        self.authService = authService
        self.router = router
        
        // Подписка на изменения пользователя
        userProvider.$currentUser
            .assign(to: &$state.user)
    }
    
    func send(_ event: ProfileEvent) {
        switch event {
        case .loadProfile:
            state.isLoading = true
            userProvider.refreshProfile()
            
        case .editProfile:
            router?.showProfileEditor()
            
        case .logout:
            authService.logout()
            router?.showAuth()
        }
    }
}
```

---

## Заключение

MAA обеспечивает:
- **Тестируемость** — чистые функции и изоляция слоёв
- **Переиспользуемость** — Provider'ы между сценами
- **Масштабируемость** — чёткое разделение ответственности
- **Поддерживаемость** — предсказуемый поток данных
