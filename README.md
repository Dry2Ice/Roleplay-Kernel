# Roleplay Kernel for SillyTavern

SillyTavern UI-extension и локальный Python-sidecar, реализующие stateful roleplay runtime: модульный planner, renderer, critic, repair, типизированное состояние и подтверждение опасных изменений.

## Требования

- SillyTavern `1.19.0` или новее.
- Python `3.11` или новее.
- Chat Completion API с источником `Custom (OpenAI-compatible)`.
- API endpoint, поддерживающий JSON mode для внутренних planner/extractor/critic вызовов.
- Для режима профиля ST должен быть доступен локальному sidecar через тот же origin, а профиль — иметь сохранённый secret reference в `secrets.json`; режим пользовательских аккаунтов ST с отдельным login не поддерживается.

## Установка

1. Запустите установщик из исходного checkout:

```powershell
.\scripts\install-sidecar.ps1 -SillyTavernPath "C:\SillyTavern" -Port 8787 -NoStart
```

Установщик копирует UI-extension в `public/scripts/extensions/third-party/RoleplayKernel`, server plugin — в `plugins/roleplay-kernel`, создаёт venv и включает `enableServerPlugins` в `config.yaml`.

2. Перезапустите SillyTavern.
3. Откройте настройки Roleplay Kernel, выберите профиль ST и нажмите `Запустить runtime`. Кнопка сама создаёт конфигурацию, выбирает свободный порт, запускает Python скрыто и передаёт в UI `integration_key`.
4. После этого `Активировать` можно вызывать автоматически из кнопки запуска; консоль и отдельный запуск sidecar не нужны.

Установка только через `Extensions → Install extension` копирует браузерную часть и не может установить server plugin; для полной версии нужен один запуск installer из PowerShell.

## Ручная конфигурация

Скопируйте `config.example.json`, замените upstream URL/model и integration key, затем запустите:

```powershell
python -m pip install -e .
python -m roleplay_kernel.sidecar --config .\config.json
```

`config.json` и каталог `state` не должны попадать в Git. API key читается из переменной окружения, указанной в `upstream_api_key_env`.

## Использование

- Интерфейс аддона и все системные сообщения — на английском; выбор языка в UI отсутствует, язык ответа фиксирован на `English`.
- Язык ответа задаётся один раз в `GenerationEnvelope.language` и синхронизируется в состояние сессии на каждом ходу, поэтому переключение настроек больше не оставляет старую сессию на прежнем языке.
- Render-запрос содержит явное требование языка в system и user сообщении (`OUTPUT_LANGUAGE_DATA`), а не только код `en`, который модели часто игнорируют.
- Если модель всё же ответила на русском при запрошенном английском, детерминированный валидатор создаёт finding `response_language_mismatch`, он виден в панели и запускает repair в режиме `strict`.

- Кнопка `Запустить runtime` запускает sidecar через ST Server Plugin без отдельного окна/консоли, автоматически подставляет URL и integration key.
- В настройках Roleplay Kernel выберите режим: `Balanced` (полный pipeline), `Fast` (render + state extraction) или `Lite` (один render-вызов для ограниченных провайдеров).
- Прогресс отображается в панели Kernel в реальном времени: подпись этапа, счётчик `выполнено/всего` и проценты. Интервал опроса — 1 секунда, control-статус не блокируется выполняющимся запросом.
- Пауза между внутренними запросами задаётся отдельно в секундах и отсчитывается после начала получения ответа предыдущего запроса.
- В настройках Roleplay Kernel выберите сохранённый Chat Completion профиль ST в поле `ST connection profile`.
- Sidecar получает из профиля `source`, URL, model и `secret-id`, затем сам обращается к ST backend через локальный CSRF-сеанс; сырой API key не передаётся и не читается extension.
- Если профиль не выбран, используется ручной `upstream_*` из `config.json`.
- Кнопка `Активировать` переводит Chat Completion source в Custom и включает маршрутизацию.
- Статус ядра виден всегда: цветной индикатор над панелью расширений и запись в
  wand-меню (иконка кубов) с прогрессом, кнопками Start/Stop и переходом в настройки.
- Если runtime недоступен, показывается баннер с кнопкой `Restart runtime`; маршрутизация
  при этом отключается, а подключение ST восстанавливается.
- Extension никогда не записывает integration key в постоянные настройки подключения ST: ключ передаётся только в per-request заголовке. Если runtime недоступен три опроса подряд, маршрутизация отключается автоматически, а подключение ST восстанавливается.
- Выключение расширения в списке extensions вызывает hook `disable`, который восстанавливает исходные source, URL, model и заголовки, поэтому генерация продолжает работать даже без панели.
- UI-extension передаёт sidecar отдельный transcript snapshot и финальный prompt ST; character card, World Info, Prompt Manager, history, swipes и chat metadata обрабатываются штатно.
- Kernel использует отдельный session id для каждого чата.
- High-impact или низкоуверенные изменения появляются как `Pending` и требуют `Подтвердить delta` или `Отклонить delta`.
- `Сбросить сессию` удаляет только состояние Kernel; история SillyTavern не изменяется.
- Если чат отредактирован, обрезан, свайпнут или пополнен вне ядра, расхождение больше не блокирует генерацию: недостающие реплики дописываются в историю ядра, а в панели появляется уведомление о синхронизации. Реплики, вытесненные чатом, помечаются как `superseded` и больше не попадают в промпты, хотя остаются в ledger.
- Каждая запись состояния помнит реплику, которая её создала (`provenance`). Если исходная реплика была переписана, ядро сообщает finding `state_orphaned_by_rewrite` вместо молчаливого расхождения.
- Промпты получают `TIME_PASSAGE_DATA` с реальным временем, прошедшим с прошлого хода, поэтому время в сюжете не выдумывается моделью.

## Ограничения MVP

- Маршрутизируются `normal`, `regenerate` и `swipe`.
- `continue`, `impersonate`, quiet generations и group chats не маршрутизируются.
- В выпадающем списке поддерживаются профили Chat Completion с источниками `OpenAI` и `Custom`; из выбранного профиля используются connection source, URL, model и secret reference, а profile proxy и sampling preset пока не переносятся.
- Стандартный лимит render-ответа — `8192` токенов, внутренних planner/critic вызовов — `4096`; значение ST `max_tokens` не может уменьшить этот безопасный минимум.
- Таймаут одного upstream-запроса — `600` секунд, чтобы reasoning-модели успевали завершить длинный ответ.
- Sampler-level control отсутствует; следующий adapter предназначен для `vLLM` или `llama.cpp`.
- Sidecar process по умолчанию разрешён только на loopback; для удалённого доступа нужен отдельный TLS reverse proxy.
- UI-extension сам по себе не может установить server plugin; полная версия требует один запуск installer, после чего запуск выполняется кнопкой.
- Endpoint `POST /api/plugins/roleplay-kernel/restart` перезапускает runtime принудительно.
- Плагин совместим с release line SillyTavern `1.19.x`; `minimum_client_version` зафиксирован в manifest.

## Безопасность

- Sidecar слушает `127.0.0.1` по умолчанию, но integration key обязателен для generation и control endpoints даже на loopback. Не публикуйте sidecar в интернет и не храните upstream API key в `config.json`.
- Sidecar отклоняет запросы с не-loopback заголовком `Host` (защита от DNS-rebinding через браузер) и блокирует адрес после 10 неудачных авторизаций в минуту.
- Server plugin перезапускает упавший runtime автоматически: каждые 5 секунд проверяется `/health`, после трёх неудач подряд процесс перезапускается, а счётчик перезапусков виден в `/api/plugins/roleplay-kernel/status`.
- Extension сверяет версию sidecar по `/health` и показывает предупреждение, если версии расходятся.

## Разработка

```
pip install -e '.[dev]'
npm install

ruff check .
mypy src tests
python -m unittest discover -s tests -q
npm run lint      # ESLint, sourceType: module
npm test          # jsdom-тесты панели и регрессий
```

План работ и контрольные точки поставки — в `ROADMAP.md`. Скрипт `scripts/bump_version.py`
поднимает версию в `pyproject.toml`, `manifest.json` и `sidecar.py` одновременно.

## Лицензия

MIT. См. `LICENSE`.
