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

- Кнопка `Запустить runtime` запускает sidecar через ST Server Plugin без отдельного окна/консоли, автоматически подставляет URL и integration key.
- В настройках Roleplay Kernel выберите режим: `Balanced` (полный pipeline), `Fast` (render + state extraction) или `Lite` (один render-вызов для ограниченных провайдеров).
- В настройках Roleplay Kernel выберите сохранённый Chat Completion профиль ST в поле `ST connection profile`.
- Sidecar получает из профиля `source`, URL, model и `secret-id`, затем сам обращается к ST backend через локальный CSRF-сеанс; сырой API key не передаётся и не читается extension.
- Если профиль не выбран, используется ручной `upstream_*` из `config.json`.
- Кнопка `Активировать` переводит Chat Completion source в Custom и включает маршрутизацию.
- UI-extension передаёт sidecar отдельный transcript snapshot и финальный prompt ST; character card, World Info, Prompt Manager, history, swipes и chat metadata обрабатываются штатно.
- Kernel использует отдельный session id для каждого чата.
- High-impact или низкоуверенные изменения появляются как `Pending` и требуют `Подтвердить delta` или `Отклонить delta`.
- `Сбросить сессию` удаляет только состояние Kernel; история SillyTavern не изменяется.

## Ограничения MVP

- Маршрутизируются `normal`, `regenerate` и `swipe`.
- `continue`, `impersonate`, quiet generations и group chats не маршрутизируются.
- В выпадающем списке поддерживаются профили Chat Completion с источниками `OpenAI` и `Custom`; из выбранного профиля используются connection source, URL, model и secret reference, а profile proxy и sampling preset пока не переносятся.
- Стандартный лимит render-ответа — `8192` токенов, внутренних planner/critic вызовов — `4096`; значение ST `max_tokens` не может уменьшить этот безопасный минимум.
- Таймаут одного upstream-запроса — `300` секунд, чтобы reasoning-модели успевали завершить длинный ответ.
- Sampler-level control отсутствует; следующий adapter предназначен для `vLLM` или `llama.cpp`.
- Sidecar process по умолчанию разрешён только на loopback; для удалённого доступа нужен отдельный TLS reverse proxy.
- UI-extension сам по себе не может установить server plugin; полная версия требует один запуск installer, после чего запуск выполняется кнопкой.
- Плагин совместим с release line SillyTavern `1.19.x`; `minimum_client_version` зафиксирован в manifest.

## Безопасность

Sidecar слушает `127.0.0.1` по умолчанию, но integration key обязателен для generation и control endpoints даже на loopback. Не публикуйте sidecar в интернет и не храните upstream API key в `config.json`.

## Лицензия

MIT. См. `LICENSE`.
