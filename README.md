# Roleplay Kernel for SillyTavern

SillyTavern UI-extension и локальный Python-sidecar, реализующие stateful roleplay runtime: модульный planner, renderer, critic, repair, типизированное состояние и подтверждение опасных изменений.

## Требования

- SillyTavern `1.19.0` или новее.
- Python `3.11` или новее.
- Chat Completion API с источником `Custom (OpenAI-compatible)`.
- API endpoint, поддерживающий JSON mode для внутренних planner/extractor/critic вызовов.
- Для режима профиля ST должен быть доступен локальному sidecar через тот же origin, а профиль — иметь сохранённый secret reference в `secrets.json`; режим пользовательских аккаунтов ST с отдельным login не поддерживается.

## Установка

1. Скопируйте `manifest.json`, `index.js`, `style.css`, `settings.html` и `i18n/` в `public/scripts/extensions/third-party/RoleplayKernel/`. Установщик sidecar запускайте из исходного checkout этого репозитория.
2. Для ручного fallback upstream задайте API key в той же PowerShell-сессии (для выбранного профиля ST это не требуется):

```powershell
$env:OPENAI_API_KEY = "..."
```

3. Запустите PowerShell-установщик:

```powershell
.\scripts\install-sidecar.ps1 -SillyTavernPath "C:\SillyTavern" -UpstreamBaseUrl "https://api.openai.com/v1" -UpstreamModel "gpt-4o-mini" -ApiKeyEnv "OPENAI_API_KEY"
```

4. Для последующих запусков используйте созданный установщиком launcher:

```powershell
& "C:\SillyTavern\data\roleplay-kernel\Start-RoleplayKernel.ps1"
```

5. В настройках Roleplay Kernel вставьте выведенный `Integration key` и нажмите `Активировать`.
6. Перезапустите SillyTavern, если UI-extension не появился в списке.

Установщик не включает `enableServerPlugins`: Node server plugin не требуется. Повторный запуск сохраняет существующий `config.json` и integration key, а запуск из уже установленного UI-extension не копирует UI-файлы поверх самих себя.

## Ручная конфигурация

Скопируйте `config.example.json`, замените upstream URL/model и integration key, затем запустите:

```powershell
python -m pip install -e .
python -m roleplay_kernel.sidecar --config .\config.json
```

`config.json` и каталог `state` не должны попадать в Git. API key читается из переменной окружения, указанной в `upstream_api_key_env`.

## Использование

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
- Sampler-level control отсутствует; следующий adapter предназначен для `vLLM` или `llama.cpp`.
- Sidecar process по умолчанию разрешён только на loopback; для удалённого доступа нужен отдельный TLS reverse proxy.
- Плагин совместим с release line SillyTavern `1.19.x`; `minimum_client_version` зафиксирован в manifest.

## Безопасность

Sidecar слушает `127.0.0.1` по умолчанию, но integration key обязателен для generation и control endpoints даже на loopback. Не публикуйте sidecar в интернет и не храните upstream API key в `config.json`.
