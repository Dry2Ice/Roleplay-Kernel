# Roleplay Kernel — roadmap

Статус версии: см. `manifest.json` и `pyproject.toml` (`version`).

Правило поставки: каждая фаза завершается прогоном всех проверок, bump версии, commit и
push в GitHub. Установка в локальный SillyTavern делается только на контрольных точках,
перечисленных в разделе «Контрольные точки», а не после каждого промежуточного коммита.

Проверки, обязательные для каждой фазы:

```
ruff check .
mypy src tests
python -m unittest discover -s tests -q
node --check (копия index.js как .mjs)
node tests/js/panel-harness.mjs      # рендер панели в jsdom
node --test tests/js/*.test.mjs      # JS-регрессии
```

---

## v0.1.1 — Базовый уровень надёжности и безопасности  ✅ завершена

Commit `39ea16e`. Установлена в ST, проверена, отправлена в GitHub.

- [x] Валидация `Host` в sidecar (защита от DNS-rebinding через браузер).
- [x] Throttle неудачных авторизаций на sidecar.
- [x] Heartbeat и авто-рестарт sidecar в server plugin.
- [x] Endpoint `POST /api/plugins/roleplay-kernel/restart`.
- [x] Version handshake: extension сверяет версию sidecar и предупреждает при расхождении.
- [x] Регрессия-тест «integration key никогда не попадает в настройки подключения ST».
- [x] ESLint (`sourceType: module`) — ловит синтаксис, который пропускает `node --check`.
- [x] jsdom-harness перенесён в репозиторий как настоящий тест (`tests/js`).
- [x] CI workflow: ruff, mypy, unittest, ESM-парс, harness, JS-тесты.
- [x] Скрипт `scripts/bump_version.py` для единого подъёма версии.

Проверено: 90 python-тестов, 6 JS-тестов, mypy, ruff, eslint, ESM-парс — зелёные.

## v0.2.0 — Streaming, отмена, бюджеты по этапам  ✅ завершена

- [x] `DeltaSink` в протоколе `ChatProvider` и обоих провайдерах.
- [x] Разбор SSE в провайдерах (`_read_sse_stream`) с лимитом размера и `finish_reason`.
- [x] `on_delta` в `Engine.advance` только для render-этапа.
- [x] `_SseWriter`: стриминг в ST, детект разрыва соединения, ошибки как SSE-событие.
- [x] `ClientGoneError` + `_aborting_sink`: Stop в ST отменяет ход, а не жжёт квоту.
- [x] Бюджет хода `turn_budget_seconds` и `post_render_grace_seconds` в конфиге.
- [x] Деградация без потери ответа: `planner_skipped`, `extract_skipped`,
      `critic_skipped`, `state_frozen` вместо потери поста.
- [x] `TurnMetrics`: provider_calls, first_token, render_seconds, total_seconds.
- [x] Автоматический downgrade в `Lite` по истории 429 + индикация в панели.
- [x] Метрики в панели: calls, first token, total.
- [x] `scripts/bump_version.py` переписан: ищет версию регуляркой, идемпотентен.
- [x] Версия 0.2.0 синхронно в `pyproject.toml`, `sidecar.py`, `index.js`, `manifest.json`.

Проверено: 94 python-теста, 6 JS-тестов, mypy, ruff, eslint, ESM-парс — зелёные.
Контрольная точка: установка в ST, замер времени до первого токена.

## v0.3.0 — Наблюдаемость  ✅ завершена

- [x] `diagnostics` в control API: версии, конфиг, upstream, метрики, лимиты.
      Секреты не попадают в отчёт — только флаги присутствия.
- [x] `self_test` в control API: state_dir, integration_key, upstream_profile,
      проба провайдера одним запросом в 16 токенов.
- [x] Кнопка `Run self-test` со списком проверок OK/FAIL в панели.
- [x] Кнопка `Copy diagnostics`: версии, настройки, статус runtime, отчёт sidecar,
      последние тосты — в буфер обмена.
- [x] **Индикатор вне панели**: цветной статус-пилюля над панелью расширений и запись
      в wand-меню с прогрессом и кнопками Start/Stop/Settings (`wand.html`).
- [x] **Баннер недоступности** с кнопкой `Restart runtime` и закрытием.
- [x] Защита от гонки при монтировании (`mountInFlight`), иначе дублировался wand-вход.
- [x] **Найден и исправлен баг**: `renderUi` пересоздавал панель, а `bindUi`
      выходил по `uiReady`, из-за чего новые кнопки оставались без обработчиков.
      Регрессия закрыта тестом `buttons stay wired after repeated activation events`.
- [x] Тесты на индикатор, wand-меню, баннер и отсутствие дублей.

Проверено: 96 python-тестов, 12 JS-тестов, mypy, ruff, eslint — зелёные.

## v0.4.0 — Качество состояния  ✅ завершена

- [x] `superseded` у реплики: реплика, вытесненная авторитетным транскриптом ST,
      остаётся в ledger, но исключается из всех промптов.
- [x] `Session.supersede_turns()` + `active_turns()`: аддитивная синхронизация
      больше не засоряет контекст старыми репликами.
- [x] `provenance` в состоянии: каждая запись помнит реплику, которая её создала.
      В промпты не попадает.
- [x] `audit_state_provenance()`: finding `state_orphaned_by_rewrite`, когда запись
      состояния опирается на переписанный кусок чата. Автоудаления нет — только сигнал.
- [x] Продвижение времени: `elapsed_since_last_turn()` и блок `TIME_PASSAGE_DATA`
      в промптах планировщика и рендера.

Проверено: 100 python-тестов, 12 JS-тестов, mypy, ruff, eslint — зелёные.

## v0.5.0 — Качество промптов  ⬜ не начата

- [ ] Извлечение «голоса» персонажа из карточки и закрепление в render-промпте.
- [ ] Точечный repair: критику и ошибочный фрагмент вместо всего поста.
- [ ] Критик получает релевантный срез состояния вместо полного.
- [ ] Structure-aware антиповтор: список уже использованных образов и реплик.

## v0.6.0 — UX подтверждений и прозрачность решений  ⬜ не начата

- [ ] Понятный diff для delta: «Theo lost his key · Accept / Reject».
- [ ] Панель «что решило ядро»: цель сцены, модули, находки критика.
- [ ] История ходов с метриками и повтором конкретного хода.

---

## Отложено сознательно

- Векторный кэш и семантическая дедупликация: низкая отдача для ролёвки, высокая цена.
- Семантический поиск по памяти: зависит от модели и провайдера, отложено до v0.7.
- Полноценный SSE в ST для всех этапов: сначала нужен измеряемый baseline v0.2.0.
