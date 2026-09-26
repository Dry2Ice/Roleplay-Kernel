const MODULE_NAME = 'roleplay_kernel';
const PROTOCOL_VERSION = 1;
const ENVELOPE_PREFIX = '[ROLEPLAY_KERNEL_ENVELOPE_V1]';
const CONTROL_PREFIX = '[ROLEPLAY_KERNEL_CONTROL_V1]';
const CONTROL_MODEL_PREFIX = 'roleplay-kernel-control/';
const RUNTIME_PLUGIN_ID = 'roleplay-kernel';
const EXTENSION_VERSION = '0.2.0';
const SUPPORTED_GENERATIONS = new Set(['normal', 'regenerate', 'swipe']);
const SUPPORTED_PROFILE_SOURCES = new Set(['openai', 'custom']);
const SAMPLING_FIELDS = [
    'temperature',
    'max_tokens',
    'top_p',
    'frequency_penalty',
    'presence_penalty',
    'stop',
    'seed',
    'n',
    'logit_bias',
    'top_k',
    'min_p',
    'repetition_penalty',
];
const DEFAULT_SETTINGS = Object.freeze({
    enabled: false,
    autoRoute: true,
    sidecarUrl: 'http://127.0.0.1:8787/v1',
    integrationKey: '',
    model: 'roleplay-kernel',
    profileId: '',
    mode: 'balanced',
    requestDelaySeconds: 0,
    language: 'en',
    pov: 'third_person_limited',
    tense: 'past',
    previousConnection: null,
});

let settings = null;
let currentGenerationType = 'normal';
let uiReady = false;
const statusRequests = new Map();
let progressTimer = null;
let verifiedSidecarBase = null;
let versionWarningShown = null;
let wandMounted = false;
let mountInFlight = null;
let offlineNotified = false;
const toasts = [];

function noteToast(kind, message) {
    toasts.push({ kind, message: String(message) });
    if (toasts.length > 50) {
        toasts.shift();
    }
    const notify = toastr?.[kind];
    if (typeof notify === 'function') {
        notify.call(toastr, message);
    }
}

function info(message) {
    noteToast('info', message);
}

function success(message) {
    noteToast('success', message);
}

function warning(message) {
    noteToast('warning', message);
}

function failure(message) {
    noteToast('error', message);
}
let consecutiveStatusFailures = 0;

function getContext() {
    return SillyTavern.getContext();
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
}

function getExtensionName() {
    const path = new URL('.', import.meta.url).pathname;
    return path.split('/').filter(Boolean).at(-1) || 'RoleplayKernel';
}

function ensureSettings() {
    const context = getContext();
    context.extensionSettings[MODULE_NAME] ??= structuredClone(DEFAULT_SETTINGS);
    const stored = context.extensionSettings[MODULE_NAME];
    for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
        stored[key] ??= value;
    }
    stored.model = DEFAULT_SETTINGS.model;
    stored.language = DEFAULT_SETTINGS.language;
    settings = stored;
    return settings;
}

function profileUsesProxy(profile) {
    const value = String(profile?.proxy ?? '').trim().toLowerCase();
    return Boolean(value && !['none', 'null', 'undefined'].includes(value));
}

function connectionProfiles() {
    const context = getContext();
    if (context.extensionSettings?.disabledExtensions?.includes('connection-manager')) {
        return [];
    }
    const manager = context.extensionSettings?.connectionManager;
    if (!Array.isArray(manager?.profiles)) {
        return [];
    }
    return manager.profiles.filter(profile => {
        if (!profile || profile.mode !== 'cc' || !profile.id) {
            return false;
        }
        const source = profileSource(profile);
        return Boolean(
            source
            && SUPPORTED_PROFILE_SOURCES.has(source)
            && profile.model
            && !profileUsesProxy(profile)
            && (source !== 'custom' || profile['api-url'])
        );
    });
}

function profileSource(profile) {
    const apiMap = getContext().CONNECT_API_MAP || {};
    const source = apiMap[profile?.api]?.source;
    return typeof source === 'string' && source ? source : null;
}

function selectedConnectionProfile() {
    const manager = getContext().extensionSettings?.connectionManager;
    const profileId = settings?.profileId || manager?.selectedProfile;
    if (!profileId) {
        return null;
    }
    return connectionProfiles().find(
        profile => String(profile.id) === String(profileId),
    ) || null;
}

function profilePayload(profile) {
    const source = profileSource(profile);
    if (!source) {
        return null;
    }
    return {
        profile_id: String(profile.id),
        st_base_url: window.location.origin,
        source,
        api_url: String(profile['api-url'] || ''),
        model: String(profile.model || ''),
        secret_id: String(profile['secret-id'] || ''),
    };
}

function refreshProfileOptions() {
    if (!settings) {
        return;
    }
    const select = document.getElementById('rpk_profile');
    const hint = document.getElementById('rpk_profile_hint');
    if (!select) {
        return;
    }
    const profiles = connectionProfiles();
    const manager = getContext().extensionSettings?.connectionManager;
    if (!settings.profileId && manager?.selectedProfile) {
        const selected = profiles.find(profile => String(profile.id) === String(manager.selectedProfile));
        if (selected) {
            settings.profileId = String(selected.id);
            saveSettings();
        }
    } else if (
        settings.profileId
        && Array.isArray(manager?.profiles)
        && !profiles.some(profile => String(profile.id) === String(settings.profileId))
    ) {
        settings.profileId = '';
        saveSettings();
    }
    select.replaceChildren();
    const directOption = document.createElement('option');
    directOption.value = '';
    directOption.textContent = 'Direct upstream from sidecar config';
    select.append(directOption);
    for (const profile of profiles) {
        const option = document.createElement('option');
        option.value = String(profile.id);
        option.textContent = `${profile.name || profile.id} · ${profileSource(profile)}`;
        select.append(option);
    }
    select.value = settings.profileId;
    if (hint) {
        const selected = selectedConnectionProfile();
        hint.textContent = selected
            ? `Profile: ${selected.name || selected.id}; the API key stays in ST secrets.json.`
            : 'Select a saved ST profile or use the manual sidecar configuration.';
    }
}

function saveSettings() {
    getContext().saveSettingsDebounced();
}

function ensureChatBinding() {
    const context = getContext();
    if (!context.chatMetadata || typeof context.chatMetadata !== 'object') {
        return null;
    }
    context.chatMetadata[MODULE_NAME] ??= {
        schemaVersion: 1,
        sessionId: context.uuidv4(),
        stateVersion: 0,
        pendingCount: 0,
        pendingRequestId: null,
        status: 'idle',
        desynchronized: false,
    };
    return context.chatMetadata[MODULE_NAME];
}

function saveChatBinding() {
    const context = getContext();
    if (context.chatMetadata?.[MODULE_NAME]) {
        context.saveMetadataDebounced();
    }
}

function validateSidecarUrl(value) {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) {
        throw new Error('Sidecar URL must use HTTP or HTTPS');
    }
    if (url.protocol === 'http:') {
        const host = url.hostname.toLowerCase();
        if (!['localhost', '127.0.0.1', '[::1]', '::1'].includes(host)) {
            throw new Error('Plain HTTP is allowed only for localhost');
        }
    }
    if (url.search || url.hash) {
        throw new Error('Sidecar URL must not contain query or fragment');
    }
    const pathname = url.pathname.replace(/\/+$/, '');
    if (!pathname) {
        url.pathname = '/v1';
    } else if (!pathname.endsWith('/v1')) {
        throw new Error('Sidecar URL must end with /v1');
    }
    return url.toString().replace(/\/+$/, '');
}

function isCustomSource() {
    const context = getContext();
    return context.mainApi === 'openai'
        && context.chatCompletionSettings.chat_completion_source === 'custom';
}

function shouldRoute() {
    const context = getContext();
    return Boolean(
        settings?.enabled
        && settings.autoRoute
        && isCustomSource()
        && !context.groupId
    );
}

function transcriptSnapshot() {
    const context = getContext();
    const messages = Array.isArray(context.chat) ? context.chat : [];
    const selected = messages
        .filter(message => (
            !message.is_system
            && !message.extra?.[context.symbols?.ignore]
            && message.extra?.type !== 'narrator'
        ))
        .slice(-200);
    const reversed = [];
    const encoder = new TextEncoder();
    let remainingBytes = 1500000;
    for (let index = selected.length - 1; index >= 0 && remainingBytes > 0; index -= 1) {
        const message = selected[index];
        let content = String(
            typeof context.substituteParams === 'function'
                ? context.substituteParams(message.mes || '')
                : message.mes || '',
        ).slice(0, 64000);
        while (content && encoder.encode(content).length > remainingBytes) {
            content = content.slice(0, -1);
        }
        reversed.push({
            role: message.is_user ? 'user' : 'assistant',
            name: String(message.name || '').slice(0, 200),
            content,
            swipe_id: message.swipe_id == null ? null : String(message.swipe_id).slice(0, 128),
        });
        remainingBytes -= encoder.encode(content).length;
    }
    return reversed.reverse().map((item, index) => ({ index, ...item }));
}

function requestKey() {
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    return Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('');
}

function integrationHeaders(existing = '') {
    const authorization = `Bearer ${settings.integrationKey}`;
    const yaml = SillyTavern.libs?.yaml;
    if (!yaml?.parse || !yaml?.stringify) {
        return `Authorization: ${JSON.stringify(authorization)}`;
    }
    let parsed;
    try {
        parsed = existing ? yaml.parse(String(existing)) : {};
    } catch {
        return `Authorization: ${JSON.stringify(authorization)}`;
    }
    if (Array.isArray(parsed)) {
        const entries = parsed.filter(item => item && typeof item === 'object');
        const withoutAuthorization = entries.filter(item => !Object.keys(item).some(key => /^authorization$/i.test(key)));
        withoutAuthorization.push({ Authorization: authorization });
        return yaml.stringify(withoutAuthorization);
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return `Authorization: ${JSON.stringify(authorization)}`;
    }
    const headers = {};
    for (const [key, value] of Object.entries(parsed)) {
        if (!/^authorization$/i.test(key)) {
            headers[key] = value;
        }
    }
    headers.Authorization = authorization;
    return yaml.stringify(headers);
}

function hasKernelEnvelope(messages) {
    return Array.isArray(messages)
        && messages.some(message => (
            typeof message?.content === 'string'
            && message.content.includes(ENVELOPE_PREFIX)
        ));
}

function onPromptReady(data) {
    if (
        !shouldRoute()
        || data?.dryRun
        || !SUPPORTED_GENERATIONS.has(currentGenerationType)
    ) {
        return;
    }
    const binding = ensureChatBinding();
    if (!binding) {
        return;
    }
    const transcript = transcriptSnapshot();
    const key = requestKey();
    const envelope = {
        protocol: PROTOCOL_VERSION,
        operation: 'generate',
        session_id: binding.sessionId,
        request_key: key,
        transcript,
        generation_type: currentGenerationType,
        mode: settings.mode,
        request_delay_seconds: Number(settings.requestDelaySeconds) || 0,
        language: settings.language,
        pov: settings.pov,
        tense: settings.tense,
    };
    const profile = selectedConnectionProfile();
    if (profile) {
        envelope.upstream_profile = profilePayload(profile);
    }
    if (!Array.isArray(data.chat)) {
        return;
    }
    data.chat.push({
        role: 'system',
        content: ENVELOPE_PREFIX + JSON.stringify(envelope),
    });
}

function attachSampling(request) {
    const message = Array.isArray(request.messages)
        ? request.messages.find(item => (
            typeof item?.content === 'string'
            && item.content.includes(ENVELOPE_PREFIX)
        ))
        : null;
    if (!message) {
        return;
    }
    try {
        const envelope = JSON.parse(message.content.slice(message.content.indexOf(ENVELOPE_PREFIX) + ENVELOPE_PREFIX.length));
        const sampling = {};
        for (const field of SAMPLING_FIELDS) {
            if (Object.prototype.hasOwnProperty.call(request, field) && request[field] != null) {
                sampling[field] = request[field];
            }
        }
        envelope.sampling = sampling;
        message.content = ENVELOPE_PREFIX + JSON.stringify(envelope);
    } catch {
        return;
    }
}

function onSettingsReady(request) {
    const type = String(request.type || currentGenerationType);
    if (
        !settings?.enabled
        || !settings.integrationKey
        || !SUPPORTED_GENERATIONS.has(type)
        || !hasKernelEnvelope(request.messages)
    ) {
        return;
    }
    try {
        attachSampling(request);
        request.chat_completion_source = 'custom';
        request.custom_url = validateSidecarUrl(settings.sidecarUrl);
        request.model = DEFAULT_SETTINGS.model;
        request.custom_include_body = '';
        request.custom_exclude_body = '';
        request.custom_include_headers = integrationHeaders(request.custom_include_headers);
        request.custom_prompt_post_processing = '';
    } catch (error) {
        request.custom_url = 'http://127.0.0.1:1/v1';
        failure(`Roleplay Kernel: ${String(error.message || error)}`);
    }
}

async function runtimeRequest(action) {
    const context = getContext();
    const method = action === 'status' ? 'GET' : 'POST';
    let response;
    try {
        response = await fetch(`/api/plugins/${RUNTIME_PLUGIN_ID}/${action}`, {
            method,
            headers: context.getRequestHeaders(),
        });
    } catch (error) {
        throw new Error(`Server plugin unavailable: ${String(error.message || error)}`);
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.error) {
        throw new Error(data?.error || `Server plugin returned HTTP ${response.status}`);
    }
    return data;
}

function renderRuntimeStatus(data) {
    const label = document.getElementById('rpk_runtime_status');
    const launch = document.getElementById('rpk_launch');
    const stop = document.getElementById('rpk_stop');
    const running = Boolean(data?.running);
    if (label) {
        label.textContent = running
            ? `Runtime: running${data.port ? ` (port ${data.port})` : ''}`
            : 'Runtime: stopped';
    }
    if (launch) {
        launch.disabled = running;
    }
    if (stop) {
        stop.disabled = !running;
    }
}

async function refreshRuntimeStatus({ silent = true } = {}) {
    try {
        const data = await runtimeRequest('status');
        renderRuntimeStatus(data);
        return data;
    } catch (error) {
        renderRuntimeStatus({ running: false });
        if (!silent) {
            failure(String(error.message || error));
        }
        return null;
    }
}

async function launchRuntime() {
    try {
        info('Starting Roleplay Kernel runtime...');
        const data = await runtimeRequest('launch');
        if (data.sidecar_url) {
            settings.sidecarUrl = data.sidecar_url;
        }
        if (data.integration_key) {
            settings.integrationKey = data.integration_key;
        }
        verifiedSidecarBase = null;
        saveSettings();
        renderRuntimeStatus(data);
        await activateRouting();
        success('Runtime started');
    } catch (error) {
        failure(String(error.message || error));
    }
}

async function stopRuntime() {
    try {
        if (settings.enabled) {
            disableRouting();
        }
        const data = await runtimeRequest('stop');
        renderRuntimeStatus(data);
        verifiedSidecarBase = null;
        info('Runtime stopped');
    } catch (error) {
        failure(String(error.message || error));
    }
}

function checkVersionCompatibility(health) {
    const reported = typeof health?.version === 'string' ? health.version : null;
    if (!reported || reported === EXTENSION_VERSION) {
        return;
    }
    const message = `Roleplay Kernel: extension ${EXTENSION_VERSION} but sidecar ${reported}. `
        + 'Re-run the installer to update the sidecar.';
    const banner = document.getElementById('rpk_version');
    if (banner) {
        banner.textContent = `v${EXTENSION_VERSION} · sidecar v${reported}`;
        banner.dataset.state = 'mismatch';
    }
    if (versionWarningShown === message) {
        return;
    }
    versionWarningShown = message;
    console.warn(message);
    warning(message);
}

async function assertSidecarIdentity(force = false) {
    const configuredUrl = validateSidecarUrl(settings.sidecarUrl);
    const baseUrl = configuredUrl.replace(/\/v1$/, '');
    if (!force && verifiedSidecarBase === baseUrl) {
        return configuredUrl;
    }
    let response;
    try {
        response = await fetch(`${baseUrl}/health`, {
            method: 'GET',
            headers: { Accept: 'application/json' },
            cache: 'no-store',
        });
    } catch (error) {
        verifiedSidecarBase = null;
        throw new Error(`Sidecar is not reachable at ${baseUrl}: ${String(error.message || error)}`);
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.service !== 'roleplay-kernel-sidecar') {
        verifiedSidecarBase = null;
        throw new Error(
            `Port ${baseUrl} is not a Roleplay Kernel sidecar `
            + `(service=${data?.service || 'no response'})`,
        );
    }
    checkVersionCompatibility(data);
    verifiedSidecarBase = baseUrl;
    return configuredUrl;
}

async function controlTunnel(action, payload = {}) {
    const sidecarUrl = await assertSidecarIdentity();
    const response = await fetch(`${sidecarUrl}/chat/completions`, {
        method: 'POST',
        headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
            Authorization: `Bearer ${settings.integrationKey}`,
        },
        body: JSON.stringify({
            model: `${CONTROL_MODEL_PREFIX}${action}`,
            messages: [{
                role: 'system',
                content: CONTROL_PREFIX + JSON.stringify(payload),
            }],
            temperature: 0,
            max_tokens: 4096,
            stream: false,
        }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.error) {
        throw new Error(data?.error?.message || `Sidecar request failed: ${response.status}`);
    }
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content !== 'string') {
        throw new Error('Sidecar returned an invalid control response');
    }
    return JSON.parse(content);
}

function renderStatus(status) {
    const state = document.getElementById('rpk_status');
    const version = document.getElementById('rpk_version');
    const details = document.getElementById('rpk_details');
    const approve = document.getElementById('rpk_approve');
    const reject = document.getElementById('rpk_reject');
    if (!state || !version || !details || !approve || !reject) {
        return;
    }
    if (!status) {
        state.textContent = 'Not connected';
        version.textContent = '';
        details.textContent = '';
        approve.disabled = true;
        reject.disabled = true;
        renderHeaderStatus('not connected', 'offline');
        return;
    }
    const running = status.status === 'running';
    state.textContent = running
        ? 'Request in progress'
        : (status.exists ? 'Connected' : 'No session yet');
    version.textContent = running
        ? `v${status.version || '—'} · state ${status.state_version > 0 ? status.state_version : '—'}`
        : `v${status.version || '—'} · state ${status.state_version || 0}`;
    if (versionWarningShown === null) {
        delete version.dataset.state;
    }
    const findings = Array.isArray(status.findings) ? status.findings : [];
    const hard = findings.filter(item => item?.severity === 'hard').length;
    details.textContent = [
        status.transcript_matches === false ? 'Transcript out of sync' : null,
        status.economy_mode ? 'Economy mode (rate limited upstream)' : null,
        `Mode: ${status.status || 'idle'}`,
        `Modules: ${Array.isArray(status.active_modules) ? status.active_modules.length : 0}`,
        `Critic errors: ${hard}`,
        `Pending: ${status.pending_count || 0}`,
    ].filter(Boolean).join(' · ');
    approve.disabled = !status.pending_request_id;
    reject.disabled = !status.pending_request_id;
    if (!running && status.status !== 'error') {
        const pending = status.pending_count || 0;
        renderHeaderStatus(
            pending > 0 ? `awaiting confirmation: ${pending}` : 'ready',
            'ok',
        );
    }
}

function formatSeconds(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        return '—';
    }
    return value < 10 ? `${value.toFixed(1)}s` : `${Math.round(value)}s`;
}

function renderMetrics(status) {
    const box = document.getElementById('rpk_metrics');
    const calls = document.getElementById('rpk_metrics_calls');
    const first = document.getElementById('rpk_metrics_first');
    const total = document.getElementById('rpk_metrics_total');
    const metrics = status?.metrics;
    if (!box || !metrics || typeof metrics !== 'object') {
        if (box) {
            box.hidden = true;
        }
        return;
    }
    box.hidden = false;
    if (calls) {
        calls.textContent = `calls ${metrics.provider_calls ?? 0}`;
    }
    if (first) {
        first.textContent = `first token ${formatSeconds(metrics.first_token_seconds)}`;
    }
    if (total) {
        total.textContent = `total ${formatSeconds(metrics.total_seconds)}`;
    }
}

function kernelState(status) {
    if (!settings?.enabled) {
        return { state: 'offline', text: 'Kernel is off' };
    }
    if (status?.status === 'running') {
        const completed = Number(status?.progress?.completed) || 0;
        const total = Number(status?.progress?.total) || 0;
        return {
            state: 'running',
            text: total > 0 ? `Working ${completed}/${total}` : 'Working',
        };
    }
    if (status?.status === 'offline' || !status) {
        return { state: 'error', text: 'Sidecar unreachable' };
    }
    const pending = Number(status.pending_count) || 0;
    if (pending > 0) {
        return { state: 'ok', text: `Confirming ${pending} change(s)` };
    }
    return { state: 'ok', text: status.economy_mode ? 'Economy mode' : 'Ready' };
}

function renderOutsideIndicator(status) {
    const { state, text } = kernelState(status);
    const pill = document.querySelector('.rpk-status-pill');
    if (pill) {
        pill.hidden = false;
        pill.dataset.state = state;
        const dot = pill.querySelector('.rpk-pill-dot');
        if (dot) {
            dot.dataset.state = state;
        }
        const label = pill.querySelector('.rpk-pill-text');
        if (label) {
            label.textContent = 'Roleplay Kernel';
        }
        const detail = pill.querySelector('.rpk-pill-detail');
        if (detail) {
            detail.textContent = text;
        }
    }
    const dot = document.querySelector('.rpk-wand-entry .rpk-pill-dot');
    if (dot) {
        dot.dataset.state = state;
    }
    const wandDetail = document.querySelector('.rpk-wand-entry [data-role="detail"]');
    if (wandDetail) {
        wandDetail.textContent = text;
    }
    const bar = document.querySelector('.rpk-wand-entry [data-role="bar"]');
    if (bar) {
        const total = Number(status?.progress?.total) || 0;
        const completed = Number(status?.progress?.completed) || 0;
        bar.style.width = total > 0 ? `${Math.min(100, Math.round((completed / total) * 100))}%` : '0';
    }
    showBanner(state === 'error');
}

function showBanner(visible) {
    const banner = document.querySelector('.rpk-banner');
    if (!banner) {
        return;
    }
    if (visible && !offlineNotified) {
        offlineNotified = true;
        failure('Roleplay Kernel is unavailable — the ST connection was restored');
    }
    if (!visible) {
        offlineNotified = false;
    }
    banner.hidden = !visible;
}

function mountStatusIndicator() {
    // APP_READY and the initial call can race; mounting twice would leave a
    // duplicate wand entry with live buttons.
    if (mountInFlight) {
        return mountInFlight;
    }
    mountInFlight = mountStatusIndicatorInner().finally(() => {
        mountInFlight = null;
    });
    return mountInFlight;
}

async function mountStatusIndicatorInner() {
    if (!document.querySelector('.rpk-status-pill')) {        const host = document.getElementById('rm_extensions_block')
            ?? document.querySelector('#extensions_settings2')
            ?? document.body;
        const template = await getContext().renderExtensionTemplateAsync(
            `third-party/${getExtensionName()}`,
            'settings',
        );
        const parser = new DOMParser();
        const parsed = parser.parseFromString(template, 'text/html');
        for (const selector of ['.rpk-status-pill', '.rpk-banner']) {
            const node = parsed.querySelector(selector);
            if (node) {
                host.prepend(node);
            }
        }
        const restart = document.getElementById('rpk_banner_restart');
        restart?.addEventListener('click', () => void launchRuntime());
        document.getElementById('rpk_banner_dismiss')?.addEventListener('click', () => {
            const banner = document.querySelector('.rpk-banner');
            if (banner) {
                banner.hidden = true;
            }
        });
    }
    if (wandMounted) {
        return;
    }
    const menu = document.getElementById('extensionsMenu');
    if (!menu) {
        return;
    }
    const template = await getContext().renderExtensionTemplateAsync(
        `third-party/${getExtensionName()}`,
        'wand',
    );
    const holder = document.createElement('div');
    holder.id = 'roleplay_kernel_wand_container';
    holder.className = 'extension_container';
    holder.innerHTML = template;
    menu.prepend(holder);
    holder.querySelector('[data-action="launch"]')?.addEventListener('click', () => void launchRuntime());
    holder.querySelector('[data-action="stop"]')?.addEventListener('click', () => void stopRuntime());
    holder.querySelector('[data-action="open"]')?.addEventListener('click', () => {
        const root = document.getElementById('extensions_settings-button');
        root?.click();
        window.setTimeout(() => {
            document.getElementById('rpk_panel_root')?.scrollIntoView({ block: 'center' });
        }, 300);
    });
    wandMounted = true;
    renderOutsideIndicator(null);
}

function renderChecks(report) {
    const box = document.getElementById('rpk_checks');
    if (!box) {
        return;
    }
    const checks = Array.isArray(report?.checks) ? report.checks : [];
    if (!checks.length) {
        box.hidden = true;
        box.textContent = '';
        return;
    }
    box.hidden = false;
    box.replaceChildren();
    for (const check of checks) {
        if (!check || typeof check !== 'object') {
            continue;
        }
        const row = document.createElement('div');
        row.className = 'rpk-check';
        row.dataset.ok = String(Boolean(check.ok));
        const mark = document.createElement('strong');
        mark.textContent = check.ok ? 'OK' : 'FAIL';
        const name = document.createElement('span');
        name.textContent = String(check.name ?? '');
        row.append(mark, name);
        if (check.detail) {
            const detail = document.createElement('span');
            detail.className = 'rpk-check-detail';
            detail.textContent = String(check.detail);
            row.append(detail);
        }
        box.append(row);
    }
}

async function runSelfTest() {
    const button = document.getElementById('rpk_self_test');
    if (button) {
        button.disabled = true;
    }
    try {
        const report = await controlTunnel('self_test', {});
        renderChecks(report);
        const failed = (Array.isArray(report?.checks) ? report.checks : [])
            .filter(check => check && !check.ok)
            .map(check => check.name);
        if (failed.length) {
            failure(`Self-test failed: ${failed.join(', ')}`);
        } else {
            success('Self-test passed: every layer is reachable');
        }
    } catch (error) {
        failure(String(error.message || error));
    } finally {
        if (button) {
            button.disabled = false;
        }
    }
}

async function copyDiagnostics() {
    try {
        const report = await controlTunnel('diagnostics', {});
        const status = await safeStatusSnapshot();
        const payload = {
            extension_version: EXTENSION_VERSION,
            st_version: String(getContext().release?.version ?? 'unknown'),
            st_user_agent: navigator.userAgent,
            settings: {
                enabled: Boolean(settings.enabled),
                autoRoute: Boolean(settings.autoRoute),
                mode: settings.mode,
                requestDelaySeconds: settings.requestDelaySeconds,
                profileSelected: Boolean(settings.profileId),
                sidecarUrl: settings.sidecarUrl,
                hasIntegrationKey: Boolean(settings.integrationKey),
            },
            runtime: await runtimeRequest('status').catch(error => ({ error: String(error.message || error) })),
            sidecar: report,
            status,
            recent_toasts: toasts.slice(-5),
        };
        const text = JSON.stringify(payload, null, 2);
        await navigator.clipboard.writeText(text);
        success('Diagnostics copied to the clipboard');
    } catch (error) {
        failure(`Could not copy diagnostics: ${String(error.message || error)}`);
    }
}

async function safeStatusSnapshot() {
    const binding = ensureChatBinding();
    if (!binding) {
        return null;
    }
    try {
        return await controlTunnel('status', { session_id: binding.sessionId });
    } catch (error) {
        return { error: String(error.message || error) };
    }
}

const RPK_PHASE_LABELS = {
    starting: 'Preparing request',
    plan: 'Planning the scene',
    render: 'Generating the reply',
    extract: 'Extracting state changes',
    critic: 'Reviewing the reply',
    repair: 'Repairing the reply',
    fallback: 'Fallback reply without planning',
    done: 'Done',
    idle: 'Kernel idle',
    error: 'Request failed',
};

function renderHeaderStatus(text, state) {
    const header = document.getElementById('rpk_header_status');
    if (!header) {
        return;
    }
    header.textContent = text;
    header.dataset.state = state;
}

function renderProgress(progress) {
    const card = document.getElementById('rpk_progress')?.closest('.rpk-progress-card');
    const bar = document.getElementById('rpk_progress_bar');
    const track = document.getElementById('rpk_progress');
    const text = document.getElementById('rpk_progress_text');
    const label = document.getElementById('rpk_progress_label');
    const count = document.getElementById('rpk_progress_count');
    const active = Boolean(progress?.active);
    const completed = Number(progress?.completed) || 0;
    const total = Number(progress?.total) || 0;
    const percent = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0;
    const phase = typeof progress?.phase === 'string' ? progress.phase : 'idle';
    const phaseLabel = RPK_PHASE_LABELS[phase] || progress?.message || 'Working';
    if (card) {
        card.classList.toggle('is-active', active);
        card.classList.toggle('is-error', phase === 'error');
    }
    if (bar) {
        bar.style.width = `${percent}%`;
    }
    if (track) {
        track.setAttribute('aria-valuenow', String(percent));
        track.setAttribute('aria-valuetext', `${phaseLabel} (${percent}%)`);
    }
    if (label) {
        label.textContent = phaseLabel;
    }
    if (count) {
        count.textContent = total > 0 ? `${completed}/${total} · ${percent}%` : percent > 0 ? `${percent}%` : '—';
    }
    if (text) {
        const detail = typeof progress?.message === 'string' && progress.message
            && progress.message !== phaseLabel
            ? progress.message
            : null;
        text.textContent = detail || (active ? 'Request running, refreshing every second' : 'Ready');
    }
    if (active) {
        renderHeaderStatus(`${phaseLabel} ${completed}/${total || '?'}`, 'running');
    } else if (phase === 'error') {
        renderHeaderStatus('error', 'error');
    }
}

function startProgressPolling() {
    if (progressTimer !== null) {
        return;
    }
    progressTimer = window.setInterval(() => {
        if (settings?.enabled) {
            void refreshStatus({ silent: true, includeTranscript: false });
        }
    }, 1000);
}

function stopProgressPolling() {
    if (progressTimer !== null) {
        window.clearInterval(progressTimer);
        progressTimer = null;
    }
}

async function refreshStatus({ silent = false, includeTranscript = true } = {}) {
    const binding = ensureChatBinding();
    if (!binding) {
        return null;
    }
    const sessionId = binding.sessionId;
    const existing = statusRequests.get(sessionId);
    if (existing) {
        return existing;
    }
    const request = (async () => {
        try {
            const statusPayload = { session_id: sessionId };
            if (includeTranscript) {
                statusPayload.transcript = transcriptSnapshot();
            }
            const status = await controlTunnel('status', statusPayload);
            const current = ensureChatBinding();
            if (!current || current.sessionId !== sessionId) {
                return null;
            }
            current.stateVersion = status.state_version || 0;
            current.pendingCount = status.pending_count || 0;
            current.pendingRequestId = status.pending_request_id || null;
            current.status = status.status || 'idle';
            if (status.transcript_matches === true) {
                current.desynchronized = false;
            }
    saveChatBinding();
            renderStatus(status);
            renderProgress(status.progress);
            renderMetrics(status);
            renderOutsideIndicator(status);
            if (status.transcript_reconciled && !current.reconciledNotified) {
                current.reconciledNotified = true;
                saveChatBinding();
                info(
                    'Roleplay Kernel: the chat history changed outside the kernel, so the kernel was resynced with SillyTavern.',
                );
            }
            consecutiveStatusFailures = 0;
            if (!silent) {
                success('Roleplay Kernel connected');
            }
            return status;
        } catch (error) {
            const current = ensureChatBinding();
            if (!current || current.sessionId !== sessionId) {
                return null;
            }
            current.status = 'offline';
            saveChatBinding();
            renderStatus(null);
            renderProgress(null);
            renderOutsideIndicator({ status: 'offline' });
            consecutiveStatusFailures += 1;
            if (settings?.enabled && consecutiveStatusFailures >= 3) {
                autoReleaseRouting();
            }
            if (!silent) {
                failure(String(error.message || error));
            }
            return null;
        } finally {
            statusRequests.delete(sessionId);
        }
    })();
    statusRequests.set(sessionId, request);
    return request;
}

function autoReleaseRouting() {
    consecutiveStatusFailures = 0;
    if (!settings?.enabled) {
        return;
    }
    settings.enabled = false;
    stopProgressPolling();
    if (!restorePreviousConnection()) {
        stripInjectedIntegrationKey();
    }
    saveSettings();
    renderRuntimeStatus({ running: false });
    warning(
        'Roleplay Kernel: runtime is unreachable, routing disabled. The ST connection was restored.',
    );
}

function restorePreviousConnection() {
    const previous = settings?.previousConnection;
    if (!previous) {
        return false;
    }
    const completion = getContext().chatCompletionSettings;
    completion.chat_completion_source = previous.source;
    completion.custom_url = previous.url;
    completion.custom_model = previous.model;
    completion.custom_prompt_post_processing = previous.postProcessing ?? '';
    completion.custom_include_headers = previous.includeHeaders ?? '';
    saveSettings();
    return true;
}

function stripInjectedIntegrationKey() {
    const completion = getContext().chatCompletionSettings;
    const current = String(completion.custom_include_headers ?? '');
    if (!current) {
        return false;
    }
    if (!settings?.integrationKey || !current.includes(settings.integrationKey)) {
        return false;
    }
    const cleaned = current
        .split('\n')
        .filter(line => !line.includes(settings.integrationKey))
        .join('\n')
        .trim();
    completion.custom_include_headers = cleaned;
    saveSettings();
    return true;
}

function assertRoutingConnection() {
    if (!settings?.enabled || !settings.integrationKey) {
        return;
    }
    try {
        const completion = getContext().chatCompletionSettings;
        completion.chat_completion_source = 'custom';
        completion.custom_url = validateSidecarUrl(settings.sidecarUrl);
        completion.custom_model = DEFAULT_SETTINGS.model;
        completion.custom_prompt_post_processing = '';
        stripInjectedIntegrationKey();
    } catch {
        return;
    }
}

async function activateRouting() {
    const previous = {
        source: getContext().chatCompletionSettings.chat_completion_source,
        url: getContext().chatCompletionSettings.custom_url,
        model: getContext().chatCompletionSettings.custom_model,
        postProcessing: getContext().chatCompletionSettings.custom_prompt_post_processing,
        includeHeaders: getContext().chatCompletionSettings.custom_include_headers,
    };
    try {
        ensureSettings();
        const context = getContext();
        if (context.mainApi !== 'openai') {
            throw new Error('Select a Chat Completion API first');
        }
        if (
            !settings.integrationKey
            || settings.integrationKey.length < 32
            || /[^\u0021-\u007e]/.test(settings.integrationKey)
        ) {
            throw new Error('Provide a valid integration key of at least 32 characters');
        }
        await controlTunnel('health');
        settings.previousConnection ??= previous;
        context.chatCompletionSettings.chat_completion_source = 'custom';
        context.chatCompletionSettings.custom_url = validateSidecarUrl(settings.sidecarUrl);
        context.chatCompletionSettings.custom_model = DEFAULT_SETTINGS.model;
        context.chatCompletionSettings.custom_prompt_post_processing = '';
        settings.enabled = true;
        settings.autoRoute = true;
        startProgressPolling();
        assertRoutingConnection();
        saveSettings();
        const status = await refreshStatus();
        if (!status) {
            throw new Error('Sidecar did not respond');
        }
        success('Roleplay Kernel activated for the Custom OpenAI source');
    } catch (error) {
        settings.enabled = false;
        if (!settings.previousConnection) {
            const completion = getContext().chatCompletionSettings;
            completion.chat_completion_source = previous.source;
            completion.custom_url = previous.url;
            completion.custom_model = previous.model;
            completion.custom_prompt_post_processing = previous.postProcessing;
            completion.custom_include_headers = previous.includeHeaders;
        } else {
            restorePreviousConnection();
        }
        saveSettings();
        failure(String(error.message || error));
    }
}

function disableRouting() {
    settings.enabled = false;
    stopProgressPolling();
    restorePreviousConnection();
    saveSettings();
    renderStatus(null);
    info('Roleplay Kernel routing disabled');
}

async function runControl(action) {
    const binding = ensureChatBinding();
    if (!binding) {
        return;
    }
    if (action === 'reset') {
        const confirmed = await contextPopupConfirm(
            'Reset Roleplay Kernel state?',
            'The SillyTavern chat is kept, but the canonical kernel state is deleted.',
        );
        if (!confirmed) {
            return;
        }
    }
    const sessionId = binding.sessionId;
    try {
        const status = await controlTunnel(action, { session_id: sessionId });
        const current = ensureChatBinding();
        if (!current || current.sessionId !== sessionId) {
            return;
        }
        current.stateVersion = status.state_version || 0;
        current.pendingCount = status.pending_count || 0;
        current.pendingRequestId = status.pending_request_id || null;
        current.status = status.status || (action === 'reset' ? 'idle' : current.status);
        if (action === 'reset') {
            current.desynchronized = false;
        } else if (status.transcript_matches === true) {
            current.desynchronized = false;
        }
        saveChatBinding();
        renderStatus(status);
        renderProgress(status.progress);
        success('State updated');
    } catch (error) {
        failure(String(error.message || error));
    }
}

async function contextPopupConfirm(title, message) {
    const context = getContext();
    return context.Popup.show.confirm(title, message);
}

function bindUi() {
    uiReady = false;    const sidecarUrl = document.getElementById('rpk_sidecar_url');
    const integrationKey = document.getElementById('rpk_integration_key');
    const model = document.getElementById('rpk_model');
    const profile = document.getElementById('rpk_profile');
    const mode = document.getElementById('rpk_mode');
    const requestDelay = document.getElementById('rpk_request_delay');
    const autoRoute = document.getElementById('rpk_auto_route');
    const launch = document.getElementById('rpk_launch');
    const stop = document.getElementById('rpk_stop');
    const activate = document.getElementById('rpk_activate');
    const disable = document.getElementById('rpk_disable');
    const refresh = document.getElementById('rpk_refresh');
    const approve = document.getElementById('rpk_approve');
    const reject = document.getElementById('rpk_reject');
    const reset = document.getElementById('rpk_reset');
    const selfTest = document.getElementById('rpk_self_test');
    const diagnostics = document.getElementById('rpk_diagnostics');
    const required = {
        rpk_sidecar_url: sidecarUrl,
        rpk_integration_key: integrationKey,
        rpk_model: model,
        rpk_profile: profile,
        rpk_mode: mode,
        rpk_request_delay: requestDelay,
        rpk_auto_route: autoRoute,
        rpk_launch: launch,
        rpk_stop: stop,
        rpk_activate: activate,
        rpk_disable: disable,
        rpk_refresh: refresh,
        rpk_approve: approve,
        rpk_reject: reject,
        rpk_reset: reset,
        rpk_self_test: selfTest,
        rpk_diagnostics: diagnostics,
    };
    const missing = Object.entries(required)
        .filter(([, element]) => !element)
        .map(([id]) => id);
    if (missing.length) {
        console.error('[RoleplayKernel] settings panel is missing elements', missing);
        return;
    }
    uiReady = true;
    model.value = DEFAULT_SETTINGS.model;
    model.readOnly = true;
    sidecarUrl.value = settings.sidecarUrl;
    integrationKey.value = settings.integrationKey;
    model.value = settings.model;
    mode.value = settings.mode;
    requestDelay.value = String(settings.requestDelaySeconds ?? 0);
    autoRoute.checked = settings.autoRoute;
    refreshProfileOptions();
    sidecarUrl.addEventListener('change', async () => {
        try {
            settings.sidecarUrl = validateSidecarUrl(sidecarUrl.value);
            sidecarUrl.value = settings.sidecarUrl;
            verifiedSidecarBase = null;
            saveSettings();
            await assertSidecarIdentity(true);
        } catch (error) {
            failure(String(error.message || error));
        }
    });
    integrationKey.addEventListener('change', () => {
        settings.integrationKey = integrationKey.value;
        saveSettings();
    });
    model.addEventListener('change', () => {
        settings.model = DEFAULT_SETTINGS.model;
        model.value = DEFAULT_SETTINGS.model;
        model.readOnly = true;
        saveSettings();
    });
    profile.addEventListener('change', () => {
        settings.profileId = profile.value;
        saveSettings();
        refreshProfileOptions();
    });
    mode.addEventListener('change', () => {
        settings.mode = mode.value;
        saveSettings();
    });
    requestDelay.addEventListener('change', () => {
        const value = Math.max(0, Math.min(600, Number(requestDelay.value) || 0));
        settings.requestDelaySeconds = value;
        requestDelay.value = String(value);
        saveSettings();
    });
    autoRoute.addEventListener('change', () => {
        settings.autoRoute = autoRoute.checked;
        saveSettings();
    });
    launch.addEventListener('click', () => void launchRuntime());
    stop.addEventListener('click', () => void stopRuntime());
    activate.addEventListener('click', () => void activateRouting());
    disable.addEventListener('click', disableRouting);
    refresh.addEventListener('click', () => void refreshStatus());
    approve.addEventListener('click', () => void runControl('commit'));
    reject.addEventListener('click', () => void runControl('reject'));
    reset.addEventListener('click', () => void runControl('reset'));
    selfTest.addEventListener('click', () => {
        void runSelfTest();
    });
    diagnostics.addEventListener('click', () => void copyDiagnostics());
    renderStatus(null);
    renderProgress(null);
    renderRuntimeStatus({ running: false });
    void refreshRuntimeStatus();
    void refreshStatus({ silent: true });
    startProgressPolling();
}

const RPK_PANEL_ID = 'rpk_panel';
const RPK_RENDER_ATTEMPTS = 20;
const RPK_RENDER_RETRY_MS = 250;

function expandPanel(holder) {
    const drawer = holder?.querySelector('.inline-drawer');
    const content = drawer?.querySelector(':scope > .inline-drawer-content');
    const icon = drawer?.querySelector(':scope > .inline-drawer-toggle .inline-drawer-icon');
    if (content) {
        content.style.display = 'block';
    }
    if (icon) {
        icon.classList.remove('down', 'fa-circle-chevron-down');
        icon.classList.add('up', 'fa-circle-chevron-up');
    }
}

function renderPanelFallback(host, error) {
    document.getElementById(RPK_PANEL_ID)?.remove();
    const holder = document.createElement('div');
    holder.id = RPK_PANEL_ID;
    holder.className = 'extension_container';
    holder.innerHTML = `
        <div class="roleplay-kernel-settings">
            <div class="inline-drawer">
                <div class="inline-drawer-toggle inline-drawer-header">
                    <b>Roleplay Kernel</b>
                    <span class="rpk-header-status" data-state="error">panel error</span>
                    <div class="inline-drawer-icon fa-solid fa-circle-chevron-up up"></div>
                </div>
                <div class="inline-drawer-content" style="display: block;">
                    <small class="rpk-details">Could not load the settings panel: ${escapeHtml(String(error?.message || error))}</small>
                    <div class="rpk-actions">
                        <button id="rpk_panel_retry" class="menu-button"><i class="fa-solid fa-rotate"></i><span>Retry</span></button>
                    </div>
                </div>
            </div>
        </div>`;
    host.appendChild(holder);
    holder.querySelector('#rpk_panel_retry')?.addEventListener('click', () => {
        uiReady = false;
        void renderUi();
    });
}

async function renderUi(attempt = 0) {
    const host = document.getElementById('extensions_settings2');
    if (!host) {
        if (attempt < RPK_RENDER_ATTEMPTS) {
            window.setTimeout(() => void renderUi(attempt + 1), RPK_RENDER_RETRY_MS);
        } else {
            console.error('[RoleplayKernel] container extensions_settings2 not found');
        }
        return;
    }
    if (uiReady && document.getElementById(RPK_PANEL_ID)?.isConnected) {
        return;
    }
    try {
        const extensionName = getExtensionName();
        const html = await getContext().renderExtensionTemplateAsync(
            `third-party/${extensionName}`,
            'settings',
        );
        if (typeof html !== 'string' || !html.trim()) {
            throw new Error('ST returned an empty settings template');
        }
        document.getElementById(RPK_PANEL_ID)?.remove();
        const holder = document.createElement('div');
        holder.id = RPK_PANEL_ID;
        holder.className = 'extension_container';
        holder.innerHTML = html;
        host.appendChild(holder);
        expandPanel(holder);
        bindUi();
    } catch (error) {
        console.error('[RoleplayKernel] failed to render settings panel', error);
        if (attempt < 3) {
            window.setTimeout(() => void renderUi(attempt + 1), 500);
            return;
        }
        uiReady = true;
        renderPanelFallback(host, error);
    }
}

function registerEvents() {
    const context = getContext();
    context.eventSource.on(context.eventTypes.GENERATION_STARTED, (type, _params, dryRun) => {
        if (dryRun) {
            return;
        }
        currentGenerationType = String(type || 'normal');
    });
    context.eventSource.on(context.eventTypes.GENERATION_ENDED, () => {
        currentGenerationType = 'normal';
        if (settings?.enabled) {
            void refreshStatus({ silent: true });
        }
    });
    context.eventSource.on(context.eventTypes.CHAT_COMPLETION_PROMPT_READY, onPromptReady);
    context.eventSource.on(context.eventTypes.CHAT_COMPLETION_SETTINGS_READY, onSettingsReady);
    context.eventSource.on(context.eventTypes.CHAT_CHANGED, () => {
        ensureChatBinding();
        renderStatus(null);
        if (settings?.enabled) {
            void refreshStatus({ silent: true });
        }
    });
    context.eventSource.on(context.eventTypes.MESSAGE_RECEIVED, () => {
        if (settings?.enabled) {
            void refreshStatus({ silent: true });
        }
    });
    const markDesynchronized = () => {
        const binding = ensureChatBinding();
        if (binding) {
            binding.desynchronized = true;
            saveChatBinding();
        }
    };
    context.eventSource.on(context.eventTypes.MESSAGE_EDITED, markDesynchronized);
    context.eventSource.on(context.eventTypes.MESSAGE_DELETED, markDesynchronized);
    context.eventSource.on(context.eventTypes.MESSAGE_SWIPED, markDesynchronized);
    const refreshProfiles = () => {
        refreshProfileOptions();
        assertRoutingConnection();
        if (settings?.enabled) {
            void refreshStatus({ silent: true });
        }
    };
    for (const eventName of [
        'CONNECTION_PROFILE_LOADED',
        'CONNECTION_PROFILE_CREATED',
        'CONNECTION_PROFILE_UPDATED',
        'CONNECTION_PROFILE_DELETED',
        'OAI_PRESET_CHANGED_AFTER',
        'CHATCOMPLETION_SOURCE_CHANGED',
        'CHATCOMPLETION_MODEL_CHANGED',
    ]) {
        context.eventSource.on(context.eventTypes[eventName], refreshProfiles);
    }
}

export function onActivate() {
    ensureSettings();
    registerEvents();
    const context = getContext();
    const render = () => void renderUi();
    const mount = () => void mountStatusIndicator().catch(() => {});
    context.eventSource.on(context.eventTypes.APP_READY, () => {
        render();
        mount();
    });
    context.eventSource.on(context.eventTypes.EXTENSION_SETTINGS_LOADED, render);
    render();
    mount();
    startProgressPolling();
}

export function onInstall() {
    ensureSettings();
    saveSettings();
}

export function onUpdate() {
    ensureSettings();
    saveSettings();
}

export function onDisable() {
    try {
        settings = getContext().extensionSettings?.[MODULE_NAME] ?? settings;
        settings.enabled = false;
        stopProgressPolling();
        if (!restorePreviousConnection()) {
            stripInjectedIntegrationKey();
        }
        saveSettings();
        console.info('[RoleplayKernel] disabled: previous ST connection restored');
    } catch (error) {
        console.error('[RoleplayKernel] failed to restore connection on disable', error);
    }
}
