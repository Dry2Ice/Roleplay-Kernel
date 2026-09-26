import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { JSDOM } from 'jsdom';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

/**
 * Minimal SillyTavern runtime used by the extension DOM tests.
 *
 * It mirrors the parts of ST 1.19 the extension actually touches: the
 * `#extensions_settings2` column, the sticky APP_READY event, extension settings
 * storage and the connection settings that the extension is allowed to touch.
 */
export function createHarness({ enabled = false, integrationKey = 'k'.repeat(44) } = {}) {
    const dom = new JSDOM(
        `<!DOCTYPE html><html><body>
            <div id="rm_extensions_block">
                <div class="extensions_block">
                    <div id="extensions_settings" class="flex1 wide50p"></div>
                    <div id="extensions_settings2" class="flex1 wide50p">
                        <div id="websearch_container" class="extension_container"></div>
                        <div id="dice_container" class="extension_container"></div>
                    </div>
                </div>
            </div>
        </body></html>`,
        { url: 'http://127.0.0.1:8000/', pretendToBeVisual: true },
    );

    const { window } = dom;
    const previous = {};
    const globals = {
        window,
        document: window.document,
        HTMLElement: window.HTMLElement,
        Node: window.Node,
        NodeFilter: window.NodeFilter,
        CSS: window.CSS ?? { supports: () => false },
        structuredClone,
    };
    for (const [key, value] of Object.entries(globals)) {
        previous[key] = globalThis[key];
        globalThis[key] = value;
    }
    Object.defineProperty(globalThis, 'navigator', {
        value: window.navigator,
        configurable: true,
    });

    const APP_READY = 'app_ready';
    const EXTENSION_SETTINGS_LOADED = 'extension_settings_loaded';

    class EventEmitter {
        constructor(sticky = []) {
            this.handlers = new Map();
            this.sticky = new Set(sticky);
            this.lastArgs = new Map();
        }

        on(type, callback) {
            if (typeof type !== 'string' || !type) {
                throw new TypeError(`Invalid event type: ${String(type)}`);
            }
            if (!this.handlers.has(type)) {
                this.handlers.set(type, []);
            }
            this.handlers.get(type).push(callback);
            if (this.sticky.has(type) && this.lastArgs.has(type)) {
                callback(...this.lastArgs.get(type));
            }
            return this;
        }

        async emit(type, ...args) {
            this.lastArgs.set(type, args);
            for (const callback of this.handlers.get(type) ?? []) {
                await callback(...args);
            }
        }
    }

    const eventTypes = {
        APP_READY,
        EXTENSION_SETTINGS_LOADED,
        GENERATION_STARTED: 'generation_started',
        GENERATION_ENDED: 'generation_ended',
        CHAT_COMPLETION_PROMPT_READY: 'chat_completion_prompt_ready',
        CHAT_COMPLETION_SETTINGS_READY: 'chat_completion_settings_ready',
        CHAT_CHANGED: 'chat_id_changed',
        MESSAGE_RECEIVED: 'message_received',
        MESSAGE_EDITED: 'message_edited',
        MESSAGE_DELETED: 'message_deleted',
        MESSAGE_SWIPED: 'message_swiped',
        CONNECTION_PROFILE_LOADED: 'connection_profile_loaded',
        CONNECTION_PROFILE_CREATED: 'connection_profile_created',
        CONNECTION_PROFILE_UPDATED: 'connection_profile_updated',
        CONNECTION_PROFILE_DELETED: 'connection_profile_deleted',
        OAI_PRESET_CHANGED_AFTER: 'oai_preset_changed_after',
        CHATCOMPLETION_SOURCE_CHANGED: 'chatcompletion_source_changed',
        CHATCOMPLETION_MODEL_CHANGED: 'chatcompletion_model_changed',
    };

    const extensionSettings = {
        roleplay_kernel: {
            enabled,
            autoRoute: true,
            sidecarUrl: 'http://127.0.0.1:8787/v1',
            integrationKey,
            model: 'roleplay-kernel',
            profileId: '',
            mode: 'balanced',
            requestDelaySeconds: 0,
            language: 'en',
            pov: 'third_person_limited',
            tense: 'past',
            previousConnection: null,
        },
        disabledExtensions: [],
        connectionManager: { selectedProfile: '', profiles: [] },
    };

    const chatCompletionSettings = {
        chat_completion_source: 'custom',
        custom_url: 'https://api.provider.example/v1',
        custom_model: 'some-model',
        custom_prompt_post_processing: 'strict',
        custom_include_headers: 'Authorization: Bearer user-own-token',
    };

    const saves = { count: 0 };
    const context = {
        extensionSettings,
        eventSource: new EventEmitter([APP_READY]),
        eventTypes,
        chatCompletionSettings,
        mainApi: 'openai',
        chatMetadata: null,
        uuidv4: () => 'session-under-test',
        saveSettingsDebounced: () => { saves.count += 1; },
        saveSettings: async () => { saves.count += 1; },
        CONNECT_API_MAP: {},
        getRequestHeaders: () => ({}),
        renderExtensionTemplateAsync: async (_name, template) => {
            if (template !== 'settings') {
                throw new Error(`unknown template ${template}`);
            }
            return fs.readFileSync(path.join(extensionDir(), 'settings.html'), 'utf8');
        },
    };

    const toasts = [];
    const errors = [];
    const previousToastr = globalThis.toastr;
    const previousConsoleError = console.error;
    const previousConsoleWarn = console.warn;
    globalThis.toastr = {
        info: message => toasts.push(['info', message]),
        success: message => toasts.push(['success', message]),
        warning: message => toasts.push(['warning', message]),
        error: message => toasts.push(['error', message]),
    };
    console.error = (...args) => errors.push(args.map(String).join(' '));
    console.warn = (...args) => errors.push(args.map(String).join(' '));

    globalThis.SillyTavern = {
        getContext: () => context,
        libs: { yaml: null },
    };

    async function load() {
        const entry = path.join(extensionDir(), 'index.js');
        return import(`${pathToFileURL(entry).href}?t=${Date.now()}`);
    }

    async function activate() {
        const module = await load();
        module.onActivate();
        await context.eventSource.emit(APP_READY);
        await new Promise(resolve => setTimeout(resolve, 50));
        return module;
    }

    function cleanup() {
        for (const [key, value] of Object.entries(previous)) {
            globalThis[key] = value;
        }
        globalThis.toastr = previousToastr;
        console.error = previousConsoleError;
        console.warn = previousConsoleWarn;
        delete globalThis.SillyTavern;
        window.close();
    }

    return {
        window,
        document: window.document,
        context,
        chatCompletionSettings,
        extensionSettings,
        saves,
        toasts,
        errors,
        activate,
        cleanup,
    };
}

function extensionDir() {
    return process.env.RPK_EXT_DIR
        ? path.resolve(process.env.RPK_EXT_DIR)
        : REPO_ROOT;
}
