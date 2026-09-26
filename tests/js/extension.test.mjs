import assert from 'node:assert/strict';
import test from 'node:test';

import { createHarness } from './harness.mjs';

test('the settings panel is inserted into the extensions column and expanded', async () => {
    const harness = createHarness();
    try {
        await harness.activate();
        const panel = harness.document.getElementById('rpk_panel');
        assert.ok(panel, 'panel must be inserted');
        assert.equal(panel.parentElement.id, 'extensions_settings2');
        const content = panel.querySelector('.inline-drawer > .inline-drawer-content');
        assert.equal(content.style.display, 'block', 'panel must be expanded by default');
        assert.ok(harness.document.getElementById('rpk_progress_bar'), 'progress bar must exist');
        assert.ok(harness.document.getElementById('rpk_header_status'), 'header status must exist');
        assert.equal(harness.errors.length, 0, `unexpected errors: ${harness.errors.join(' | ')}`);
    } finally {
        harness.cleanup();
    }
});

test('the panel is rendered only once across repeated activation', async () => {
    const harness = createHarness();
    try {
        await harness.activate();
        await harness.context.eventSource.emit('app_ready');
        await harness.context.eventSource.emit('extension_settings_loaded');
        await new Promise(resolve => setTimeout(resolve, 50));
        const panels = harness.document.querySelectorAll('#rpk_panel');
        assert.equal(panels.length, 1, 'panel must not be duplicated');
    } finally {
        harness.cleanup();
    }
});

test('the language selector is absent and the language is fixed to English', async () => {
    const harness = createHarness();
    try {
        await harness.activate();
        assert.equal(harness.document.getElementById('rpk_language'), null);
        assert.equal(harness.extensionSettings.roleplay_kernel.language, 'en');
    } finally {
        harness.cleanup();
    }
});

test('a stale model or language value is normalised on activation', async () => {
    const harness = createHarness();
    try {
        harness.extensionSettings.roleplay_kernel.model = 'GLM-5.3-Flash';
        harness.extensionSettings.roleplay_kernel.language = 'ru';
        await harness.activate();
        assert.equal(harness.extensionSettings.roleplay_kernel.model, 'roleplay-kernel');
        assert.equal(harness.extensionSettings.roleplay_kernel.language, 'en');
    } finally {
        harness.cleanup();
    }
});

test('the integration key is never written into the ST connection settings', async () => {
    const harness = createHarness();
    try {
        await harness.activate();
        const headers = String(harness.chatCompletionSettings.custom_include_headers ?? '');
        assert.ok(
            !headers.includes(harness.extensionSettings.roleplay_kernel.integrationKey),
            `integration key leaked into custom_include_headers: ${headers}`,
        );
        assert.ok(
            headers.includes('user-own-token'),
            "the user's own header must be preserved",
        );
    } finally {
        harness.cleanup();
    }
});

test('the user own authorization header survives an activation round trip', async () => {
    const harness = createHarness();
    try {
        await harness.activate();
        const before = harness.chatCompletionSettings.custom_include_headers;
        await harness.context.eventSource.emit('chatcompletion_source_changed');
        assert.equal(harness.chatCompletionSettings.custom_include_headers, before);
    } finally {
        harness.cleanup();
    }
});
