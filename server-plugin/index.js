'use strict';

const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');

const PLUGIN_ID = 'roleplay-kernel';
const PLUGIN_ROOT = __dirname;
const RUNTIME_SETTINGS_PATH = path.join(PLUGIN_ROOT, 'runtime.json');
const DATA_ROOT = process.env.RPK_DATA_DIR || path.join(PLUGIN_ROOT, 'data');
const DEFAULT_CONFIG_PATH = path.join(DATA_ROOT, 'config.json');
const DEFAULT_PORT = 8787;

let child = null;
let starting = null;
let activeRuntime = null;

function runtimeSettings() {
    return readJson(RUNTIME_SETTINGS_PATH) || {};
}

function configPath() {
    return runtimeSettings().configPath || DEFAULT_CONFIG_PATH;
}

function stateDirectory(filePath) {
    return path.join(path.dirname(filePath), 'state');
}

function readJson(filePath) {
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
    } catch {
        return null;
    }
}

function writeJson(filePath, value) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
    try {
        fs.chmodSync(filePath, 0o600);
    } catch {
        return;
    }
}

function isUsableKey(value) {
    return typeof value === 'string'
        && value.length >= 32
        && [...value].every(character => character.charCodeAt(0) >= 0x21 && character.charCodeAt(0) <= 0x7e);
}

function findFreePort() {
    return new Promise((resolve, reject) => {
        const server = net.createServer();
        server.once('error', reject);
        server.listen(0, '127.0.0.1', () => {
            const address = server.address();
            const port = typeof address === 'object' && address ? address.port : DEFAULT_PORT;
            server.close(error => {
                if (error) {
                    reject(error);
                    return;
                }
                resolve(port);
            });
        });
    });
}

function requestHealth(port, timeoutMs = 750) {
    return new Promise(resolve => {
        const request = http.get({
            host: '127.0.0.1',
            port,
            path: '/health',
            headers: { Accept: 'application/json' },
            timeout: timeoutMs,
        }, response => {
            let body = '';
            response.setEncoding('utf8');
            response.on('data', chunk => {
                if (body.length < 65536) {
                    body += chunk;
                }
            });
            response.on('end', () => {
                if (!response.statusCode || response.statusCode >= 400) {
                    resolve(null);
                    return;
                }
                try {
                    const data = JSON.parse(body);
                    resolve(data && data.service === 'roleplay-kernel-sidecar' ? data : null);
                } catch {
                    resolve(null);
                }
            });
        });
        request.once('timeout', () => {
            request.destroy();
            resolve(null);
        });
        request.once('error', () => resolve(null));
    });
}

function waitForHealth(port, childProcess) {
    const deadline = Date.now() + 15000;
    return new Promise((resolve, reject) => {
        const poll = async () => {
            if (childProcess && childProcess.pid == null) {
                reject(new Error('Roleplay Kernel runtime process could not be started'));
                return;
            }
            if (childProcess && childProcess.exitCode !== null) {
                reject(new Error(`Roleplay Kernel runtime exited with code ${childProcess.exitCode}`));
                return;
            }
            const health = await requestHealth(port);
            if (health) {
                resolve(health);
                return;
            }
            if (Date.now() >= deadline) {
                reject(new Error('Roleplay Kernel runtime did not become ready within 15 seconds'));
                return;
            }
            setTimeout(() => void poll(), 250);
        };
        void poll();
    });
}

function resolveRuntime() {
    const settings = readJson(RUNTIME_SETTINGS_PATH) || {};
    const bundled = settings.executable || path.join(PLUGIN_ROOT, 'runtime', 'roleplay-kernel.exe');
    if (typeof bundled === 'string' && fs.existsSync(bundled)) {
        return {
            command: bundled,
            args: ['--config', settings.configPath || configPath()],
            cwd: settings.cwd || PLUGIN_ROOT,
            env: process.env,
        };
    }
    const python = settings.python || process.env.RPK_PYTHONW || 'pythonw.exe';
    return {
        command: python,
        args: [
            '-m',
            settings.module || 'roleplay_kernel.sidecar',
            '--config',
            settings.configPath || configPath(),
        ],
        cwd: settings.cwd || PLUGIN_ROOT,
        env: {
            ...process.env,
            ...(settings.pythonPath ? { PYTHONPATH: settings.pythonPath } : {}),
        },
    };
}

function buildConfig(port) {
    const filePath = configPath();
    const existing = readJson(filePath) || {};
    const integrationKey = isUsableKey(existing.integration_key)
        ? existing.integration_key
        : crypto.randomBytes(32).toString('base64');
    return {
        host: '127.0.0.1',
        port,
        upstream_base_url: existing.upstream_base_url || 'http://127.0.0.1:1/v1',
        upstream_model: existing.upstream_model || 'profile-required',
        upstream_api_key_env: existing.upstream_api_key_env || 'OPENAI_API_KEY',
        upstream_token_parameter: existing.upstream_token_parameter || 'max_tokens',
        integration_key: integrationKey,
        state_dir: existing.state_dir || stateDirectory(filePath),
        mode: existing.mode || 'balanced',
        context_window: existing.context_window || 32768,
        token_budget: existing.token_budget || 18000,
        max_output_tokens: Math.max(Number(existing.max_output_tokens) || 0, 8192),
        max_internal_tokens: Math.max(Number(existing.max_internal_tokens) || 0, 4096),
        upstream_timeout_seconds: Math.max(Number(existing.upstream_timeout_seconds) || 0, 300),
        max_repairs: existing.max_repairs ?? 1,
        max_context_chars: existing.max_context_chars || 16000,
        allow_insecure_http: existing.allow_insecure_http === true,
    };
}

function runtimeStatus() {
    const running = Boolean(child && child.exitCode === null);
    return {
        running,
        pid: running ? child.pid : null,
        port: activeRuntime?.port || null,
        sidecar_url: activeRuntime?.sidecarUrl || null,
    };
}

async function launchRuntime() {
    if (child && child.exitCode === null) {
        return runtimeStatus();
    }
    if (starting) {
        return starting;
    }
    starting = (async () => {
        const port = await findFreePort();
        const selectedConfigPath = configPath();
        const config = buildConfig(port);
        writeJson(selectedConfigPath, config);
        const runtime = resolveRuntime();
        const sidecarUrl = `http://127.0.0.1:${port}/v1`;
        const processHandle = spawn(runtime.command, runtime.args, {
            cwd: runtime.cwd,
            env: runtime.env,
            windowsHide: true,
            detached: false,
            stdio: 'ignore',
        });
        child = processHandle;
        activeRuntime = { port, sidecarUrl, configPath: selectedConfigPath, integrationKey: config.integration_key };
        processHandle.once('error', error => {
            child = null;
            activeRuntime = null;
            console.error(`[${PLUGIN_ID}] failed to start runtime: ${error.message}`);
        });
        processHandle.once('exit', () => {
            if (child === processHandle) {
                child = null;
                activeRuntime = null;
            }
        });
        try {
            await waitForHealth(port, processHandle);
        } catch (error) {
            if (processHandle.exitCode === null) {
                processHandle.kill();
            }
            child = null;
            activeRuntime = null;
            throw error;
        }
        return {
            ...runtimeStatus(),
            integration_key: config.integration_key,
            config_path: selectedConfigPath,
        };
    })();
    try {
        return await starting;
    } finally {
        starting = null;
    }
}

function stopRuntime() {
    if (child && child.exitCode === null) {
        child.kill();
    }
    child = null;
    activeRuntime = null;
    return runtimeStatus();
}

async function init(router) {
    router.get('/status', (_request, response) => {
        response.json(runtimeStatus());
    });
    router.post('/launch', async (_request, response) => {
        try {
            response.json(await launchRuntime());
        } catch (error) {
            response.status(500).json({ error: String(error.message || error) });
        }
    });
    router.post('/stop', (_request, response) => {
        response.json(stopRuntime());
    });
}

async function exit() {
    stopRuntime();
}

module.exports = {
    init,
    exit,
    info: {
        id: PLUGIN_ID,
        name: 'Roleplay Kernel',
        description: 'Starts and manages the local Roleplay Kernel runtime.',
    },
};
