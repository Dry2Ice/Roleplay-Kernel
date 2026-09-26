import js from '@eslint/js';
import globals from 'globals';

export default [
    {
        ignores: ['node_modules/**', 'build/**', '.ruff_cache/**', '.mypy_cache/**'],
    },
    js.configs.recommended,
    {
        // The extension entry point is an ES module even though it is served as
        // .js, so the parser must be told explicitly. Plain `node --check`
        // parses .js as CommonJS and silently accepts broken module syntax,
        // which is how a SyntaxError reached a release once.
        files: ['index.js'],
        languageOptions: {
            ecmaVersion: 2023,
            sourceType: 'module',
            globals: {
                ...globals.browser,
                SillyTavern: 'readonly',
                toastr: 'readonly',
                jQuery: 'readonly',
                $: 'readonly',
            },
        },
        rules: {
            'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
            eqeqeq: ['error', 'always', { null: 'ignore' }],
            'no-var': 'error',
            'prefer-const': 'error',
            'no-console': ['error', { allow: ['warn', 'error', 'info'] }],
        },
    },
    {
        files: ['server-plugin/**/*.js'],
        languageOptions: {
            ecmaVersion: 2023,
            sourceType: 'commonjs',
            globals: { ...globals.node },
        },
        rules: {
            'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
            eqeqeq: ['error', 'always', { null: 'ignore' }],
            'no-var': 'error',
            'prefer-const': 'error',
        },
    },
    {
        files: ['tests/js/**/*.mjs'],
        languageOptions: {
            ecmaVersion: 2023,
            sourceType: 'module',
            globals: { ...globals.node },
        },
        rules: {
            'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
        },
    },
];
