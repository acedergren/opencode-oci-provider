# opencode-oci-provider

OCI GenAI provider for OpenCode with interactive setup wizard.

## Quick Start

```bash
npm install              # Install dependencies
npm run dev             # Run CLI setup wizard
npm run build           # Build distribution (CJS + ESM)
npm run test            # Run tests
```

## Essential Commands

| Command | Purpose |
|---------|---------|
| `npm run build` | Build dist/ with CJS + ESM + types |
| `npm run dev` | Run setup wizard in development |
| `npm run test` | Run Vitest suite |
| `npm publish` | Publish to npm (auto-runs prepublishOnly) |

## Key Info

- **Language:** TypeScript (ES2020, strict mode)
- **Runtime:** Node.js 18+
- **Build Tool:** tsup (bundles to CJS + ESM)
- **Package Manager:** npm
- **License:** MIT
- **AI SDK Compatibility:** This provider is compatible with AI SDK v5.x (OpenCode uses v5.0.124)
  - **Important:** When testing, use `ai@5.x` NOT `ai@6.x` - version 6 has breaking changes
  - Provider interface: `@ai-sdk/provider@2.0.1` (LanguageModelV2 interface)
  - Peer dependency: `@ai-sdk/provider@>=1.0.0 <3.0.0` (allows v1.x and v2.x)

## Detailed Guidelines

For specific conventions and workflows:

- [Code Style & Conventions](.claude/code-style.md)
- [TypeScript Patterns](.claude/tech-stack.md)
- [Testing & Quality](.claude/post-task-checklist.md)
- [Available Commands](.claude/suggested-commands.md)
- [Project Architecture](.claude/project-overview.md)

## Tool Calling Workaround (Gemini/Llama Models)

OCI GenAI has stricter JSON Schema validation than standard OpenAI-compatible APIs. The provider implements automatic schema cleaning in `cleanJsonSchema()` (`src/index.ts:~789-850`).

### Problem
Gemini and Llama models reject tool calls with unsupported JSON Schema keywords. Error: `"Please pass in correct format of request"` or `"required fields ['pattern'] are not defined"`.

### Solution Implemented
The `cleanJsonSchema()` method strips unsupported keywords:
- `$schema`, `$ref`, `$defs`, `definitions`, `$id`, `$comment`
- `additionalProperties`, `title`, `examples`, `default`, `format`
- `minLength`, `maxLength`, `minItems`, `maxItems`
- `exclusiveMinimum`, `exclusiveMaximum`, `propertyNames`, `const`
- `pattern` (only when it's a regex constraint, NOT a property name)

### Key Fix
The `pattern` keyword needs special handling:
```typescript
// Only remove 'pattern' when it's a regex constraint (string value + type: 'string')
// Preserve 'pattern' when it's a property name in tools like glob/grep
if (key === 'pattern' && typeof value === 'string' && schema.type === 'string') {
  continue; // Skip regex constraints only
}
```

### Model Compatibility

| Model | Tool Calling | Notes |
|-------|-------------|-------|
| Cohere Command R+ | Works (XML output) | Tools work but output as XML text |
| Google Gemini | Works (with workaround) | Requires schema cleaning |
| Meta Llama | Works (with workaround) | Requires schema cleaning |
| xAI Grok | Untested | US regions only |

### Debugging
Enable debug logging to see cleaned schemas:
```bash
OCI_DEBUG=1 opencode run "list files" --model oci-eu/google.gemini-2.5-flash
```

## OCI API Parameters

The provider forwards these AI SDK and OCI-specific parameters:

| Parameter | Source | Builders |
|-----------|--------|----------|
| `topK` | `options.topK` | Generic, Cohere V1, Cohere V2 |
| `seed` | `providerOptions['oci-genai'].seed` | Generic, Cohere V1, Cohere V2 |
| `safetyMode` | `providerOptions['oci-genai'].safetyMode` | Cohere V1, Cohere V2 |
| `maxCompletionTokens` | `providerOptions['oci-genai'].maxCompletionTokens` | Generic |
| `responseFormat` | `options.responseFormat` (text/json) | Generic |
| `toolChoice` | `options.toolChoice` (auto/required/none/tool) | Generic, Cohere V2 |
| `stopSequences` | `options.stopSequences` | Generic, Cohere V2 |
| `reasoningEffort` | `providerOptions['oci-genai'].reasoningEffort` | Generic (not xAI) |
| `thinkingBudgetTokens` | `providerOptions['oci-genai'].thinkingBudgetTokens` | Cohere V1, Cohere V2 |

### Finish Reason Mappings

OCI finish reasons are mapped to AI SDK v5 values:
- `COMPLETE/stop/STOP` → `'stop'`
- `MAX_TOKENS/length` → `'length'`
- `TOOL_CALL/tool_calls/TOOL_USE` → `'tool-calls'`
- `CONTENT_FILTER/content_filter/ERROR_TOXIC` → `'content-filter'`
- `ERROR/ERROR_LIMIT` → `'error'`
- `USER_CANCEL` → `'other'`

### Authentication

Supported auth provider types via `OCIProviderSettings.authProvider`:
- `'config-file'` (default) — uses `~/.oci/config`
- `'session-token'` — uses session token auth
- Pre-built `AuthenticationDetailsProvider` instance — for async providers (InstancePrincipals, ResourcePrincipal, OKE Workload Identity)

Region is auto-detected from OCI config profile when not explicitly set.

### AbortSignal Support

Both `doGenerate` and `doStream` support cancellation via `options.abortSignal`.

### Response Metadata

`doGenerate` returns `request.body` (the chat request sent to OCI) and `response.{id, timestamp, modelId}` for telemetry. `doStream` emits a `response-metadata` event with `modelId` and `timestamp`.

## Pre-Commit

Before committing:

```bash
npm run build   # Verify build succeeds
npm run test    # Verify tests pass
```

Then commit with descriptive message addressing the "why".

## Project Structure

```
src/
├── index.ts           # Main provider (OCIProvider, OCIChatLanguageModelV2)
├── cli.ts            # Interactive setup wizard
└── data/
    └── regions.ts    # Region & model availability data
```

---

**Last Updated:** 2026-02-05
**Repo:** https://github.com/acedergr/opencode-oci-provider
