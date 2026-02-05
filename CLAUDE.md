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
