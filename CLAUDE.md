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

## Detailed Guidelines

For specific conventions and workflows:

- [Code Style & Conventions](.claude/code-style.md)
- [TypeScript Patterns](.claude/tech-stack.md)
- [Testing & Quality](.claude/post-task-checklist.md)
- [Available Commands](.claude/suggested-commands.md)
- [Project Architecture](.claude/project-overview.md)

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

**Last Updated:** 2026-02-01  
**Repo:** https://github.com/acedergr/opencode-oci-provider
