# Technology Stack

## Runtime & Build
- **Node.js**: >=18
- **TypeScript**: ^5.9.0 (ES2020 target)
- **Build Tool**: tsup (bundles to CJS + ESM with types)
- **Dev Runtime**: tsx (for running TypeScript files in development)

## Core Dependencies
- **OCI SDKs**: 
  - `oci-common` (^2.124.0) - Common OCI utilities
  - `oci-generativeai` (^2.124.0) - GenAI service client
  - `oci-generativeaiinference` (^2.124.0) - Inference API client
- **CLI UI**: `@inquirer/prompts` (^7.5.1) - Interactive prompts
- **Formatting**: `chalk` (^5.4.2) - Terminal color output

## Peer Dependencies
- **@ai-sdk/provider**: >=1.0.0 <3.0.0 (AI SDK V2 compatibility)

## Dev Dependencies
- **Testing**: `vitest` (^2.0.0)
- **Type Definitions**: `@types/node` (^22.0.0)

## Build & Distribution
- **Main Entry**: dist/index.js (CommonJS)
- **ESM Entry**: dist/index.mjs
- **Types**: dist/index.d.ts (generated via tsup)
- **CLI Entry**: ./dist/cli.js (executable via `opencode-oci-setup`)

## Config Files
- **tsconfig.json**: TypeScript configuration with strict mode enabled
- **package.json**: Project metadata and scripts
