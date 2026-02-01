# Code Style & Conventions

## TypeScript Configuration
- **Target**: ES2020
- **Module Resolution**: bundler
- **Strict Mode**: Enabled (all strict checks active)
- **Declaration Maps**: Enabled (for source mapping)
- **ESM Interop**: Enabled

## Naming Conventions
- **Classes**: PascalCase (e.g., `OCIProvider`, `OCIChatLanguageModelV2`)
- **Functions**: camelCase (e.g., `createOCIProvider`, `discoverAvailableModels`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `SWE_PRESETS`, `KNOWN_MODELS`)
- **Interfaces**: PascalCase with `I` prefix optional (e.g., `OCIProviderSettings`, `ModelInfo`)
- **Variables**: camelCase

## Code Organization
- **Exports**: Named exports preferred over default exports
- **Type Safety**: Strict typing throughout
- **Error Handling**: Explicit error handling with proper error types
- **Comments**: Used sparingly, only for non-obvious logic

## Module Structure
- **index.ts**: Main provider implementation and exports
- **cli.ts**: Interactive setup wizard (oclif style prompts)
- **data/regions.ts**: Configuration data and constants

## Key Patterns
- Functional utilities for model/region logic
- Class-based architecture for provider implementation
- Interactive CLI using Inquirer library
- Configuration-driven setup process
