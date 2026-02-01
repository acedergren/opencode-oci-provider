# Project Overview

## Purpose
`opencode-oci-provider` is an OCI GenAI provider for the OpenCode CLI tool with an interactive setup wizard. It enables OpenCode to use Oracle Cloud Infrastructure's generative AI models (Cohere, Google, xAI, Meta Llama).

## Key Features
- Interactive setup wizard (`opencode-oci-setup` CLI)
- Support for multiple AI model providers (Cohere, Google, xAI, Meta)
- Region availability detection with xAI region indicators
- On-Demand and Dedicated AI Cluster modes
- Configuration generation for `opencode.json` and `.env.oci-genai`
- AI SDK V2 compatible provider implementation

## Project Structure
```
src/
├── index.ts           # Main provider implementation (OCIProvider, OCIChatLanguageModelV2)
├── cli.ts            # Interactive setup wizard
└── data/
    └── regions.ts    # Region and model availability data
```

## Main Components
- **OCIProvider**: Main provider class implementing AI SDK provider interface
- **OCIChatLanguageModelV2**: Language model implementation for chat operations
- **CLI Setup Wizard**: Interactive prompts for configuration discovery and setup

## Codebase Style
- Modern TypeScript (ES2020 target)
- Strict type checking enabled
- Modular architecture with clear separation of concerns
- Interactive CLI using Inquirer prompts
- Functional approach for utility functions
