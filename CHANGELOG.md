# Changelog

All notable changes to the OCI GenAI Provider will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-05

### Added

- **Gemini Tool Calling Support**: Full multi-turn tool calling now works with Google Gemini models
  - AssistantMessage uses `toolCalls` array at message level (OCI Generic format)
  - ToolMessage uses `toolCallId` at message level with TEXT content
- **Cohere V2 API Support**: Command A models now use COHEREV2 format for improved tool handling
- **SSE Streaming**: Proper Server-Sent Events parsing for all model families
  - Cohere V2 streaming with `message.content[]` format
  - Generic/Gemini streaming with `event.message.content[]` format
- **xAI Grok 4.1 Fast**: Added support for Grok reasoning model
- **Comprehensive Test Suite**: 142 unit tests covering all model families

### Fixed

- **Gemini Multi-Turn Tool Calls**: Fixed "Please pass in correct format of request" error
  - Tool calls now correctly placed in `toolCalls` array (not `TOOL_CALL` content type)
  - Tool results use `toolCallId` at message level (not `FUNCTION_RESPONSE` content)
- **Cohere V2 Streaming**: Fixed SSE parsing for `message.content[]` array format
- **Usage Token Fields**: Fixed AI SDK v5 compatibility (`inputTokens`/`outputTokens`)
- **JSON Schema Cleaning**: Improved handling of `pattern` keyword (regex vs property name)

### Changed

- Model family detection now distinguishes `cohere-v2` for Command A models
- Updated AGENTS.md with Gemini tool handling documentation
- Line number references updated for `convertMessagesToGenericFormat()` (~line 1988)

### Model Compatibility

| Model | Tool Calling | Notes |
|-------|--------------|-------|
| `cohere.command-a-03-2025` | 17/17 tests | Full support |
| `google.gemini-2.5-flash` | 15/17 tests | Streaming tools limited by OCI |
| `meta.llama-3.3-70b-instruct` | 13/17 tests | Text conversion for tool history |
| `xai.grok-*` | Works | Text conversion for tool history |

## [0.1.0] - 2026-01-15

### Added

- Initial release with OCI GenAI integration
- Support for Cohere, Google Gemini, Meta Llama, and xAI Grok models
- Basic text generation and streaming
- Tool calling foundation with JSON Schema cleaning
- Multi-region support (US and EU endpoints)
