# Changelog

All notable changes to the OCI GenAI Provider will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-05

### Added

- **Tool Choice Support**: Forward AI SDK `toolChoice` (auto/required/none/tool) to OCI Generic format models (Gemini, Llama, xAI, OpenAI)
- **AbortSignal Support**: Wire up `abortSignal` for cancellation in both `doGenerate` and `doStream`
- **Response Metadata**: Return `request.body`, `response.id`, `response.timestamp`, and `response.modelId` from `doGenerate`; emit `response-metadata` stream event in `doStream`
- **New OCI API Parameters**:
  - `topK` for all model families (Cohere, Cohere V2, Generic)
  - `seed` for reproducible outputs across all model families
  - `responseFormat` (text/json-object/json-schema) for Generic format models
  - `maxCompletionTokens` for reasoning model token budgeting
  - `safetyMode` (CONTEXTUAL/STRICT/OFF) for Cohere V1 and V2 models
  - `stopSequences` for Cohere V2 models (was missing, already existed for V1)
- **Session Token Auth**: Support `session-token` auth provider type for OCI CLI session authentication
- **Pre-built Auth Providers**: Accept pre-built `AuthenticationDetailsProvider` instances for async auth types (InstancePrincipals, ResourcePrincipal, OKE Workload Identity)
- **Region Auto-Detection**: Automatically read region from OCI config profile when not explicitly set
- **Reasoning Tokens in Usage**: Pass through `reasoningTokens` count when available from OCI response
- 38 new tests covering all added functionality (183 total)

### Fixed

- **Finish Reason Mappings**: Map `ERROR` → `'error'`, `ERROR_TOXIC` → `'content-filter'`, `ERROR_LIMIT` → `'error'`, `USER_CANCEL` → `'other'` (previously all defaulted to `'stop'`)

## [0.2.3] - 2026-02-05

### Fixed

- **xAI Grok reasoningEffort**: Don't send `reasoningEffort` parameter for xAI Grok models (not supported by API)

### Tests

- Add xAI reasoningEffort exclusion tests and fix Gemini parallel tool test

## [0.2.2] - 2026-02-05

### Fixed

- **Gemini Streaming Tool Calls**: Handle `toolCalls` in streaming Generic format for Gemini models

## [0.2.1] - 2026-02-05

### Added

- **Llama Tool Calling**: Full tool calling support for Meta Llama models with text-based tool history conversion
- **xAI Grok 4.1 Fast**: Reasoning model support for xAI Grok
- **Multi-Region Support**: Configure provider for US and EU OCI regions
- **Regression Test Suite**: Comprehensive tests for Llama tool calling, error handling, and multi-turn conversations

### Fixed

- **Gemini Multi-Turn Tool Calling**: Correct OCI Generic format with `toolCalls` array at message level
- **Gemini Parallel Tool Calls**: Fallback to text representation when parallel calls not supported
- **Model ID Corrections**: Fixed model IDs for Gemini Pro, Llama 4 Maverick, xAI Grok
- **Reasoning Configuration**: Disable `reasoningEffort` for Gemini Flash-Lite and all Google Gemini models where unsupported
- **Build on npm Install**: Add `prepare` script for git-based installs

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
