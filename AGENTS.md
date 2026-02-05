# OCI GenAI Provider - Agent Guidelines

## Overview

This provider enables OpenCode to use Oracle Cloud Infrastructure (OCI) GenAI models. OCI GenAI has stricter API requirements than standard OpenAI-compatible endpoints, requiring specific workarounds.

## Critical: Tool Calling Compatibility

### The Problem

OCI GenAI validates JSON Schema strictly. Standard AI SDK tool schemas include keywords that Gemini and Llama models reject:

```
Error: "Please pass in correct format of request"
Error: "required fields ['pattern'] are not defined in the schema properties"
```

### The Solution

The provider implements `cleanJsonSchema()` in `src/index.ts:~789-850` that strips unsupported keywords before sending to OCI.

**Stripped keywords:**
- Schema metadata: `$schema`, `$ref`, `$defs`, `definitions`, `$id`, `$comment`
- Validation constraints: `additionalProperties`, `format`, `pattern` (regex only)
- String constraints: `minLength`, `maxLength`
- Array constraints: `minItems`, `maxItems`
- Number constraints: `exclusiveMinimum`, `exclusiveMaximum`
- Other: `title`, `examples`, `default`, `propertyNames`, `const`

### Special Case: `pattern` Keyword

The `pattern` keyword has dual meaning in JSON Schema:
1. **Regex constraint** - e.g., `{ "type": "string", "pattern": "^[a-z]+$" }` - MUST be removed
2. **Property name** - e.g., `{ "properties": { "pattern": { "type": "string" } } }` - MUST be preserved

The fix distinguishes these cases:
```typescript
if (key === 'pattern' && typeof value === 'string' && schema.type === 'string') {
  continue; // Only skip regex constraints
}
```

## Model Compatibility Matrix

| Model | Provider | Tool Calling | Notes |
|-------|----------|--------------|-------|
| `cohere.command-r-plus-08-2024` | Cohere | Works | Returns XML-formatted tool calls |
| `cohere.command-r-08-2024` | Cohere | Works | Returns XML-formatted tool calls |
| `google.gemini-2.0-flash-001` | Google | Works | Requires schema cleaning |
| `google.gemini-1.5-pro-002` | Google | Works | Requires schema cleaning |
| `google.gemini-2.5-flash` | Google | Works | Requires schema cleaning |
| `meta.llama-3.1-405b-instruct` | Meta | Works | Requires schema cleaning + text conversion |
| `meta.llama-3.1-70b-instruct` | Meta | Works | Requires schema cleaning + text conversion |
| `meta.llama-3.3-70b-instruct` | Meta | Works | Requires schema cleaning + text conversion |
| `xai.grok-*` | xAI | Works | Requires text conversion for tool history |

## Llama and xAI Tool Handling

Llama and xAI/Grok models reject the `TOOL` role and `TOOL_CALL` content type in message history. The provider converts these to text representations:

**Tool calls in assistant messages** (converted to TEXT):
```
[Called tool "bash" with: {"command":"ls"}]
```

**Tool results** (converted to USER messages):
```
[Tool result from "bash": file1.txt\nfile2.txt]
```

This conversion is handled in `convertMessagesToGenericFormat()` (~line 961) with `isLlama` and `isXAI` checks.

## Debugging

Enable debug logging to inspect cleaned schemas:

```bash
OCI_DEBUG=1 opencode run "list files" --model oci-eu/google.gemini-2.5-flash
```

This outputs:
- Original tool schemas
- Cleaned schemas sent to OCI
- Raw API request/response

## When Modifying Tool Handling

If you need to modify tool/function calling:

1. **Location**: `src/index.ts` in `OCIChatLanguageModelV2.doGenerate()`
2. **Schema cleaning**: `cleanJsonSchema()` method (~line 789)
3. **Tool format**: OCI expects `{ type: "FUNCTION", functionDeclarations: [...] }`
4. **Test with**: Both Gemini AND Llama models (different validation)

### Test Commands

```bash
# Build
npm run build

# Test with Gemini
OCI_DEBUG=1 opencode run "list files" --model oci-eu/google.gemini-2.5-flash

# Test with Llama  
OCI_DEBUG=1 opencode run "list files" --model oci-eu/meta.llama-3.3-70b-instruct

# Test with Cohere (baseline - less strict)
OCI_DEBUG=1 opencode run "list files" --model oci-eu/cohere.command-r-plus-08-2024
```

## Key Files

| File | Purpose |
|------|---------|
| `src/index.ts` | Main provider, includes `cleanJsonSchema()` |
| `src/index.test.ts` | Tests including schema cleaning |
| `src/data/regions.ts` | Region and model availability |

---

**Last Updated:** 2026-02-05
