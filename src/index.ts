/**
 * OpenCode OCI Provider - AI SDK V2 compatible provider for OCI GenAI
 *
 * This package provides a LanguageModelV2 implementation for OpenCode,
 * with SWE-optimized defaults and tool calling support (including MCP).
 */
import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2Usage,
  LanguageModelV2Text,
  LanguageModelV2Reasoning,
  LanguageModelV2ToolCall,
  LanguageModelV2FunctionTool,
  ProviderV2,
  JSONSchema7,
} from '@ai-sdk/provider';
import * as oci from 'oci-generativeaiinference';
import * as common from 'oci-common';
import { isDedicatedOnly, getModelDisplayName } from './data/regions.js';

export interface OCIProviderSettings {
  compartmentId?: string;
  region?: string;
  configProfile?: string;
  servingMode?: 'on-demand' | 'dedicated';
  endpointId?: string;
}

/**
 * Model-specific SWE presets for optimal coding performance
 */
interface SWEPreset {
  temperature: number;
  topP: number;
  frequencyPenalty: number;
  presencePenalty: number;
  supportsTools: boolean;
  supportsPenalties: boolean;
  supportsStopSequences?: boolean;  // Default true if not specified
  supportsReasoning?: boolean;      // Default false if not specified
}

const SWE_PRESETS: Record<string, SWEPreset> = {
  // Cohere models - good for instruction following, supports tools
  // Note: Only cohere.command-a-reasoning-* models support thinking (handled in getSWEPreset)
  'cohere': {
    temperature: 0.2,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
    supportsReasoning: false,  // Only reasoning models support thinking (see getSWEPreset)
  },
  // Google Gemini - excellent for code (OCI does NOT support frequencyPenalty/presencePenalty)
  // Only Gemini Flash with explicit reasoningEffort config supports reasoning parameter
  'google': {
    temperature: 0.1,
    topP: 0.95,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: false,
    supportsReasoning: false,  // Only Flash variants with explicit config support reasoningEffort
  },
  // xAI Grok - supports tools, but NOT frequencyPenalty/presencePenalty or stop sequences
  // Reasoning is controlled by model variant selection (e.g., grok-4-1-fast-reasoning vs grok-4-1-fast-non-reasoning)
  // Models like grok-3-mini "think before responding" and return reasoning content
  'xai': {
    temperature: 0.1,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: false,
    supportsStopSequences: false,  // Per OCI docs, stop sequences not listed as supported
    supportsReasoning: false,      // Base value; overridden in getSWEPreset for reasoning model variants
  },
  // Meta Llama - balanced for code, no reasoning support
  'meta': {
    temperature: 0.2,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
    supportsReasoning: false,
  },
  // OpenAI gpt-oss models - supports tools, reasoning, and penalties
  'openai': {
    temperature: 0.1,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
    supportsReasoning: true,  // Has reasoning capabilities
  },
  // Default fallback
  'default': {
    temperature: 0.2,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
    supportsReasoning: false,
  },
};

function getModelProvider(modelId: string): string {
  const prefix = modelId.split('.')[0];
  return prefix || 'default';
}

function getSWEPreset(modelId: string): SWEPreset {
  const provider = getModelProvider(modelId);
  const basePreset = SWE_PRESETS[provider] || SWE_PRESETS['default'];

  // Gemini Flash-Lite has thinking disabled for speed/cost optimization
  if (modelId.includes('flash-lite')) {
    return { ...basePreset, supportsReasoning: false };
  }

  // xAI Grok models: reasoning is controlled by model variant, not API parameter
  // Models with "-reasoning" suffix or "mini" (which think before responding) support reasoning
  if (modelId.startsWith('xai.')) {
    // grok-4-1-fast-reasoning has explicit reasoning suffix
    // grok-4-1-fast-non-reasoning explicitly does NOT support reasoning
    // grok-3-mini and grok-3-mini-fast are "lightweight models that think before responding"
    const isReasoningModel = (modelId.endsWith('-reasoning') || 
                              modelId.includes('grok-3-mini')) &&
                             !modelId.includes('-non-reasoning');
    return { ...basePreset, supportsReasoning: isReasoningModel };
  }

  // Cohere reasoning models (command-a-reasoning-*) support thinking via thinkingBudgetTokens (not reasoningEffort)
  if (modelId.includes('reasoning')) {
    return { ...basePreset, supportsReasoning: true };
  }

  // Note: Google Gemini models do NOT support reasoningEffort parameter in OCI GenAI
  // Even though Gemini 2.5 Flash has reasoning capabilities, OCI doesn't expose the parameter

  // Note: Meta Llama 4 models do NOT support reasoningEffort parameter in OCI
  // They may have internal reasoning but don't expose API control for it

  return basePreset;
}

type ModelFamily = 'cohere' | 'cohere-v2' | 'generic';

/**
 * Determine which API format to use for a model.
 * - cohere-v2: Command A models require COHEREV2 format
 * - cohere: Legacy Command R/R+ models use COHERE format
 * - generic: All other models (Gemini, Llama, Grok) use GENERIC format
 */
function getModelFamily(modelId: string): ModelFamily {
  if (modelId.startsWith('cohere.')) {
    // Command A models require V2 API format
    if (modelId.includes('command-a')) {
      return 'cohere-v2';
    }
    return 'cohere';
  }
  return 'generic';
}

function generateId(): string {
  return `oci-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

function mapFinishReason(raw: string | undefined): LanguageModelV2FinishReason {
  switch (raw) {
    case 'MAX_TOKENS':
    case 'length':
      return 'length';
    case 'COMPLETE':
    case 'stop':
    case 'STOP':
      return 'stop';
    case 'TOOL_CALL':
    case 'tool_calls':
    case 'TOOL_USE':
      return 'tool-calls';
    case 'CONTENT_FILTER':
    case 'content_filter':
      return 'content-filter';
    default:
      return 'stop';
  }
}

function createUsage(promptTokens?: number, completionTokens?: number): LanguageModelV2Usage {
  return {
    inputTokens: promptTokens,
    outputTokens: completionTokens,
    totalTokens: promptTokens !== undefined && completionTokens !== undefined
      ? promptTokens + completionTokens
      : undefined,
  };
}

/**
 * Parse OCI API errors and return user-friendly messages.
 * OCI errors are often cryptic (e.g., "Please pass in correct format of request")
 * and don't provide actionable information.
 */
function parseOCIError(error: any, modelId: string): Error {
  const originalMessage = error?.message || String(error);
  
  // Extract useful info from OCI error structure
  const statusCode = error?.statusCode || error?.response?.status;
  const serviceCode = error?.serviceCode || error?.code;
  const opcRequestId = error?.opcRequestId;
  
  // Common OCI error patterns and their user-friendly translations
  const errorPatterns: Array<{ pattern: RegExp | string; message: string; hint?: string }> = [
    {
      pattern: /Please pass in correct format of request/i,
      message: 'OCI API rejected the request format',
      hint: 'This usually indicates an issue with message structure or tool call format. Enable OCI_DEBUG=1 for details.',
    },
    {
      pattern: /Service request limit is exceeded|request is throttled/i,
      message: 'Rate limit exceeded',
      hint: 'Wait a moment before retrying. Consider using a different model or region.',
    },
    {
      pattern: /NotAuthorizedOrNotFound|404/i,
      message: 'Model or resource not found',
      hint: `Verify that model "${modelId}" is available in your region and compartment.`,
    },
    {
      pattern: /InvalidParameter|validation error/i,
      message: 'Invalid parameter in request',
      hint: 'Check model-specific parameter limits (temperature, max_tokens, etc.).',
    },
    {
      pattern: /Authentication|Unauthorized|401/i,
      message: 'Authentication failed',
      hint: 'Check your OCI config profile and API key setup.',
    },
    {
      pattern: /InternalServerError|500|503/i,
      message: 'OCI service error',
      hint: 'This is an OCI-side issue. Try again or check OCI status page.',
    },
    {
      pattern: /context.*length|token.*limit|too long/i,
      message: 'Input exceeds model context length',
      hint: 'Reduce the size of your prompt or conversation history.',
    },
  ];
  
  // Find matching pattern
  for (const { pattern, message, hint } of errorPatterns) {
    const matches = typeof pattern === 'string' 
      ? originalMessage.includes(pattern)
      : pattern.test(originalMessage);
    
    if (matches) {
      let friendlyMessage = `[OCI GenAI] ${message}`;
      if (hint) {
        friendlyMessage += `\n  Hint: ${hint}`;
      }
      if (process.env.OCI_DEBUG) {
        friendlyMessage += `\n  Original error: ${originalMessage}`;
        if (opcRequestId) {
          friendlyMessage += `\n  Request ID: ${opcRequestId}`;
        }
      }
      return new Error(friendlyMessage);
    }
  }
  
  // No pattern matched - return enhanced generic error
  let genericMessage = `[OCI GenAI] API error for model "${modelId}"`;
  if (statusCode) {
    genericMessage += ` (HTTP ${statusCode})`;
  }
  genericMessage += `: ${originalMessage}`;
  
  if (process.env.OCI_DEBUG && opcRequestId) {
    genericMessage += `\n  Request ID: ${opcRequestId}`;
  }
  
  return new Error(genericMessage);
}

/**
 * Convert JSON Schema to Cohere parameter definitions
 */
function jsonSchemaToCohereparams(schema: JSONSchema7): Record<string, any> {
  const params: Record<string, any> = {};

  if (schema.type === 'object' && schema.properties) {
    const required = new Set(schema.required || []);

    for (const [name, propDef] of Object.entries(schema.properties)) {
      if (typeof propDef === 'boolean') continue;
      const propSchema = propDef as JSONSchema7;

      params[name] = {
        type: propSchema.type || 'string',
        description: propSchema.description || '',
        isRequired: required.has(name),
      };
    }
  }

  return params;
}

/**
 * Parse SSE (Server-Sent Events) stream from OCI GenAI
 * OCI returns text/event-stream format when isStream: true
 */
async function handleSSEStream(
  sseStream: ReadableStream<Uint8Array>,
  controller: ReadableStreamDefaultController<LanguageModelV2StreamPart>,
  modelFamily: string,
  swePreset: { supportsReasoning?: boolean },
  textId: string,
  reasoningId: string
): Promise<void> {
  const reader = sseStream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  
  // State tracking
  let textStarted = false;
  let reasoningStarted = false;
  let finishReason: LanguageModelV2FinishReason = 'stop';
  let promptTokens = 0;
  let completionTokens = 0;
  const toolCalls: Map<string, { toolName: string; input: string }> = new Map();
  const toolCallsStarted: Set<string> = new Set();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (!data || data === '[DONE]') continue;

          try {
            const event = JSON.parse(data);

            if (process.env.OCI_DEBUG) {
              console.error('[OCI Debug SSE Event]', JSON.stringify(event, null, 2));
            }

            // Handle different event types based on model family
            if (modelFamily === 'cohere-v2' || modelFamily === 'cohere') {
              await handleCohereSSEEvent(
                event,
                controller,
                { textId, reasoningId, textStarted, reasoningStarted, toolCalls, toolCallsStarted },
                (updates) => {
                  if (updates.textStarted !== undefined) textStarted = updates.textStarted;
                  if (updates.reasoningStarted !== undefined) reasoningStarted = updates.reasoningStarted;
                  if (updates.finishReason !== undefined) finishReason = updates.finishReason;
                  if (updates.promptTokens !== undefined) promptTokens = updates.promptTokens;
                  if (updates.completionTokens !== undefined) completionTokens = updates.completionTokens;
                }
              );
            } else {
              // Generic/Gemini format
              await handleGenericSSEEvent(
                event,
                controller,
                { textId, reasoningId, textStarted, reasoningStarted, toolCalls, toolCallsStarted },
                swePreset,
                (updates) => {
                  if (updates.textStarted !== undefined) textStarted = updates.textStarted;
                  if (updates.reasoningStarted !== undefined) reasoningStarted = updates.reasoningStarted;
                  if (updates.finishReason !== undefined) finishReason = updates.finishReason;
                  if (updates.promptTokens !== undefined) promptTokens = updates.promptTokens;
                  if (updates.completionTokens !== undefined) completionTokens = updates.completionTokens;
                }
              );
            }
          } catch (parseError) {
            if (process.env.OCI_DEBUG) {
              console.error('[OCI Debug SSE Parse Error]', parseError, 'Data:', data);
            }
          }
        }
      }
    }

    // End any open streams
    if (textStarted) {
      controller.enqueue({ type: 'text-end', id: textId });
    }
    if (reasoningStarted) {
      controller.enqueue({ type: 'reasoning-end', id: reasoningId });
    }

    // Emit completed tool calls
    for (const [toolCallId, { toolName, input }] of toolCalls) {
      if (!toolCallsStarted.has(toolCallId)) {
        controller.enqueue({ type: 'tool-input-start', id: toolCallId, toolName });
      }
      controller.enqueue({ type: 'tool-input-end', id: toolCallId });
      controller.enqueue({
        type: 'tool-call',
        toolCallId,
        toolName,
        input,
      });
    }

    // Determine final finish reason
    if (toolCalls.size > 0) {
      finishReason = 'tool-calls';
    }

    controller.enqueue({
      type: 'finish',
      finishReason,
      usage: createUsage(promptTokens, completionTokens),
    });
    controller.close();
  } finally {
    reader.releaseLock();
  }
}

/**
 * Handle Cohere-specific SSE events
 * OCI Cohere V2 streaming uses a direct message.content format rather than event types
 * Format: { apiFormat: "COHEREV2", message: { role: "ASSISTANT", content: [{ type: "TEXT", text: "..." }] } }
 */
async function handleCohereSSEEvent(
  event: any,
  controller: ReadableStreamDefaultController<LanguageModelV2StreamPart>,
  state: {
    textId: string;
    reasoningId: string;
    textStarted: boolean;
    reasoningStarted: boolean;
    toolCalls: Map<string, { toolName: string; input: string }>;
    toolCallsStarted: Set<string>;
  },
  updateState: (updates: {
    textStarted?: boolean;
    reasoningStarted?: boolean;
    finishReason?: LanguageModelV2FinishReason;
    promptTokens?: number;
    completionTokens?: number;
  }) => void
): Promise<void> {
  const eventType = event.type || event.eventType;

  // OCI Cohere V2 streaming format: direct message content without event types
  // Each SSE chunk contains: { message: { content: [{ type: "TEXT", text: "..." }] } }
  if (event.message?.content && Array.isArray(event.message.content)) {
    for (const contentPart of event.message.content) {
      if (contentPart.type === 'TEXT' && contentPart.text) {
        if (!state.textStarted) {
          controller.enqueue({ type: 'text-start', id: state.textId });
          updateState({ textStarted: true });
        }
        controller.enqueue({ type: 'text-delta', id: state.textId, delta: contentPart.text });
      }
      // Handle tool calls in streaming format
      if (contentPart.type === 'TOOL_CALL') {
        const toolId = contentPart.id || generateId();
        const toolName = contentPart.name || contentPart.function?.name || '';
        const input = JSON.stringify(contentPart.parameters || contentPart.function?.arguments || {});
        
        if (!state.toolCalls.has(toolId)) {
          state.toolCalls.set(toolId, { toolName, input });
          state.toolCallsStarted.add(toolId);
          controller.enqueue({ type: 'tool-input-start', id: toolId, toolName });
          controller.enqueue({ type: 'tool-input-delta', id: toolId, delta: input });
        }
      }
    }
  }
  
  // Handle tool calls from message.toolCalls array
  if (event.message?.toolCalls && Array.isArray(event.message.toolCalls)) {
    for (const tc of event.message.toolCalls) {
      const toolId = tc.id || generateId();
      const toolName = tc.name || tc.function?.name || '';
      const input = JSON.stringify(tc.parameters || tc.function?.arguments || {});
      
      if (!state.toolCalls.has(toolId)) {
        state.toolCalls.set(toolId, { toolName, input });
        state.toolCallsStarted.add(toolId);
        controller.enqueue({ type: 'tool-input-start', id: toolId, toolName });
        controller.enqueue({ type: 'tool-input-delta', id: toolId, delta: input });
      }
    }
  }

  // Handle usage in streaming events
  if (event.usage) {
    updateState({
      promptTokens: event.usage.inputTokens || event.usage.promptTokens || event.usage.billed_units?.input_tokens,
      completionTokens: event.usage.outputTokens || event.usage.completionTokens || event.usage.billed_units?.output_tokens,
    });
  }

  // Handle finish reason
  if (event.finishReason) {
    updateState({ finishReason: mapFinishReason(event.finishReason) });
  }

  // Also handle explicit event type format (fallback for other Cohere formats)
  switch (eventType) {
    case 'message-start':
      // Message start event - nothing to emit yet
      break;

    case 'content-start':
      // Content is starting
      if (!state.textStarted) {
        controller.enqueue({ type: 'text-start', id: state.textId });
        updateState({ textStarted: true });
      }
      break;

    case 'content-delta':
      // Text content delta
      const textDelta = event.delta?.message?.content?.text ||
                       event.delta?.text ||
                       event.text;
      if (textDelta) {
        if (!state.textStarted) {
          controller.enqueue({ type: 'text-start', id: state.textId });
          updateState({ textStarted: true });
        }
        controller.enqueue({ type: 'text-delta', id: state.textId, delta: textDelta });
      }
      break;

    case 'content-end':
      if (state.textStarted) {
        controller.enqueue({ type: 'text-end', id: state.textId });
        updateState({ textStarted: false });
      }
      break;

    case 'thinking-start':
    case 'reasoning-start':
      if (!state.reasoningStarted) {
        controller.enqueue({ type: 'reasoning-start', id: state.reasoningId });
        updateState({ reasoningStarted: true });
      }
      break;

    case 'thinking-delta':
    case 'reasoning-delta':
      const thinkingDelta = event.delta?.thinking || event.delta?.text || event.thinking;
      if (thinkingDelta) {
        if (!state.reasoningStarted) {
          controller.enqueue({ type: 'reasoning-start', id: state.reasoningId });
          updateState({ reasoningStarted: true });
        }
        controller.enqueue({ type: 'reasoning-delta', id: state.reasoningId, delta: thinkingDelta });
      }
      break;

    case 'thinking-end':
    case 'reasoning-end':
      if (state.reasoningStarted) {
        controller.enqueue({ type: 'reasoning-end', id: state.reasoningId });
        updateState({ reasoningStarted: false });
      }
      break;

    case 'tool-plan-delta':
      // Tool plan is reasoning about tool usage
      const planDelta = event.delta?.message?.tool_plan || event.delta?.tool_plan;
      if (planDelta) {
        if (!state.reasoningStarted) {
          controller.enqueue({ type: 'reasoning-start', id: state.reasoningId });
          updateState({ reasoningStarted: true });
        }
        controller.enqueue({ type: 'reasoning-delta', id: state.reasoningId, delta: planDelta });
      }
      break;

    case 'tool-call-start':
      // Tool call starting
      const toolCallData = event.delta?.message?.tool_calls || event.delta?.tool_calls;
      if (toolCallData) {
        const toolId = toolCallData.id || generateId();
        const toolName = toolCallData.function?.name || toolCallData.name || '';
        state.toolCalls.set(toolId, { toolName, input: '' });
        state.toolCallsStarted.add(toolId);
        controller.enqueue({ type: 'tool-input-start', id: toolId, toolName });
      }
      break;

    case 'tool-call-delta':
      // Tool call argument streaming
      const argsDelta = event.delta?.message?.tool_calls?.function?.arguments ||
                       event.delta?.tool_calls?.function?.arguments ||
                       event.delta?.arguments;
      if (argsDelta) {
        // Find the current tool call being built
        const currentToolId = Array.from(state.toolCalls.keys()).pop();
        if (currentToolId) {
          const tc = state.toolCalls.get(currentToolId);
          if (tc) {
            tc.input += argsDelta;
            controller.enqueue({ type: 'tool-input-delta', id: currentToolId, delta: argsDelta });
          }
        }
      }
      break;

    case 'tool-call-end':
      // Tool call completed - don't emit full tool-call yet, wait for message-end
      break;

    case 'message-end':
      // Extract usage and finish reason
      const usage = event.delta?.usage || event.usage;
      if (usage) {
        updateState({
          promptTokens: usage.billed_units?.input_tokens || usage.inputTokens || usage.promptTokens,
          completionTokens: usage.billed_units?.output_tokens || usage.outputTokens || usage.completionTokens,
        });
      }
      const reason = event.delta?.finish_reason || event.finish_reason || event.finishReason;
      if (reason) {
        updateState({ finishReason: mapFinishReason(reason) });
      }
      break;

    default:
      // Handle nested response structure (some events have response object)
      if (event.response || event.chatResponse) {
        const resp = event.response || event.chatResponse;
        // Check for text in response
        if (resp.text && !state.textStarted) {
          controller.enqueue({ type: 'text-start', id: state.textId });
          updateState({ textStarted: true });
          controller.enqueue({ type: 'text-delta', id: state.textId, delta: resp.text });
        }
        // Check for tool calls in response
        if (resp.toolCalls) {
          for (const tc of resp.toolCalls) {
            const toolId = tc.id || generateId();
            const toolName = tc.name || tc.function?.name || '';
            const input = JSON.stringify(tc.parameters || tc.function?.arguments || {});
            state.toolCalls.set(toolId, { toolName, input });
          }
        }
        // Check for finish reason
        if (resp.finishReason) {
          updateState({ finishReason: mapFinishReason(resp.finishReason) });
        }
      }
  }
}

/**
 * Handle Generic/Gemini SSE events
 * OCI Generic format also uses message.content[] structure similar to Cohere V2
 */
async function handleGenericSSEEvent(
  event: any,
  controller: ReadableStreamDefaultController<LanguageModelV2StreamPart>,
  state: {
    textId: string;
    reasoningId: string;
    textStarted: boolean;
    reasoningStarted: boolean;
    toolCalls: Map<string, { toolName: string; input: string }>;
    toolCallsStarted: Set<string>;
  },
  swePreset: { supportsReasoning?: boolean },
  updateState: (updates: {
    textStarted?: boolean;
    reasoningStarted?: boolean;
    finishReason?: LanguageModelV2FinishReason;
    promptTokens?: number;
    completionTokens?: number;
  }) => void
): Promise<void> {
  // OCI Generic/Gemini format: direct message content (same as Cohere V2)
  // Format: { message: { role: "ASSISTANT", content: [{ type: "TEXT", text: "..." }] } }
  if (event.message?.content && Array.isArray(event.message.content)) {
    for (const contentPart of event.message.content) {
      if (contentPart.type === 'TEXT' && contentPart.text) {
        if (!state.textStarted) {
          controller.enqueue({ type: 'text-start', id: state.textId });
          updateState({ textStarted: true });
        }
        controller.enqueue({ type: 'text-delta', id: state.textId, delta: contentPart.text });
      }
      // Handle tool calls in streaming format
      if (contentPart.type === 'TOOL_CALL' || contentPart.type === 'FUNCTION_CALL') {
        const toolId = contentPart.id || generateId();
        const toolName = contentPart.name || contentPart.function?.name || '';
        const input = JSON.stringify(contentPart.parameters || contentPart.args || contentPart.function?.arguments || {});
        
        if (!state.toolCalls.has(toolId)) {
          state.toolCalls.set(toolId, { toolName, input });
          state.toolCallsStarted.add(toolId);
          controller.enqueue({ type: 'tool-input-start', id: toolId, toolName });
          controller.enqueue({ type: 'tool-input-delta', id: toolId, delta: input });
        }
      }
    }
  }

  // Handle tool calls from message.toolCalls array (Gemini/Google format)
  // OCI Generic format returns: { message: { content: [...], toolCalls: [...] } }
  if (event.message?.toolCalls && Array.isArray(event.message.toolCalls)) {
    for (const tc of event.message.toolCalls) {
      const toolId = tc.id || generateId();
      const toolName = tc.name || tc.function?.name || '';
      // Arguments can be a string or an object
      const rawArgs = tc.arguments || tc.function?.arguments || tc.parameters || {};
      const input = typeof rawArgs === 'string' ? rawArgs : JSON.stringify(rawArgs);
      
      if (!state.toolCalls.has(toolId)) {
        state.toolCalls.set(toolId, { toolName, input });
        state.toolCallsStarted.add(toolId);
        controller.enqueue({ type: 'tool-input-start', id: toolId, toolName });
        controller.enqueue({ type: 'tool-input-delta', id: toolId, delta: input });
      }
    }
  }

  // Handle finish reason at event level
  if (event.finishReason) {
    updateState({ finishReason: mapFinishReason(event.finishReason) });
  }

  // Handle usage at event level
  if (event.usage) {
    updateState({
      promptTokens: event.usage.promptTokens || event.usage.inputTokens || event.usage.prompt_tokens,
      completionTokens: event.usage.completionTokens || event.usage.outputTokens || event.usage.completion_tokens,
    });
  }

  // Also handle standard OpenAI-style choices array format (fallback)
  const choices = event.choices || (event.chatResponse?.choices);
  if (choices && choices.length > 0) {
    const choice = choices[0];
    const delta = choice.delta || choice.message;

    // Text content
    if (delta?.content) {
      if (Array.isArray(delta.content)) {
        for (const part of delta.content) {
          if (part.type === 'TEXT' && part.text) {
            if (!state.textStarted) {
              controller.enqueue({ type: 'text-start', id: state.textId });
              updateState({ textStarted: true });
            }
            controller.enqueue({ type: 'text-delta', id: state.textId, delta: part.text });
          }
        }
      } else if (typeof delta.content === 'string') {
        if (!state.textStarted) {
          controller.enqueue({ type: 'text-start', id: state.textId });
          updateState({ textStarted: true });
        }
        controller.enqueue({ type: 'text-delta', id: state.textId, delta: delta.content });
      }
    }

    // Reasoning content
    if (delta?.reasoningContent && swePreset.supportsReasoning) {
      if (!state.reasoningStarted) {
        controller.enqueue({ type: 'reasoning-start', id: state.reasoningId });
        updateState({ reasoningStarted: true });
      }
      controller.enqueue({ type: 'reasoning-delta', id: state.reasoningId, delta: delta.reasoningContent });
    }

    // Tool calls
    const toolCalls = delta?.tool_calls || delta?.toolCalls || delta?.function_call;
    if (toolCalls) {
      const calls = Array.isArray(toolCalls) ? toolCalls : [toolCalls];
      for (const tc of calls) {
        const toolId = tc.id || tc.index?.toString() || generateId();
        const funcCall = tc.function || tc;
        const toolName = funcCall.name || tc.name;

        if (!state.toolCalls.has(toolId)) {
          state.toolCalls.set(toolId, { toolName: toolName || '', input: '' });
        }

        // Stream arguments
        const args = funcCall.arguments;
        if (args) {
          const existing = state.toolCalls.get(toolId)!;
          if (!state.toolCallsStarted.has(toolId)) {
            state.toolCallsStarted.add(toolId);
            controller.enqueue({ type: 'tool-input-start', id: toolId, toolName: existing.toolName || toolName });
          }
          existing.input += args;
          controller.enqueue({ type: 'tool-input-delta', id: toolId, delta: args });
        }
      }
    }

    // Finish reason
    if (choice.finish_reason || choice.finishReason) {
      updateState({ finishReason: mapFinishReason(choice.finish_reason || choice.finishReason) });
    }
  }

  // Usage info
  const usage = event.usage || event.chatResponse?.usage;
  if (usage) {
    updateState({
      promptTokens: usage.prompt_tokens || usage.promptTokens || usage.inputTokens,
      completionTokens: usage.completion_tokens || usage.completionTokens || usage.outputTokens,
    });
  }
}

/**
 * Handle non-streaming response by simulating streaming output
 * Used as fallback when true streaming is not available
 */
async function handleNonStreamingResponse(
  chatResult: any,
  controller: ReadableStreamDefaultController<LanguageModelV2StreamPart>,
  modelFamily: string,
  swePreset: { supportsReasoning?: boolean },
  textId: string,
  reasoningId: string
): Promise<void> {
  let text = '';
  let reasoningContent = '';
  let finishReason: LanguageModelV2FinishReason;
  let promptTokens = 0;
  let completionTokens = 0;
  const toolCalls: LanguageModelV2ToolCall[] = [];

  if (process.env.OCI_DEBUG) {
    console.error('[OCI Debug] Non-streaming Chat Result:', JSON.stringify(chatResult, null, 2));
    console.error('[OCI Debug] Model Family:', modelFamily);
  }

  if (modelFamily === 'cohere-v2') {
    const v2Response = chatResult?.chatResponse as any;
    const message = v2Response?.message;

    if (message?.content && Array.isArray(message.content)) {
      for (const part of message.content) {
        const partType = (part.type || '').toUpperCase();
        if (partType === 'TEXT' && part.text) {
          text += part.text;
        } else if (partType === 'THINKING' && part.thinking) {
          reasoningContent = part.thinking;
        }
      }
    }

    if (message?.toolCalls && Array.isArray(message.toolCalls)) {
      for (const toolCall of message.toolCalls) {
        const funcCall = toolCall.function || toolCall;
        const args = typeof funcCall.arguments === 'string'
          ? funcCall.arguments
          : JSON.stringify(funcCall.arguments || {});
        toolCalls.push({
          type: 'tool-call',
          toolCallId: toolCall.id || generateId(),
          toolName: funcCall.name || toolCall.name,
          input: args,
        });
      }
    }

    if (message?.toolPlan && swePreset.supportsReasoning) {
      reasoningContent = message.toolPlan + (reasoningContent ? '\n' + reasoningContent : '');
    }

    const hasToolCalls = toolCalls.length > 0;
    finishReason = hasToolCalls ? 'tool-calls' : mapFinishReason(v2Response?.finishReason);

    const usage = v2Response?.usage;
    promptTokens = usage?.promptTokens || usage?.inputTokens || 0;
    completionTokens = usage?.completionTokens || usage?.outputTokens || 0;
  } else if (modelFamily === 'cohere') {
    const cohereResponse = chatResult?.chatResponse as any;
    text = cohereResponse?.text || '';

    const cohereContent = cohereResponse?.content as any[] | undefined;
    if (cohereContent && Array.isArray(cohereContent)) {
      for (const part of cohereContent) {
        if (part.type === 'THINKING' && part.thinking) {
          reasoningContent = part.thinking;
        }
      }
    }

    const cohereToolCalls = cohereResponse?.toolCalls;
    if (cohereToolCalls && Array.isArray(cohereToolCalls)) {
      for (const toolCall of cohereToolCalls) {
        toolCalls.push({
          type: 'tool-call',
          toolCallId: generateId(),
          toolName: toolCall.name,
          input: JSON.stringify(toolCall.parameters || {}),
        });
      }
    }

    finishReason = toolCalls.length > 0 ? 'tool-calls' : mapFinishReason(cohereResponse?.finishReason);
  } else {
    // Generic format (Gemini, Llama, xAI, etc.)
    const genericResponse = chatResult?.chatResponse as any;
    const choice = genericResponse?.choices?.[0];
    const message = choice?.message;

    if (message?.content && Array.isArray(message.content)) {
      for (const part of message.content) {
        const partType = (part.type || '').toUpperCase();
        if (partType === 'TEXT' && part.text) {
          text += part.text;
        } else if (partType === 'TOOL_CALL' || partType === 'FUNCTION_CALL' || part.functionCall) {
          const toolPart = part as any;
          const funcCall = toolPart.functionCall || toolPart.function || toolPart;
          const args = typeof toolPart.arguments === 'string'
            ? toolPart.arguments
            : typeof funcCall.arguments === 'string'
            ? funcCall.arguments
            : JSON.stringify(toolPart.arguments || funcCall.arguments || funcCall.args || {});
          toolCalls.push({
            type: 'tool-call',
            toolCallId: toolPart.id || funcCall.id || generateId(),
            toolName: toolPart.name || funcCall.name,
            input: args,
          });
        }
      }
    } else if (typeof message?.content === 'string') {
      text = message.content;
    } else if (message?.text) {
      text = message.text;
    }

    const msgToolCalls = message?.tool_calls || message?.toolCalls || message?.function_call;
    if (msgToolCalls) {
      const calls = Array.isArray(msgToolCalls) ? msgToolCalls : [msgToolCalls];
      for (const tc of calls) {
        const funcCall = tc.function || tc;
        const args = typeof funcCall.arguments === 'string'
          ? funcCall.arguments
          : JSON.stringify(funcCall.arguments || funcCall.args || {});
        toolCalls.push({
          type: 'tool-call',
          toolCallId: tc.id || generateId(),
          toolName: funcCall.name || tc.name,
          input: args,
        });
      }
    }

    if (message?.reasoningContent && swePreset.supportsReasoning) {
      reasoningContent = message.reasoningContent;
    }

    finishReason = mapFinishReason(choice?.finishReason);
    const usageInfo = genericResponse?.usage;
    promptTokens = usageInfo?.promptTokens || 0;
    completionTokens = usageInfo?.completionTokens || 0;
  }

  // Emit reasoning content first (if present)
  if (reasoningContent) {
    controller.enqueue({ type: 'reasoning-start', id: reasoningId });
    const chunkSize = 50;
    for (let i = 0; i < reasoningContent.length; i += chunkSize) {
      controller.enqueue({
        type: 'reasoning-delta',
        id: reasoningId,
        delta: reasoningContent.slice(i, i + chunkSize),
      });
    }
    controller.enqueue({ type: 'reasoning-end', id: reasoningId });
  }

  // Emit text content
  if (text) {
    controller.enqueue({ type: 'text-start', id: textId });
    const chunkSize = 50;
    for (let i = 0; i < text.length; i += chunkSize) {
      controller.enqueue({
        type: 'text-delta',
        id: textId,
        delta: text.slice(i, i + chunkSize),
      });
    }
    controller.enqueue({ type: 'text-end', id: textId });
  }

  // Emit tool calls
  for (const toolCall of toolCalls) {
    controller.enqueue({
      type: 'tool-input-start',
      id: toolCall.toolCallId,
      toolName: toolCall.toolName,
    });
    controller.enqueue({
      type: 'tool-input-delta',
      id: toolCall.toolCallId,
      delta: toolCall.input,
    });
    controller.enqueue({
      type: 'tool-input-end',
      id: toolCall.toolCallId,
    });
    controller.enqueue(toolCall);
  }

  controller.enqueue({
    type: 'finish',
    finishReason,
    usage: createUsage(promptTokens, completionTokens),
  });
  controller.close();
}

class OCIChatLanguageModelV2 implements LanguageModelV2 {
  readonly specificationVersion = 'v2' as const;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private readonly client: oci.GenerativeAiInferenceClient;
  private readonly modelFamily: ModelFamily;
  private readonly swePreset: SWEPreset;

  constructor(
    readonly modelId: string,
    private readonly settings: OCIProviderSettings,
    private readonly isDedicatedEndpoint: boolean = false,
  ) {
    const provider = new common.ConfigFileAuthenticationDetailsProvider(
      undefined,
      settings.configProfile || 'DEFAULT'
    );
    this.client = new oci.GenerativeAiInferenceClient({
      authenticationDetailsProvider: provider,
    });

    if (settings.region) {
      this.client.region = common.Region.fromRegionId(settings.region);
    }

    this.modelFamily = getModelFamily(modelId);
    this.swePreset = getSWEPreset(modelId);
  }

  get provider(): string {
    return 'oci-genai';
  }

  async doGenerate(options: LanguageModelV2CallOptions): Promise<{
    content: LanguageModelV2Content[];
    finishReason: LanguageModelV2FinishReason;
    usage: LanguageModelV2Usage;
    warnings: LanguageModelV2CallWarning[];
  }> {
    const servingMode = this.getServingMode();
    const chatRequest = this.buildChatRequest(options);

    // Debug logging
    if (process.env.OCI_DEBUG) {
      console.error('[OCI Debug] Model:', this.modelId);
      console.error('[OCI Debug] Serving Mode:', JSON.stringify(servingMode));
      console.error('[OCI Debug] Chat Request:', JSON.stringify(chatRequest, null, 2));
    }

    const chatDetails: oci.models.ChatDetails = {
      compartmentId: this.settings.compartmentId || process.env.OCI_COMPARTMENT_ID || '',
      servingMode,
      chatRequest,
    };

    let response;
    try {
      response = await this.client.chat({ chatDetails });
    } catch (error) {
      throw parseOCIError(error, this.modelId);
    }

    if (!response || !('chatResult' in response)) {
      throw new Error('Unexpected response type from OCI GenAI');
    }

    const chatResult = response.chatResult;
    const content: LanguageModelV2Content[] = [];
    let finishReason: LanguageModelV2FinishReason;
    let promptTokens = 0;
    let completionTokens = 0;

    if (this.modelFamily === 'cohere-v2') {
      // Cohere V2 response format (CohereChatResponseV2)
      const v2Response = chatResult?.chatResponse as any;
      
      if (process.env.OCI_DEBUG) {
        console.error('[OCI Debug] Cohere V2 response:', JSON.stringify(v2Response, null, 2));
      }

      // V2 has response.message which contains content array and toolCalls
      const message = v2Response?.message;
      
      // Extract content from message.content array
      if (message?.content && Array.isArray(message.content)) {
        for (const part of message.content) {
          const partType = (part.type || '').toUpperCase();
          if (partType === 'TEXT' && part.text) {
            content.push({ type: 'text', text: part.text } as LanguageModelV2Text);
          } else if (partType === 'THINKING' && part.thinking) {
            content.push({ type: 'reasoning', text: part.thinking } as LanguageModelV2Reasoning);
          }
        }
      }

      // Extract tool calls from message.toolCalls (CohereToolCallV2 format)
      if (message?.toolCalls && Array.isArray(message.toolCalls)) {
        for (const toolCall of message.toolCalls) {
          // V2 format: { id, type: 'FUNCTION', function: { name, arguments } }
          const funcCall = toolCall.function || toolCall;
          const args = typeof funcCall.arguments === 'string'
            ? funcCall.arguments
            : JSON.stringify(funcCall.arguments || {});
          
          content.push({
            type: 'tool-call',
            toolCallId: toolCall.id || generateId(),
            toolName: funcCall.name || toolCall.name,
            input: args,
          } as LanguageModelV2ToolCall);
        }
      }

      // toolPlan contains the model's reasoning about tool usage
      if (message?.toolPlan && this.swePreset.supportsReasoning) {
        content.unshift({ type: 'reasoning', text: message.toolPlan } as LanguageModelV2Reasoning);
      }

      // finishReason is at the response level in V2
      const hasToolCalls = message?.toolCalls && message.toolCalls.length > 0;
      finishReason = hasToolCalls ? 'tool-calls' : mapFinishReason(v2Response?.finishReason);

      // Usage info
      const usage = v2Response?.usage;
      promptTokens = usage?.promptTokens || usage?.inputTokens || 0;
      completionTokens = usage?.completionTokens || usage?.outputTokens || 0;
    } else if (this.modelFamily === 'cohere') {
      const cohereResponse = chatResult?.chatResponse as oci.models.CohereChatResponse | undefined;
      const text = cohereResponse?.text || '';

      // Extract thinking content from Cohere V2 response (for reasoning models)
      const cohereContent = (cohereResponse as any)?.content as any[] | undefined;
      if (cohereContent && Array.isArray(cohereContent)) {
        for (const part of cohereContent) {
          if (part.type === 'THINKING' && part.thinking) {
            content.push({ type: 'reasoning', text: part.thinking } as LanguageModelV2Reasoning);
          }
        }
      }

      if (text) {
        content.push({ type: 'text', text } as LanguageModelV2Text);
      }

      // Handle Cohere tool calls
      const toolCalls = (cohereResponse as any)?.toolCalls;
      if (toolCalls && Array.isArray(toolCalls)) {
        for (const toolCall of toolCalls) {
          content.push({
            type: 'tool-call',
            toolCallId: generateId(),
            toolName: toolCall.name,
            input: JSON.stringify(toolCall.parameters || {}),
          } as LanguageModelV2ToolCall);
        }
      }

      finishReason = toolCalls?.length > 0 ? 'tool-calls' : mapFinishReason(cohereResponse?.finishReason);
    } else {
      const genericResponse = chatResult?.chatResponse as oci.models.GenericChatResponse | undefined;
      const choice = genericResponse?.choices?.[0];
      const message = choice?.message as oci.models.AssistantMessage | undefined;

      if (message?.content && Array.isArray(message.content)) {
        for (const part of message.content) {
          const partType = (part.type || '').toUpperCase();
          if (partType === 'TEXT') {
            const textPart = part as oci.models.TextContent;
            // Only add text content if there's actual text (Gemini may return empty TEXT objects with tool calls)
            if (textPart.text) {
              content.push({ type: 'text', text: textPart.text } as LanguageModelV2Text);
            }
          } else if (partType === 'TOOL_CALL' || partType === 'FUNCTION_CALL' || (part as any).functionCall) {
            const toolPart = part as any;
            // Handle various tool call formats (OCI generic, Gemini function_call)
            const funcCall = toolPart.functionCall || toolPart.function || toolPart;
            const args = typeof toolPart.arguments === 'string'
              ? toolPart.arguments
              : typeof funcCall.arguments === 'string'
              ? funcCall.arguments
              : JSON.stringify(toolPart.arguments || funcCall.arguments || funcCall.args || {});
            content.push({
              type: 'tool-call',
              toolCallId: toolPart.id || funcCall.id || generateId(),
              toolName: toolPart.name || funcCall.name,
              input: args,
            } as LanguageModelV2ToolCall);
          }
        }
      } else if (typeof message?.content === 'string') {
        // Fallback: message.content is a string directly
        content.push({ type: 'text', text: message.content } as LanguageModelV2Text);
      } else if ((message as any)?.text) {
        // Fallback: message has a text property directly
        content.push({ type: 'text', text: (message as any).text } as LanguageModelV2Text);
      }

      // Also check for tool_calls at the message level (some models put them there)
      const msgToolCalls = (message as any)?.tool_calls || (message as any)?.toolCalls || (message as any)?.function_call;
      if (msgToolCalls) {
        const calls = Array.isArray(msgToolCalls) ? msgToolCalls : [msgToolCalls];
        for (const tc of calls) {
          const funcCall = tc.function || tc;
          const args = typeof funcCall.arguments === 'string'
            ? funcCall.arguments
            : JSON.stringify(funcCall.arguments || funcCall.args || {});
          content.push({
            type: 'tool-call',
            toolCallId: tc.id || generateId(),
            toolName: funcCall.name || tc.name,
            input: args,
          } as LanguageModelV2ToolCall);
        }
      }

      // Extract reasoning content if present (for models that support reasoning)
      const reasoningContent = (message as any)?.reasoningContent;
      if (reasoningContent && this.swePreset.supportsReasoning) {
        content.unshift({ type: 'reasoning', text: reasoningContent } as LanguageModelV2Reasoning);
      }

      // Check if we have tool calls - if so, finishReason should be 'tool-calls'
      const hasToolCalls = content.some(c => c.type === 'tool-call');
      finishReason = hasToolCalls ? 'tool-calls' : mapFinishReason(choice?.finishReason);
      const usageInfo = genericResponse?.usage;
      promptTokens = usageInfo?.promptTokens || 0;
      completionTokens = usageInfo?.completionTokens || 0;
    }

    return {
      content,
      finishReason,
      usage: createUsage(promptTokens, completionTokens),
      warnings: [],
    };
  }

  async doStream(options: LanguageModelV2CallOptions): Promise<{
    stream: ReadableStream<LanguageModelV2StreamPart>;
  }> {
    const servingMode = this.getServingMode();
    const chatRequest = this.buildStreamingChatRequest(options);

    // Debug logging
    if (process.env.OCI_DEBUG) {
      console.error('[OCI Debug Stream] Model:', this.modelId);
      console.error('[OCI Debug Stream] Serving Mode:', JSON.stringify(servingMode));
      console.error('[OCI Debug Stream] Chat Request:', JSON.stringify(chatRequest, null, 2));
    }

    const chatDetails: oci.models.ChatDetails = {
      compartmentId: this.settings.compartmentId || process.env.OCI_COMPARTMENT_ID || '',
      servingMode,
      chatRequest,
    };

    const client = this.client;
    const modelFamily = this.modelFamily;
    const swePreset = this.swePreset;
    const modelId = this.modelId;

    const stream = new ReadableStream<LanguageModelV2StreamPart>({
      async start(controller) {
        const textId = generateId();
        const reasoningId = generateId();

        try {
          controller.enqueue({
            type: 'stream-start',
            warnings: [],
          });

          const response = await client.chat({ chatDetails });

          // Check if we got a streaming response (ReadableStream) or a non-streaming response
          if (response && typeof (response as any).getReader === 'function') {
            // True streaming response - parse SSE events
            await handleSSEStream(
              response as ReadableStream<Uint8Array>,
              controller,
              modelFamily,
              swePreset,
              textId,
              reasoningId
            );
            return;
          }

          // Non-streaming fallback (shouldn't happen with isStream: true, but handle gracefully)
          if (!response || !('chatResult' in response)) {
            controller.enqueue({
              type: 'error',
              error: new Error('Unexpected response type from OCI GenAI'),
            });
            controller.close();
            return;
          }

          // Handle non-streaming response by simulating streaming
          const chatResult = (response as any).chatResult;
          await handleNonStreamingResponse(
            chatResult,
            controller,
            modelFamily,
            swePreset,
            textId,
            reasoningId
          );
        } catch (error) {
          const parsedError = parseOCIError(error, modelId);
          controller.enqueue({ type: 'error', error: parsedError });
          controller.close();
        }
      },
    });

    return { stream };
  }

  /**
   * Build chat request with streaming enabled
   */
  private buildStreamingChatRequest(
    options: LanguageModelV2CallOptions
  ): oci.models.CohereChatRequest | oci.models.GenericChatRequest {
    const baseRequest = this.buildChatRequest(options);
    // Add streaming flag and options
    return {
      ...baseRequest,
      isStream: true,
      streamOptions: {
        isIncludeUsage: true,
      },
    } as any;
  }

  private getServingMode(): oci.models.OnDemandServingMode | oci.models.DedicatedServingMode {
    if (this.isDedicatedEndpoint) {
      return { servingType: 'DEDICATED', endpointId: this.modelId };
    }
    if (this.settings.servingMode === 'dedicated' && this.settings.endpointId) {
      return { servingType: 'DEDICATED', endpointId: this.settings.endpointId };
    }
    return { servingType: 'ON_DEMAND', modelId: this.modelId };
  }

  private buildChatRequest(
    options: LanguageModelV2CallOptions
  ): oci.models.CohereChatRequest | oci.models.GenericChatRequest {
    if (this.modelFamily === 'cohere-v2') {
      return this.buildCohereV2ChatRequest(options);
    }
    if (this.modelFamily === 'cohere') {
      return this.buildCohereChatRequest(options);
    }
    return this.buildGenericChatRequest(options);
  }

  /**
   * Apply SWE defaults when caller doesn't specify values
   */
  private applyDefaults<T>(value: T | undefined, defaultValue: T): T {
    return value !== undefined ? value : defaultValue;
  }

  private buildCohereChatRequest(options: LanguageModelV2CallOptions): oci.models.CohereChatRequest {
    const { message, chatHistory, toolResults } = this.convertMessagesToCohereFormat(options.prompt);

    // Convert tools to Cohere format
    const tools = this.swePreset.supportsTools && options.tools
      ? this.convertToolsToCohere(options.tools)
      : undefined;

    // Cohere requires isForceSingleStep=true when both message and toolResults are present
    const hasToolResults = toolResults && toolResults.length > 0;

    const request: any = {
      apiFormat: 'COHERE',
      message,
      chatHistory,
      maxTokens: options.maxOutputTokens,
      temperature: this.applyDefaults(options.temperature, this.swePreset.temperature),
      topP: this.applyDefaults(options.topP, this.swePreset.topP),
      frequencyPenalty: this.applyDefaults(options.frequencyPenalty, this.swePreset.frequencyPenalty),
      presencePenalty: this.applyDefaults(options.presencePenalty, this.swePreset.presencePenalty),
      ...(tools && { tools }),
      // Cohere SDK requires isForceSingleStep=true when both message and toolResults are present
      ...(hasToolResults && { toolResults, isForceSingleStep: true }),
    };

    // Add thinking parameter for Cohere reasoning models
    if (this.swePreset.supportsReasoning) {
      const providerOptions = options.providerOptions?.['oci-genai'] as Record<string, unknown> | undefined;
      const budgetTokens = providerOptions?.thinkingBudgetTokens as number | undefined;

      request.thinking = {
        type: 'ENABLED',
        ...(budgetTokens && { budgetTokens }),
      };
    }

    return request;
  }

  /**
   * Build Cohere V2 API request (for Command A models)
   * V2 uses a messages array format similar to OpenAI/Generic, but with Cohere-specific types
   */
  private buildCohereV2ChatRequest(options: LanguageModelV2CallOptions): any {
    const messages = this.convertMessagesToCohereV2Format(options.prompt);

    // Convert tools to Cohere V2 format (type: FUNCTION, function: {...})
    const tools = this.swePreset.supportsTools && options.tools
      ? this.convertToolsToCohereV2(options.tools)
      : undefined;

    // Map AI SDK toolChoice to Cohere V2 toolsChoice
    // Cohere V2 supports: 'REQUIRED' | 'NONE'
    // AI SDK supports: 'auto' | 'none' | 'required' | { type: 'tool', toolName: string }
    let toolsChoice: 'REQUIRED' | 'NONE' | undefined;
    if (tools) {
      const aiToolChoice = options.toolChoice;
      if (aiToolChoice?.type === 'none') {
        toolsChoice = 'NONE';
      } else if (aiToolChoice?.type === 'required' || aiToolChoice?.type === 'tool') {
        // 'required' and specific tool requests both map to REQUIRED
        toolsChoice = 'REQUIRED';
      }
      // For 'auto' or undefined, don't set toolsChoice - let the model decide
      // This is important for multi-turn: step 1 may need tools, but step 2
      // (after receiving tool results) should be free to respond with text
    }

    const request: any = {
      apiFormat: 'COHEREV2',
      messages,
      maxTokens: options.maxOutputTokens ?? 4096, // Default to 4096 if not specified
      temperature: this.applyDefaults(options.temperature, this.swePreset.temperature),
      topP: this.applyDefaults(options.topP, this.swePreset.topP),
      frequencyPenalty: this.applyDefaults(options.frequencyPenalty, this.swePreset.frequencyPenalty),
      presencePenalty: this.applyDefaults(options.presencePenalty, this.swePreset.presencePenalty),
      ...(tools && { tools }),
      ...(toolsChoice && { toolsChoice }),
    };

    // Add thinking parameter for Cohere reasoning models (Command A Reasoning)
    if (this.swePreset.supportsReasoning) {
      const providerOptions = options.providerOptions?.['oci-genai'] as Record<string, unknown> | undefined;
      const budgetTokens = providerOptions?.thinkingBudgetTokens as number | undefined;

      request.thinking = {
        type: 'ENABLED',
        ...(budgetTokens && { budgetTokens }),
      };
    }

    if (process.env.OCI_DEBUG) {
      console.error('[OCI Debug] Cohere V2 request:', JSON.stringify(request, null, 2));
    }

    return request;
  }

  private buildGenericChatRequest(options: LanguageModelV2CallOptions): oci.models.GenericChatRequest {
    const messages = this.convertMessagesToGenericFormat(options.prompt);

    // Build tools if provided and model supports them
    const tools = this.swePreset.supportsTools && options.tools
      ? this.convertTools(options.tools)
      : undefined;

    // Check if model supports stop sequences (defaults to true if not specified)
    const supportsStop = this.swePreset.supportsStopSequences !== false;

    // Base request
    const request: any = {
      apiFormat: 'GENERIC',
      messages,
      maxTokens: options.maxOutputTokens,
      temperature: this.applyDefaults(options.temperature, this.swePreset.temperature),
      topP: this.applyDefaults(options.topP, this.swePreset.topP),
      // Only include stop sequences if model supports them and they're provided
      ...(supportsStop && options.stopSequences && options.stopSequences.length > 0 && { stop: options.stopSequences }),
      ...(tools && { tools }),
    };

    // Only include penalty parameters for models that support them (e.g., not xAI/Grok)
    if (this.swePreset.supportsPenalties) {
      request.frequencyPenalty = this.applyDefaults(options.frequencyPenalty, this.swePreset.frequencyPenalty);
      request.presencePenalty = this.applyDefaults(options.presencePenalty, this.swePreset.presencePenalty);
    }

    // Include reasoningEffort for models that support reasoning API parameter
    // Note: xAI Grok models use model variant selection (grok-4-1-fast-reasoning vs non-reasoning)
    // instead of reasoningEffort parameter, so we skip it for xAI
    if (this.swePreset.supportsReasoning && !this.modelId.startsWith('xai.')) {
      const providerOptions = options.providerOptions?.['oci-genai'] as Record<string, unknown> | undefined;
      const reasoningEffort = providerOptions?.reasoningEffort as string | undefined;
      // Default to MEDIUM if not specified
      request.reasoningEffort = reasoningEffort || 'MEDIUM';
    }

    return request;
  }

  /**
   * Extract reasoning content from Generic API response
   */
  extractReasoningContent(genericResponse: any): string | undefined {
    const message = genericResponse?.choices?.[0]?.message;
    return message?.reasoningContent;
  }

  /**
   * Convert tools to Cohere format (CohereTool with CohereParameterDefinition)
   */
  private convertToolsToCohere(tools: LanguageModelV2CallOptions['tools']): any[] | undefined {
    if (!tools || tools.length === 0) return undefined;

    const cohereTools = tools
      .filter((tool): tool is LanguageModelV2FunctionTool => tool.type === 'function')
      .map(tool => {
        const paramDefs = jsonSchemaToCohereparams(tool.inputSchema as JSONSchema7);

        // Debug: log tool conversion
        if (process.env.OCI_DEBUG) {
          console.error('[OCI Debug] Converting tool:', tool.name);
          console.error('[OCI Debug] Input schema:', JSON.stringify(tool.inputSchema, null, 2));
          console.error('[OCI Debug] Cohere parameterDefinitions:', JSON.stringify(paramDefs, null, 2));
        }

        return {
          name: tool.name,
          description: tool.description || '',
          parameterDefinitions: paramDefs,
        };
      });

    return cohereTools;
  }

  /**
   * Convert tools to Cohere V2 format (CohereToolV2)
   * 
   * OCI CohereToolV2 format:
   * { type: 'FUNCTION', function: { name, description, parameters } }
   */
  private convertToolsToCohereV2(tools: LanguageModelV2CallOptions['tools']): any[] | undefined {
    if (!tools || tools.length === 0) return undefined;

    return tools
      .filter((tool): tool is LanguageModelV2FunctionTool => tool.type === 'function')
      .map(tool => {
        // Clean the JSON schema for compatibility
        const cleanedParams = this.cleanJsonSchema(tool.inputSchema);
        
        // Ensure parameters has type: 'object' (required by Cohere V2)
        const parameters = {
          type: 'object',
          ...cleanedParams,
        };

        // Debug: log tool conversion
        if (process.env.OCI_DEBUG) {
          console.error('[OCI Debug] Converting tool to Cohere V2:', tool.name);
          console.error('[OCI Debug] Original schema:', JSON.stringify(tool.inputSchema, null, 2));
          console.error('[OCI Debug] Final parameters:', JSON.stringify(parameters, null, 2));
        }

        // OCI CohereToolV2 format (from SDK types):
        // { type: 'FUNCTION', function: { name, description, parameters } }
        return {
          type: 'FUNCTION',
          function: {
            name: tool.name,
            description: tool.description || '',
            parameters,
          },
        };
      });
  }

  private convertTools(tools: LanguageModelV2CallOptions['tools']): any[] | undefined {
    if (!tools || tools.length === 0) return undefined;

    return tools
      .filter((tool): tool is LanguageModelV2FunctionTool => tool.type === 'function')
      .map(tool => {
        // Debug: log tool schema before cleaning
        if (process.env.OCI_DEBUG) {
          console.error('[OCI Debug] Converting generic tool:', tool.name);
          console.error('[OCI Debug] Raw inputSchema:', JSON.stringify(tool.inputSchema, null, 2));
        }
        
        const cleanedParams = this.cleanJsonSchema(tool.inputSchema);
        
        if (process.env.OCI_DEBUG) {
          console.error('[OCI Debug] Cleaned parameters:', JSON.stringify(cleanedParams, null, 2));
        }
        
        return {
          type: 'FUNCTION',
          name: tool.name,
          description: tool.description,
          parameters: cleanedParams,
        };
      });
  }

  /**
   * Convert a Zod v4 schema to JSON Schema if needed.
   * AI SDK v5 with Zod v4 may pass raw Zod schema objects instead of JSON Schema.
   * This method detects Zod schemas and converts them using Zod's built-in toJSONSchema.
   */
  private ensureJsonSchema(schema: any): any {
    if (!schema || typeof schema !== 'object') return schema;

    // Check if this looks like a Zod v4 schema (has _zod property or def.type)
    // Zod v4 schemas have a specific structure with _zod, def, type at top level
    const isZodSchema = (
      schema._zod !== undefined ||
      (schema.def && schema.def.type) ||
      (schema.type === 'object' && schema.shape)
    );

    if (isZodSchema) {
      try {
        // Dynamically import zod's toJSONSchema
        // This is a sync workaround - we detect if zod is available
        const zod = require('zod');
        if (zod.toJSONSchema) {
          const jsonSchema = zod.toJSONSchema(schema);
          if (process.env.OCI_DEBUG) {
            console.error('[OCI Debug] Converted Zod schema to JSON Schema:', JSON.stringify(jsonSchema, null, 2));
          }
          return jsonSchema;
        }
      } catch (e) {
        // zod not available or toJSONSchema not present, return as-is
        if (process.env.OCI_DEBUG) {
          console.error('[OCI Debug] Could not convert Zod schema:', e);
        }
      }
    }

    return schema;
  }

  /**
   * Clean JSON Schema for compatibility with Google Gemini and other models via OCI.
   * Removes unsupported properties that cause validation errors.
   * Based on common patterns from ai-sdk-provider-gemini-cli, claude-worker-proxy, etc.
   */
  private cleanJsonSchema(schema: any): any {
    // First ensure we have a JSON Schema (convert from Zod if needed)
    schema = this.ensureJsonSchema(schema);
    if (!schema || typeof schema !== 'object') return schema;

    // Keywords that Gemini and other models don't support as schema constraints.
    // Note: 'pattern' and 'format' are only removed when they're validation constraints,
    // not when they're property names in the properties object.
    const UNSUPPORTED_KEYWORDS = [
      '$schema',
      '$ref',
      'ref',
      '$defs',
      'definitions',
      '$id',
      '$comment',
      'additionalProperties',  // Gemini rejects this
      'propertyNames',
      'title',                 // Often rejected
      'examples',              // Often rejected
      'default',               // Often rejected
      'const',                 // Not widely supported
      // Note: 'format' intentionally NOT here - it could be a property name
      'minLength',
      'maxLength',
      // Note: 'pattern' intentionally NOT here - it could be a property name
      'minItems',
      'maxItems',
      'exclusiveMinimum',
      'exclusiveMaximum',
    ];

    const cleaned: any = {};
    for (const [key, value] of Object.entries(schema)) {
      // Skip unsupported keywords
      if (UNSUPPORTED_KEYWORDS.includes(key)) continue;

      // Special handling for 'pattern' - only remove if it's a regex constraint
      // (i.e., when it's a string value at the same level as type: 'string')
      if (key === 'pattern' && typeof value === 'string' && schema.type === 'string') {
        continue; // Skip pattern regex constraints
      }

      // Special handling for 'format' - only remove if it's a format constraint
      // (i.e., when it's a string value at the same level as type: 'string')
      // Preserve 'format' when it's a property name in tools like webfetch
      if (key === 'format' && typeof value === 'string' && schema.type === 'string') {
        continue; // Skip format constraints like "email", "uri", "date-time"
      }

      // Recursively clean nested objects
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        cleaned[key] = this.cleanJsonSchema(value);
      } else if (Array.isArray(value)) {
        cleaned[key] = value.map(item =>
          item && typeof item === 'object' ? this.cleanJsonSchema(item) : item
        );
      } else {
        cleaned[key] = value;
      }
    }
    return cleaned;
  }

  /**
   * Convert messages to Cohere format with chat history and tool results
   */
  private convertMessagesToCohereFormat(prompt: LanguageModelV2CallOptions['prompt']): {
    message: string;
    chatHistory: any[];
    toolResults: any[];
  } {
    if (!prompt || !Array.isArray(prompt)) {
      return { message: '', chatHistory: [], toolResults: [] };
    }

    const chatHistory: any[] = [];
    const toolResults: any[] = [];
    let systemPreamble = '';
    let lastUserMessage = '';

    for (let i = 0; i < prompt.length; i++) {
      const msg = prompt[i];

      if (msg.role === 'system') {
        systemPreamble = msg.content;
      } else if (msg.role === 'user') {
        // Get text from user message
        const text = msg.content
          .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
          .map(part => part.text)
          .join('\n');

        // If this is the last user message, save it as the main message
        if (i === prompt.length - 1 || !prompt.slice(i + 1).some(m => m.role === 'user')) {
          lastUserMessage = systemPreamble ? `${systemPreamble}\n\n${text}` : text;
        } else {
          chatHistory.push({
            role: 'USER',
            message: text,
          });
        }
      } else if (msg.role === 'assistant') {
        const text = msg.content
          .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
          .map(part => part.text)
          .join('\n');

        // Check for tool calls
        const toolCallParts = msg.content.filter(part => part.type === 'tool-call');

        if (toolCallParts.length > 0) {
          // Assistant message with tool calls
          chatHistory.push({
            role: 'CHATBOT',
            message: text,
            toolCalls: toolCallParts.map(part => ({
              name: (part as any).toolName,
              parameters: typeof (part as any).input === 'string'
                ? JSON.parse((part as any).input)
                : (part as any).input,
            })),
          });
        } else if (text) {
          chatHistory.push({
            role: 'CHATBOT',
            message: text,
          });
        }
      } else if (msg.role === 'tool') {
        // Tool results for Cohere
        for (const part of msg.content) {
          if (part.type === 'tool-result') {
            let resultValue: any;
            const output = part.output;

            // Handle undefined/null output gracefully
            if (!output) {
              resultValue = '';
            } else if (typeof output === 'string') {
              resultValue = output;
            } else if (output.type === 'text' || output.type === 'error-text') {
              resultValue = output.value ?? '';
            } else if (output.type === 'json') {
              resultValue = output.value;
            } else {
              resultValue = typeof output === 'object' ? JSON.stringify(output) : String(output);
            }

            // Find the corresponding tool call from previous assistant message
            const prevAssistant = [...chatHistory].reverse().find(m => m.role === 'CHATBOT' && m.toolCalls);
            const toolCall = prevAssistant?.toolCalls?.find((tc: any) => {
              // Match by looking at recent tool calls
              return true; // Cohere matches by order
            });

            if (toolCall) {
              toolResults.push({
                call: {
                  name: toolCall.name,
                  parameters: toolCall.parameters,
                },
                outputs: [typeof resultValue === 'string' ? { result: resultValue } : resultValue],
              });
            }
          }
        }
      }
    }

    return { message: lastUserMessage, chatHistory, toolResults };
  }

  /**
   * Convert messages to Cohere V2 format (CohereMessageV2 array)
   * V2 uses a messages array with role (SYSTEM, USER, ASSISTANT, TOOL) and content array
   */
  private convertMessagesToCohereV2Format(prompt: LanguageModelV2CallOptions['prompt']): any[] {
    if (!prompt || !Array.isArray(prompt)) {
      return [];
    }

    const messages: any[] = [];
    // Track tool call IDs for matching tool results
    const toolCallIdMap = new Map<string, string>();

    for (const msg of prompt) {
      if (msg.role === 'system') {
        messages.push({
          role: 'SYSTEM',
          content: [{ type: 'TEXT', text: msg.content }],
        });
      } else if (msg.role === 'user') {
        const content: any[] = [];
        for (const part of msg.content) {
          if (part.type === 'text') {
            content.push({ type: 'TEXT', text: part.text });
          }
          // Note: V2 also supports IMAGE content but we'll keep it simple for now
        }
        if (content.length > 0) {
          messages.push({
            role: 'USER',
            content,
          });
        }
      } else if (msg.role === 'assistant') {
        const content: any[] = [];
        const toolCalls: any[] = [];

        for (const part of msg.content) {
          if (part.type === 'text' && part.text) {
            content.push({ type: 'TEXT', text: part.text });
          } else if (part.type === 'tool-call') {
            const toolCallPart = part as any;
            const toolCallId = toolCallPart.toolCallId || generateId();
            
            // Store mapping for tool result matching
            toolCallIdMap.set(toolCallPart.toolName + '_' + JSON.stringify(toolCallPart.input), toolCallId);
            
            toolCalls.push({
              id: toolCallId,
              type: 'FUNCTION',
              function: {
                name: toolCallPart.toolName,
                arguments: typeof toolCallPart.input === 'string'
                  ? toolCallPart.input
                  : JSON.stringify(toolCallPart.input || {}),
              },
            });
          }
        }

        // In V2, toolCalls are part of the assistant message, not in content
        const assistantMsg: any = {
          role: 'ASSISTANT',
          content: content.length > 0 ? content : [{ type: 'TEXT', text: '' }],
        };
        if (toolCalls.length > 0) {
          assistantMsg.toolCalls = toolCalls;
        }
        messages.push(assistantMsg);
      } else if (msg.role === 'tool') {
        // V2 tool messages need toolCallId and content with the result
        for (const part of msg.content) {
          if (part.type === 'tool-result') {
            const toolResultPart = part as any;
            let resultText: string;
            const output = toolResultPart.output;

            if (!output) {
              resultText = '';
            } else if (typeof output === 'string') {
              resultText = output;
            } else if (output.type === 'text' || output.type === 'error-text') {
              resultText = output.value ?? '';
            } else if (output.type === 'json') {
              resultText = JSON.stringify(output.value);
            } else {
              resultText = typeof output === 'object' ? JSON.stringify(output) : String(output);
            }

            // Try to find the tool call ID from our map or use provided one
            const toolCallId = toolResultPart.toolCallId || 
              toolCallIdMap.get(toolResultPart.toolName + '_' + JSON.stringify(toolResultPart.input)) ||
              generateId();

            messages.push({
              role: 'TOOL',
              toolCallId,
              content: [{ type: 'TEXT', text: resultText }],
            });
          }
        }
      }
    }

    if (process.env.OCI_DEBUG) {
      console.error('[OCI Debug] Cohere V2 messages:', JSON.stringify(messages, null, 2));
    }

    return messages;
  }

  private convertMessagesToGenericFormat(prompt: LanguageModelV2CallOptions['prompt']): oci.models.Message[] {
    if (!prompt || !Array.isArray(prompt)) {
      return [];
    }

    // Check if this is an xAI model - they need special tool history handling
    const isXAI = this.modelId.startsWith('xai.');
    // Check if this is a Llama model - they also reject TOOL role messages
    const isLlama = this.modelId.startsWith('meta.llama');
    // Check if this is a Google model - they need FUNCTION_RESPONSE format for tool results
    const isGoogle = this.modelId.startsWith('google.');

    const messages: oci.models.Message[] = [];

    for (const message of prompt) {
      const role = message.role;

      if (role === 'system') {
        const textContent: oci.models.TextContent = {
          type: oci.models.TextContent.type,
          text: message.content,
        };
        messages.push({
          role: 'SYSTEM',
          content: [textContent],
        } as oci.models.SystemMessage);
      } else if (role === 'user') {
        const content: oci.models.ChatContent[] = [];

        for (const part of message.content) {
          if (part.type === 'text') {
            const textContent: oci.models.TextContent = {
              type: oci.models.TextContent.type,
              text: part.text,
            };
            content.push(textContent);
          } else if (part.type === 'file') {
            // Handle file/image content
            const fileData = part.data;
            if (typeof fileData === 'string') {
              // Base64 data
              content.push({
                type: 'IMAGE',
                source: {
                  type: 'BASE64',
                  mediaType: part.mediaType || 'image/png',
                  data: fileData,
                },
              } as any);
            }
          }
        }

        messages.push({
          role: 'USER',
          content,
        } as oci.models.UserMessage);
      } else if (role === 'assistant') {
        const content: oci.models.ChatContent[] = [];
        const toolCallTexts: string[] = [];
        const toolCalls: oci.models.ToolCall[] = []; // For Generic format (Google/Gemini)

        for (const part of message.content) {
          if (part.type === 'text') {
            const textContent: oci.models.TextContent = {
              type: oci.models.TextContent.type,
              text: part.text,
            };
            content.push(textContent);
          } else if (part.type === 'tool-call') {
            if (isXAI || isLlama) {
              // For xAI/Grok and Llama: Convert tool calls to text representation
              // These models reject TOOL_CALL content type in message history
              const args = typeof part.input === 'string' ? part.input : JSON.stringify(part.input);
              toolCallTexts.push(`[Called tool "${part.toolName}" with: ${args}]`);
            } else if (isGoogle) {
              // For Google/Gemini: Use toolCalls array at message level (not in content)
              // OCI Generic format expects FunctionCall objects in toolCalls array
              toolCalls.push({
                type: 'FUNCTION',
                id: part.toolCallId,
                name: part.toolName,
                arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input),
              } as oci.models.FunctionCall);
              // Also track for potential text fallback if parallel
              const args = typeof part.input === 'string' ? part.input : JSON.stringify(part.input);
              toolCallTexts.push(`[Called tool "${part.toolName}" (${part.toolCallId}) with: ${args}]`);
            } else {
              // Handle tool calls in assistant messages (standard format)
              content.push({
                type: 'TOOL_CALL',
                id: part.toolCallId,
                name: part.toolName,
                arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input),
              } as any);
            }
          }
        }

        // For xAI and Llama: append tool call descriptions as text
        if ((isXAI || isLlama) && toolCallTexts.length > 0) {
          const toolCallText = toolCallTexts.join('\n');
          // Add as text content or append to existing text
          if (content.length > 0 && content[0].type === oci.models.TextContent.type) {
            (content[0] as oci.models.TextContent).text += '\n' + toolCallText;
          } else {
            content.unshift({
              type: oci.models.TextContent.type,
              text: toolCallText,
            } as oci.models.TextContent);
          }
        }

        // Build the assistant message
        const assistantMsg: any = {
          role: 'ASSISTANT',
          content: content.length > 0 ? content : null, // Set to null if no content (per OCI docs)
        };

        // For Google/Gemini: Add toolCalls array at message level
        // But only for single tool calls - parallel calls need text fallback
        if (isGoogle && toolCalls.length === 1) {
          assistantMsg.toolCalls = toolCalls;
        } else if (isGoogle && toolCalls.length > 1) {
          // Parallel tool calls: use text fallback instead of toolCalls array
          // OCI's Generic format doesn't support parallel function calls properly for Gemini
          const toolCallText = toolCallTexts.join('\n');
          if (content.length > 0 && content[0].type === oci.models.TextContent.type) {
            (content[0] as oci.models.TextContent).text += '\n' + toolCallText;
          } else {
            content.unshift({
              type: oci.models.TextContent.type,
              text: toolCallText,
            } as oci.models.TextContent);
          }
          assistantMsg.content = content;
        }

        messages.push(assistantMsg as oci.models.AssistantMessage);
      } else if (role === 'tool') {
        // Handle tool result messages
        // For Google/Gemini: Collect all tool results to batch them
        const googleToolResults: Array<{ toolCallId: string; text: string; toolName: string }> = [];
        
        for (const part of message.content) {
          if (part.type === 'tool-result') {
            // V2 uses `output` with { type, value } structure
            let resultText: string;
            const output = part.output;

            // Handle undefined/null output gracefully
            if (!output) {
              resultText = '';
            } else if (typeof output === 'string') {
              // Raw string output
              resultText = output;
            } else if (output.type === 'text' || output.type === 'error-text') {
              resultText = output.value ?? '';
            } else if (output.type === 'json') {
              resultText = JSON.stringify(output.value);
            } else {
              // Fallback for unknown output types
              resultText = typeof output === 'object' ? JSON.stringify(output) : String(output);
            }

            if (isXAI || isLlama) {
              // For xAI/Grok and Llama: Convert tool results to USER messages
              // These models reject TOOL role messages
              const toolName = (part as any).toolName || 'tool';
              messages.push({
                role: 'USER',
                content: [{
                  type: oci.models.TextContent.type,
                  text: `[Tool result from "${toolName}": ${resultText}]`,
                } as oci.models.TextContent],
              } as oci.models.UserMessage);
            } else if (isGoogle) {
              // For Google/Gemini: Collect tool results to batch them
              const toolName = (part as any).toolName || 'tool';
              googleToolResults.push({ toolCallId: part.toolCallId, text: resultText, toolName });
            } else {
              const textContent: oci.models.TextContent = {
                type: oci.models.TextContent.type,
                text: resultText,
              };
              messages.push({
                role: 'TOOL',
                content: [textContent],
                toolCallId: part.toolCallId,
              } as any);
            }
          }
        }
        
        // For Google/Gemini: Create function response message(s)
        // OCI Generic format for Gemini has limitations with parallel tool calls
        if (isGoogle && googleToolResults.length > 0) {
          if (googleToolResults.length === 1) {
            // Single result: use standard OCI TOOL message format
            messages.push({
              role: 'TOOL',
              toolCallId: googleToolResults[0].toolCallId,
              content: [{
                type: oci.models.TextContent.type,
                text: googleToolResults[0].text,
              } as oci.models.TextContent],
            } as oci.models.ToolMessage);
          } else {
            // Multiple results: OCI's Generic format doesn't properly support parallel
            // function responses for Gemini. Combine all tool results into a single
            // USER message with all results as text parts.
            const combinedParts = googleToolResults.map(result => ({
              type: oci.models.TextContent.type,
              text: `[Tool result from "${result.toolName}" (${result.toolCallId}): ${result.text}]`,
            } as oci.models.TextContent));
            
            messages.push({
              role: 'USER',
              content: combinedParts,
            } as oci.models.UserMessage);
          }
        }
      }
    }

    return messages;
  }
}

export class OCIProvider implements ProviderV2 {
  private readonly settings: OCIProviderSettings;

  constructor(settings: OCIProviderSettings = {}) {
    this.settings = {
      compartmentId: settings.compartmentId || process.env.OCI_COMPARTMENT_ID,
      region: settings.region || process.env.OCI_REGION,
      configProfile: settings.configProfile || 'DEFAULT',
      servingMode: settings.servingMode || 'on-demand',
      endpointId: settings.endpointId || process.env.OCI_GENAI_ENDPOINT_ID,
    };
  }

  languageModel(modelId: string): LanguageModelV2 {
    const compartmentId = this.settings.compartmentId;
    if (!compartmentId) {
      throw new Error(
        'Missing compartment ID. Set OCI_COMPARTMENT_ID env var or pass compartmentId in options.'
      );
    }

    // Check if model requires dedicated cluster when using on-demand mode
    if (this.settings.servingMode === 'on-demand' && isDedicatedOnly(modelId)) {
      throw new Error(
        `${getModelDisplayName(modelId)} (${modelId}) requires a dedicated AI cluster and is not available in on-demand mode. ` +
        `To use this model, deploy it to a dedicated endpoint and configure the provider with servingMode: 'dedicated' and your endpointId.`
      );
    }

    return new OCIChatLanguageModelV2(modelId, this.settings, false);
  }

  textEmbeddingModel(_modelId: string): never {
    throw new Error('Text embedding models are not supported by OCI GenAI provider');
  }

  imageModel(_modelId: string): never {
    throw new Error('Image models are not supported by OCI GenAI provider');
  }

  getSettings(): Readonly<OCIProviderSettings> {
    return { ...this.settings };
  }
}

export function createOCI(settings: OCIProviderSettings = {}): OCIProvider {
  return new OCIProvider(settings);
}

// Default export: factory function that OpenCode will call
export default function createOCIProvider(options?: OCIProviderSettings) {
  return createOCI({
    compartmentId: options?.compartmentId || process.env.OCI_COMPARTMENT_ID,
    region: options?.region || process.env.OCI_REGION,
    ...options,
  });
}

export { createOCIProvider };
