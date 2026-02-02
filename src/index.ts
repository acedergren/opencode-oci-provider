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
  // Gemini 2.5 Pro has thinking enabled by default (cannot be turned off)
  'google': {
    temperature: 0.1,
    topP: 0.95,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: false,
    supportsReasoning: true,  // Gemini 2.5 Pro has thinking always enabled
  },
  // xAI Grok - supports tools, but NOT frequencyPenalty/presencePenalty, stop sequences, or reasoning_effort
  // Note: OCI docs suggest reasoning_effort is available, but Grok models throw:
  // "This model does not support `reasoning_effort`"
  'xai': {
    temperature: 0.1,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: false,
    supportsStopSequences: false,  // Per OCI docs, stop sequences not listed as supported
    supportsReasoning: false,      // Grok throws error: "This model does not support `reasoning_effort`"
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
  // All Grok models through OCI use supportsReasoning=false (no reasoning_effort param)
  if (modelId.startsWith('xai.')) {
    return { ...basePreset, supportsReasoning: false };
  }

  // Cohere reasoning models (command-a-reasoning-*) support thinking
  if (modelId.includes('reasoning')) {
    return { ...basePreset, supportsReasoning: true };
  }

  // Meta Llama 4 models (maverick, scout) support reasoning
  if (modelId.includes('llama-4')) {
    return { ...basePreset, supportsReasoning: true };
  }

  return basePreset;
}

type ModelFamily = 'cohere' | 'generic';

function getModelFamily(modelId: string): ModelFamily {
  if (modelId.startsWith('cohere.')) {
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

    const response = await this.client.chat({ chatDetails });

    if (!response || !('chatResult' in response)) {
      throw new Error('Unexpected response type from OCI GenAI');
    }

    const chatResult = response.chatResult;
    const content: LanguageModelV2Content[] = [];
    let finishReason: LanguageModelV2FinishReason;
    let promptTokens = 0;
    let completionTokens = 0;

    if (this.modelFamily === 'cohere') {
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

      finishReason = mapFinishReason(choice?.finishReason);
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
    const chatRequest = this.buildChatRequest(options);

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

          if (!response || !('chatResult' in response)) {
            controller.enqueue({
              type: 'error',
              error: new Error('Unexpected response type from OCI GenAI'),
            });
            controller.close();
            return;
          }

          const chatResult = response.chatResult;
          let text = '';
          let reasoningContent = '';
          let finishReason: LanguageModelV2FinishReason;
          let promptTokens = 0;
          let completionTokens = 0;
          const toolCalls: LanguageModelV2ToolCall[] = [];

          // Debug: log the full response structure
          if (process.env.OCI_DEBUG) {
            console.error('[OCI Debug] Chat Result:', JSON.stringify(chatResult, null, 2));
            console.error('[OCI Debug] Model Family:', modelFamily);
          }

          if (modelFamily === 'cohere') {
            const cohereResponse = chatResult?.chatResponse as oci.models.CohereChatResponse | undefined;
            text = cohereResponse?.text || '';

            // Extract thinking content from Cohere V2 response (for reasoning models)
            const cohereContent = (cohereResponse as any)?.content as any[] | undefined;
            if (cohereContent && Array.isArray(cohereContent)) {
              for (const part of cohereContent) {
                if (part.type === 'THINKING' && part.thinking) {
                  reasoningContent = part.thinking;
                }
              }
            }

            // Handle Cohere tool calls
            const cohereToolCalls = (cohereResponse as any)?.toolCalls;
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
            const genericResponse = chatResult?.chatResponse as oci.models.GenericChatResponse | undefined;

            if (process.env.OCI_DEBUG) {
              console.error('[OCI Debug] Generic Response:', JSON.stringify(genericResponse, null, 2));
              console.error('[OCI Debug] Choices:', JSON.stringify(genericResponse?.choices, null, 2));
            }

            const choice = genericResponse?.choices?.[0];
            const message = choice?.message as oci.models.AssistantMessage | undefined;

            if (process.env.OCI_DEBUG) {
              console.error('[OCI Debug] Choice:', JSON.stringify(choice, null, 2));
              console.error('[OCI Debug] Message:', JSON.stringify(message, null, 2));
              console.error('[OCI Debug] Message content:', JSON.stringify(message?.content, null, 2));
            }

            if (message?.content && Array.isArray(message.content)) {
              for (const part of message.content) {
                const partType = (part.type || '').toUpperCase();
                if (partType === 'TEXT') {
                  const textPart = part as oci.models.TextContent;
                  // Only add text if it exists (Gemini may return empty TEXT objects with tool calls)
                  if (textPart.text) {
                    text += textPart.text;
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
                  toolCalls.push({
                    type: 'tool-call',
                    toolCallId: toolPart.id || funcCall.id || generateId(),
                    toolName: toolPart.name || funcCall.name,
                    input: args,
                  });
                }
              }
            } else if (typeof message?.content === 'string') {
              // Fallback: message.content is a string directly
              text = message.content;
            } else if ((message as any)?.text) {
              // Fallback: message has a text property directly
              text = (message as any).text;
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
                toolCalls.push({
                  type: 'tool-call',
                  toolCallId: tc.id || generateId(),
                  toolName: funcCall.name || tc.name,
                  input: args,
                });
              }
            }

            // Extract reasoning content if present
            if ((message as any)?.reasoningContent && swePreset.supportsReasoning) {
              reasoningContent = (message as any).reasoningContent;
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

          // Emit tool calls using V2 format (tool-input-* events, then full tool-call)
          // IMPORTANT: The `id` in tool-input-* events must match toolCallId in the final tool-call
          for (const toolCall of toolCalls) {
            // Stream tool input using the SAME ID as the final tool-call
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
            // Emit complete tool call
            controller.enqueue(toolCall);
          }

          controller.enqueue({
            type: 'finish',
            finishReason,
            usage: createUsage(promptTokens, completionTokens),
          });
          controller.close();
        } catch (error) {
          controller.enqueue({ type: 'error', error });
          controller.close();
        }
      },
    });

    return { stream };
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
      apiFormat: oci.models.CohereChatRequest.apiFormat,
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
      apiFormat: oci.models.GenericChatRequest.apiFormat,
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

    // Include reasoningEffort for models that support reasoning
    if (this.swePreset.supportsReasoning) {
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

  private convertTools(tools: LanguageModelV2CallOptions['tools']): any[] | undefined {
    if (!tools || tools.length === 0) return undefined;

    return tools
      .filter((tool): tool is LanguageModelV2FunctionTool => tool.type === 'function')
      .map(tool => ({
        type: 'FUNCTION',
        name: tool.name,
        description: tool.description,
        parameters: this.cleanJsonSchema(tool.inputSchema),
      }));
  }

  /**
   * Clean JSON Schema for compatibility with Google Gemini via OCI.
   * Removes $schema, ref, and other unsupported properties.
   */
  private cleanJsonSchema(schema: any): any {
    if (!schema || typeof schema !== 'object') return schema;

    const cleaned: any = {};
    for (const [key, value] of Object.entries(schema)) {
      // Skip properties Gemini doesn't support
      if (key === '$schema' || key === 'ref' || key === '$ref') continue;

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

  private convertMessagesToGenericFormat(prompt: LanguageModelV2CallOptions['prompt']): oci.models.Message[] {
    if (!prompt || !Array.isArray(prompt)) {
      return [];
    }

    // Check if this is an xAI model - they need special tool history handling
    const isXAI = this.modelId.startsWith('xai.');

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

        for (const part of message.content) {
          if (part.type === 'text') {
            const textContent: oci.models.TextContent = {
              type: oci.models.TextContent.type,
              text: part.text,
            };
            content.push(textContent);
          } else if (part.type === 'tool-call') {
            if (isXAI) {
              // For xAI/Grok: Convert tool calls to text representation
              // Grok rejects TOOL_CALL content type in message history
              const args = typeof part.input === 'string' ? part.input : JSON.stringify(part.input);
              toolCallTexts.push(`[Called tool "${part.toolName}" with: ${args}]`);
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

        // For xAI: append tool call descriptions as text
        if (isXAI && toolCallTexts.length > 0) {
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

        messages.push({
          role: 'ASSISTANT',
          content,
        } as oci.models.AssistantMessage);
      } else if (role === 'tool') {
        // Handle tool result messages
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

            if (isXAI) {
              // For xAI/Grok: Convert tool results to USER messages
              // Grok rejects TOOL role messages
              const toolName = (part as any).toolName || 'tool';
              messages.push({
                role: 'USER',
                content: [{
                  type: oci.models.TextContent.type,
                  text: `[Tool result from "${toolName}": ${resultText}]`,
                } as oci.models.TextContent],
              } as oci.models.UserMessage);
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
