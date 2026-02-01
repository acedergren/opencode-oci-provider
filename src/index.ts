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
}

const SWE_PRESETS: Record<string, SWEPreset> = {
  // Cohere models - good for instruction following, supports tools
  'cohere': {
    temperature: 0.2,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
  },
  // Google Gemini - excellent for code
  'google': {
    temperature: 0.1,
    topP: 0.95,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
  },
  // xAI Grok - does NOT support frequencyPenalty/presencePenalty
  'xai': {
    temperature: 0.1,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: false,
  },
  // Meta Llama - balanced for code
  'meta': {
    temperature: 0.2,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
  },
  // Default fallback
  'default': {
    temperature: 0.2,
    topP: 0.9,
    frequencyPenalty: 0,
    presencePenalty: 0,
    supportsTools: true,
    supportsPenalties: true,
  },
};

function getModelProvider(modelId: string): string {
  const prefix = modelId.split('.')[0];
  return prefix || 'default';
}

function getSWEPreset(modelId: string): SWEPreset {
  const provider = getModelProvider(modelId);
  return SWE_PRESETS[provider] || SWE_PRESETS['default'];
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
          if (part.type === 'TEXT') {
            const textPart = part as oci.models.TextContent;
            content.push({ type: 'text', text: textPart.text } as LanguageModelV2Text);
          } else if (part.type === 'TOOL_CALL') {
            const toolPart = part as any;
            const args = typeof toolPart.arguments === 'string'
              ? toolPart.arguments
              : JSON.stringify(toolPart.arguments || toolPart.function?.arguments || {});
            content.push({
              type: 'tool-call',
              toolCallId: toolPart.id || generateId(),
              toolName: toolPart.name || toolPart.function?.name,
              input: args,
            } as LanguageModelV2ToolCall);
          }
        }
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

    const stream = new ReadableStream<LanguageModelV2StreamPart>({
      async start(controller) {
        const textId = generateId();

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
          let finishReason: LanguageModelV2FinishReason;
          let promptTokens = 0;
          let completionTokens = 0;
          const toolCalls: LanguageModelV2ToolCall[] = [];

          // Debug: log the full response structure
          if (process.env.OCI_DEBUG) {
            console.error('[OCI Debug] Chat Result:', JSON.stringify(chatResult, null, 2));
          }

          if (modelFamily === 'cohere') {
            const cohereResponse = chatResult?.chatResponse as oci.models.CohereChatResponse | undefined;
            text = cohereResponse?.text || '';

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
            const choice = genericResponse?.choices?.[0];
            const message = choice?.message as oci.models.AssistantMessage | undefined;

            if (message?.content && Array.isArray(message.content)) {
              for (const part of message.content) {
                if (part.type === 'TEXT') {
                  const textPart = part as oci.models.TextContent;
                  text += textPart.text;
                } else if (part.type === 'TOOL_CALL') {
                  const toolPart = part as any;
                  const args = typeof toolPart.arguments === 'string'
                    ? toolPart.arguments
                    : JSON.stringify(toolPart.arguments || toolPart.function?.arguments || {});
                  toolCalls.push({
                    type: 'tool-call',
                    toolCallId: toolPart.id || generateId(),
                    toolName: toolPart.name || toolPart.function?.name,
                    input: args,
                  });
                }
              }
            }

            finishReason = mapFinishReason(choice?.finishReason);
            const usageInfo = genericResponse?.usage;
            promptTokens = usageInfo?.promptTokens || 0;
            completionTokens = usageInfo?.completionTokens || 0;
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
          for (const toolCall of toolCalls) {
            const toolInputId = generateId();
            // Stream tool input
            controller.enqueue({
              type: 'tool-input-start',
              id: toolInputId,
              toolName: toolCall.toolName,
            });
            controller.enqueue({
              type: 'tool-input-delta',
              id: toolInputId,
              delta: toolCall.input,
            });
            controller.enqueue({
              type: 'tool-input-end',
              id: toolInputId,
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

    return {
      apiFormat: oci.models.CohereChatRequest.apiFormat,
      message,
      chatHistory,
      maxTokens: options.maxOutputTokens,
      temperature: this.applyDefaults(options.temperature, this.swePreset.temperature),
      topP: this.applyDefaults(options.topP, this.swePreset.topP),
      frequencyPenalty: this.applyDefaults(options.frequencyPenalty, this.swePreset.frequencyPenalty),
      presencePenalty: this.applyDefaults(options.presencePenalty, this.swePreset.presencePenalty),
      ...(tools && { tools }),
      ...(toolResults && toolResults.length > 0 && { toolResults }),
    } as any;
  }

  private buildGenericChatRequest(options: LanguageModelV2CallOptions): oci.models.GenericChatRequest {
    const messages = this.convertMessagesToGenericFormat(options.prompt);

    // Build tools if provided and model supports them
    const tools = this.swePreset.supportsTools && options.tools
      ? this.convertTools(options.tools)
      : undefined;

    // Base request
    const request: any = {
      apiFormat: oci.models.GenericChatRequest.apiFormat,
      messages,
      maxTokens: options.maxOutputTokens,
      temperature: this.applyDefaults(options.temperature, this.swePreset.temperature),
      topP: this.applyDefaults(options.topP, this.swePreset.topP),
      stop: options.stopSequences,
      ...(tools && { tools }),
    };

    // Only include penalty parameters for models that support them (e.g., not xAI/Grok)
    if (this.swePreset.supportsPenalties) {
      request.frequencyPenalty = this.applyDefaults(options.frequencyPenalty, this.swePreset.frequencyPenalty);
      request.presencePenalty = this.applyDefaults(options.presencePenalty, this.swePreset.presencePenalty);
    }

    return request;
  }

  /**
   * Convert tools to Cohere format (CohereTool with CohereParameterDefinition)
   */
  private convertToolsToCohere(tools: LanguageModelV2CallOptions['tools']): any[] | undefined {
    if (!tools || tools.length === 0) return undefined;

    return tools
      .filter((tool): tool is LanguageModelV2FunctionTool => tool.type === 'function')
      .map(tool => ({
        name: tool.name,
        description: tool.description || '',
        parameterDefinitions: jsonSchemaToCohereparams(tool.inputSchema as JSONSchema7),
      }));
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
            if (part.output.type === 'text' || part.output.type === 'error-text') {
              resultValue = part.output.value;
            } else if (part.output.type === 'json') {
              resultValue = part.output.value;
            } else {
              resultValue = String(part.output);
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

        for (const part of message.content) {
          if (part.type === 'text') {
            const textContent: oci.models.TextContent = {
              type: oci.models.TextContent.type,
              text: part.text,
            };
            content.push(textContent);
          } else if (part.type === 'tool-call') {
            // Handle tool calls in assistant messages
            content.push({
              type: 'TOOL_CALL',
              id: part.toolCallId,
              name: part.toolName,
              arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input),
            } as any);
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
            if (part.output.type === 'text' || part.output.type === 'error-text') {
              resultText = part.output.value;
            } else if (part.output.type === 'json') {
              resultText = JSON.stringify(part.output.value);
            } else {
              resultText = String(part.output);
            }
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
