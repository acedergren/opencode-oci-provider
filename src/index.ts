/**
 * OpenCode OCI Provider - AI SDK V2 compatible provider for OCI GenAI
 *
 * This package provides a LanguageModelV2 implementation for OpenCode,
 * which uses AI SDK v2 interfaces internally.
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
  ProviderV2,
  NoSuchModelError,
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

class OCIChatLanguageModelV2 implements LanguageModelV2 {
  readonly specificationVersion = 'v2' as const;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private readonly client: oci.GenerativeAiInferenceClient;
  private readonly modelFamily: ModelFamily;

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
    let text = '';
    let finishReason: LanguageModelV2FinishReason;
    let promptTokens = 0;
    let completionTokens = 0;

    if (this.modelFamily === 'cohere') {
      const cohereResponse = chatResult?.chatResponse as oci.models.CohereChatResponse | undefined;
      text = cohereResponse?.text || '';
      finishReason = mapFinishReason(cohereResponse?.finishReason);
    } else {
      const genericResponse = chatResult?.chatResponse as oci.models.GenericChatResponse | undefined;
      const choice = genericResponse?.choices?.[0];
      const message = choice?.message as oci.models.AssistantMessage | undefined;

      if (message?.content && Array.isArray(message.content)) {
        text = message.content
          .filter((c): c is oci.models.TextContent => c.type === 'TEXT')
          .map(c => c.text)
          .join('');
      }

      finishReason = mapFinishReason(choice?.finishReason);
      const usageInfo = genericResponse?.usage;
      promptTokens = usageInfo?.promptTokens || 0;
      completionTokens = usageInfo?.completionTokens || 0;
    }

    const content: LanguageModelV2Content[] = [];
    if (text) {
      content.push({ type: 'text', text } as LanguageModelV2Text);
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
          // Emit stream-start
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

          if (modelFamily === 'cohere') {
            const cohereResponse = chatResult?.chatResponse as oci.models.CohereChatResponse | undefined;
            text = cohereResponse?.text || '';
            finishReason = mapFinishReason(cohereResponse?.finishReason);
          } else {
            const genericResponse = chatResult?.chatResponse as oci.models.GenericChatResponse | undefined;
            const choice = genericResponse?.choices?.[0];
            const message = choice?.message as oci.models.AssistantMessage | undefined;

            if (message?.content && Array.isArray(message.content)) {
              text = message.content
                .filter((c): c is oci.models.TextContent => c.type === 'TEXT')
                .map(c => c.text)
                .join('');
            }

            finishReason = mapFinishReason(choice?.finishReason);
            const usageInfo = genericResponse?.usage;
            promptTokens = usageInfo?.promptTokens || 0;
            completionTokens = usageInfo?.completionTokens || 0;
          }

          // Emit text as deltas (V2 format with id)
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

          // Emit finish (V2 format)
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

  private buildCohereChatRequest(options: LanguageModelV2CallOptions): oci.models.CohereChatRequest {
    const prompt = this.convertMessagesToCoherePrompt(options.prompt);

    return {
      apiFormat: oci.models.CohereChatRequest.apiFormat,
      message: prompt,
      maxTokens: options.maxOutputTokens,
      temperature: options.temperature,
      topP: options.topP,
      frequencyPenalty: options.frequencyPenalty,
      presencePenalty: options.presencePenalty,
    };
  }

  private buildGenericChatRequest(options: LanguageModelV2CallOptions): oci.models.GenericChatRequest {
    const messages = this.convertMessagesToGenericFormat(options.prompt);

    return {
      apiFormat: oci.models.GenericChatRequest.apiFormat,
      messages,
      maxTokens: options.maxOutputTokens,
      temperature: options.temperature,
      topP: options.topP,
      frequencyPenalty: options.frequencyPenalty,
      presencePenalty: options.presencePenalty,
      stop: options.stopSequences,
    };
  }

  private convertMessagesToCoherePrompt(prompt: LanguageModelV2CallOptions['prompt']): string {
    if (!prompt || !Array.isArray(prompt)) {
      return '';
    }

    return prompt.map(message => {
      const role = message.role;

      if (role === 'system') {
        return `system: ${message.content}`;
      }

      if (role === 'user') {
        const textParts = message.content
          .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
          .map(part => part.text);
        return `user: ${textParts.join('\n')}`;
      }

      if (role === 'assistant') {
        const textParts = message.content
          .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
          .map(part => part.text);
        return `assistant: ${textParts.join('\n')}`;
      }

      if (role === 'tool') {
        return `tool: ${JSON.stringify(message.content)}`;
      }

      return '';
    }).filter(Boolean).join('\n');
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
          }
        }

        messages.push({
          role: 'ASSISTANT',
          content,
        } as oci.models.AssistantMessage);
      } else if (role === 'tool') {
        const textContent: oci.models.TextContent = {
          type: oci.models.TextContent.type,
          text: JSON.stringify(message.content),
        };
        messages.push({
          role: 'TOOL',
          content: [textContent],
        } as oci.models.ToolMessage);
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
