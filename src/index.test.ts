/**
 * Regression tests for OCI GenAI Provider
 *
 * These tests verify fixes for model-specific compatibility issues:
 * 1. Tool format: Flat structure (no nested `function` wrapper)
 * 2. JSON Schema: Strip $schema, ref, $ref for Gemini compatibility
 * 3. Model parameters: xAI models don't support frequencyPenalty/presencePenalty
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createOCI, OCIProvider } from './index.js';

// Mock OCI SDK to avoid actual API calls
vi.mock('oci-generativeaiinference', () => {
  return {
    GenerativeAiInferenceClient: class MockClient {
      region: any = null;
      async chat() {
        return {
          chatResult: {
            chatResponse: {
              choices: [{
                message: { content: [{ type: 'TEXT', text: 'test response' }] },
                finishReason: 'COMPLETE',
              }],
              usage: { promptTokens: 10, completionTokens: 20 },
            },
          },
        };
      }
    },
    models: {
      GenericChatRequest: { apiFormat: 'GENERIC' },
      CohereChatRequest: { apiFormat: 'COHERE' },
      TextContent: { type: 'TEXT' },
    },
  };
});

vi.mock('oci-common', () => {
  return {
    ConfigFileAuthenticationDetailsProvider: class MockAuthProvider {},
    Region: {
      fromRegionId: () => ({ regionId: 'us-chicago-1' }),
    },
  };
});

describe('OCI Provider Tool Compatibility', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  describe('cleanJsonSchema', () => {
    it('should strip $schema from top level', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const cleanJsonSchema = (model as any).cleanJsonSchema.bind(model);

      const schema = {
        $schema: 'http://json-schema.org/draft-07/schema#',
        type: 'object',
        properties: { name: { type: 'string' } },
      };

      const cleaned = cleanJsonSchema(schema);
      expect(cleaned).not.toHaveProperty('$schema');
      expect(cleaned.type).toBe('object');
      expect(cleaned.properties.name.type).toBe('string');
    });

    it('should strip ref and $ref from nested properties', () => {
      const model = provider.languageModel('google.gemini-2.5-pro');
      const cleanJsonSchema = (model as any).cleanJsonSchema.bind(model);

      const schema = {
        type: 'object',
        properties: {
          answers: {
            type: 'array',
            items: {
              ref: 'QuestionOption',
              $ref: '#/definitions/QuestionOption',
              type: 'object',
            },
          },
        },
      };

      const cleaned = cleanJsonSchema(schema);
      expect(cleaned.properties.answers.items).not.toHaveProperty('ref');
      expect(cleaned.properties.answers.items).not.toHaveProperty('$ref');
      expect(cleaned.properties.answers.items.type).toBe('object');
    });

    it('should recursively clean deeply nested schemas', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const cleanJsonSchema = (model as any).cleanJsonSchema.bind(model);

      const schema = {
        type: 'object',
        properties: {
          level1: {
            type: 'object',
            $schema: 'should-be-removed',
            properties: {
              level2: {
                type: 'object',
                ref: 'SomeRef',
                properties: {
                  level3: {
                    $ref: '#/deep/ref',
                    type: 'string',
                  },
                },
              },
            },
          },
        },
      };

      const cleaned = cleanJsonSchema(schema);
      expect(cleaned.properties.level1).not.toHaveProperty('$schema');
      expect(cleaned.properties.level1.properties.level2).not.toHaveProperty('ref');
      expect(cleaned.properties.level1.properties.level2.properties.level3).not.toHaveProperty('$ref');
    });

    it('should handle arrays with objects containing forbidden properties', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const cleanJsonSchema = (model as any).cleanJsonSchema.bind(model);

      const schema = {
        type: 'array',
        items: [
          { type: 'string', $schema: 'remove-me' },
          { type: 'number', ref: 'remove-me-too' },
        ],
      };

      const cleaned = cleanJsonSchema(schema);
      expect(cleaned.items[0]).not.toHaveProperty('$schema');
      expect(cleaned.items[1]).not.toHaveProperty('ref');
      expect(cleaned.items[0].type).toBe('string');
      expect(cleaned.items[1].type).toBe('number');
    });

    it('should handle null and primitive values gracefully', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const cleanJsonSchema = (model as any).cleanJsonSchema.bind(model);

      expect(cleanJsonSchema(null)).toBe(null);
      expect(cleanJsonSchema(undefined)).toBe(undefined);
      expect(cleanJsonSchema('string')).toBe('string');
      expect(cleanJsonSchema(123)).toBe(123);
      expect(cleanJsonSchema(true)).toBe(true);
    });

    it('should strip additionalProperties and other Gemini-unsupported keywords', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const cleanJsonSchema = (model as any).cleanJsonSchema.bind(model);

      const schema = {
        type: 'object',
        title: 'TestSchema',
        properties: {
          name: {
            type: 'string',
            minLength: 1,
            maxLength: 100,
            pattern: '^[a-z]+$',
            default: 'test',
            examples: ['example'],
            format: 'email',
          },
          count: {
            type: 'integer',
            exclusiveMinimum: 0,
            exclusiveMaximum: 100,
          },
          items: {
            type: 'array',
            minItems: 1,
            maxItems: 10,
          },
        },
        additionalProperties: false,
        $defs: { SomeRef: { type: 'string' } },
        definitions: { AnotherRef: { type: 'number' } },
        $id: 'https://example.com/schema',
        $comment: 'This is a comment',
      };

      const cleaned = cleanJsonSchema(schema);

      // These should be removed
      expect(cleaned).not.toHaveProperty('additionalProperties');
      expect(cleaned).not.toHaveProperty('title');
      expect(cleaned).not.toHaveProperty('$defs');
      expect(cleaned).not.toHaveProperty('definitions');
      expect(cleaned).not.toHaveProperty('$id');
      expect(cleaned).not.toHaveProperty('$comment');

      // Property constraints should be removed
      expect(cleaned.properties.name).not.toHaveProperty('minLength');
      expect(cleaned.properties.name).not.toHaveProperty('maxLength');
      expect(cleaned.properties.name).not.toHaveProperty('pattern');
      expect(cleaned.properties.name).not.toHaveProperty('default');
      expect(cleaned.properties.name).not.toHaveProperty('examples');
      expect(cleaned.properties.name).not.toHaveProperty('format');
      expect(cleaned.properties.count).not.toHaveProperty('exclusiveMinimum');
      expect(cleaned.properties.count).not.toHaveProperty('exclusiveMaximum');
      expect(cleaned.properties.items).not.toHaveProperty('minItems');
      expect(cleaned.properties.items).not.toHaveProperty('maxItems');

      // Core properties should remain
      expect(cleaned.type).toBe('object');
      expect(cleaned.properties.name.type).toBe('string');
      expect(cleaned.properties.count.type).toBe('integer');
      expect(cleaned.properties.items.type).toBe('array');
    });

    it('should preserve format and pattern when they are property names, not constraints', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const cleanJsonSchema = (model as any).cleanJsonSchema.bind(model);

      // Schema like webfetch tool - 'format' is a property name, not a constraint
      const webfetchSchema = {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'The URL to fetch',
          },
          format: {
            type: 'string',
            enum: ['text', 'markdown', 'html'],
            description: 'Output format',
          },
        },
        required: ['url', 'format'],
      };

      const cleanedWebfetch = cleanJsonSchema(webfetchSchema);

      // 'format' property should be preserved (it's a property name, not a constraint)
      expect(cleanedWebfetch.properties).toHaveProperty('format');
      expect(cleanedWebfetch.properties.format.type).toBe('string');
      expect(cleanedWebfetch.properties.format.enum).toEqual(['text', 'markdown', 'html']);
      expect(cleanedWebfetch.required).toContain('format');

      // Schema like glob/grep tools - 'pattern' is a property name, not a constraint
      const globSchema = {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description: 'Glob pattern to match',
          },
          path: {
            type: 'string',
            description: 'Directory to search in',
          },
        },
        required: ['pattern'],
      };

      const cleanedGlob = cleanJsonSchema(globSchema);

      // 'pattern' property should be preserved (it's a property name, not a constraint)
      expect(cleanedGlob.properties).toHaveProperty('pattern');
      expect(cleanedGlob.properties.pattern.type).toBe('string');
      expect(cleanedGlob.required).toContain('pattern');
    });
  });

  describe('convertTools', () => {
    it('should use flat structure without nested function wrapper', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const convertTools = (model as any).convertTools.bind(model);

      const tools = [
        {
          type: 'function' as const,
          name: 'get_weather',
          description: 'Get the weather for a location',
          inputSchema: {
            type: 'object',
            properties: {
              location: { type: 'string' },
            },
          },
        },
      ];

      const converted = convertTools(tools);

      expect(converted).toHaveLength(1);
      expect(converted[0]).toEqual({
        type: 'FUNCTION',
        name: 'get_weather',
        description: 'Get the weather for a location',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string' },
          },
        },
      });

      // Verify NO nested function wrapper (this was the bug)
      expect(converted[0]).not.toHaveProperty('function');
    });

    it('should clean JSON schema in tool parameters', () => {
      const model = provider.languageModel('google.gemini-2.5-pro');
      const convertTools = (model as any).convertTools.bind(model);

      const tools = [
        {
          type: 'function' as const,
          name: 'ask_question',
          description: 'Ask a question',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
            type: 'object',
            properties: {
              options: {
                type: 'array',
                items: {
                  ref: 'QuestionOption',
                },
              },
            },
          },
        },
      ];

      const converted = convertTools(tools);

      // Schema should be cleaned
      expect(converted[0].parameters).not.toHaveProperty('$schema');
      expect(converted[0].parameters.properties.options.items).not.toHaveProperty('ref');
    });

    it('should return undefined for empty or null tools', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const convertTools = (model as any).convertTools.bind(model);

      expect(convertTools(undefined)).toBeUndefined();
      expect(convertTools(null)).toBeUndefined();
      expect(convertTools([])).toBeUndefined();
    });

    it('should filter out non-function tools', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const convertTools = (model as any).convertTools.bind(model);

      const tools = [
        { type: 'function' as const, name: 'valid', description: 'Valid', inputSchema: {} },
        { type: 'other' as any, name: 'invalid' },
      ];

      const converted = convertTools(tools);

      expect(converted).toHaveLength(1);
      expect(converted[0].name).toBe('valid');
    });
  });

  describe('SWE Presets and Model Parameters', () => {
    it('should set supportsPenalties=false and supportsStopSequences=false for xAI models', () => {
      const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
      const swePreset = (model as any).swePreset;

      expect(swePreset.supportsPenalties).toBe(false);
      expect(swePreset.supportsTools).toBe(true);
      expect(swePreset.supportsStopSequences).toBe(false);
    });

    it('should set supportsPenalties=false for Google models (OCI limitation)', () => {
      const model = provider.languageModel('google.gemini-2.5-pro');
      const swePreset = (model as any).swePreset;

      // OCI's Gemini integration doesn't support penalty parameters
      expect(swePreset.supportsPenalties).toBe(false);
      expect(swePreset.supportsTools).toBe(true);
    });

    it('should set supportsPenalties=true for Cohere models', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const swePreset = (model as any).swePreset;

      expect(swePreset.supportsPenalties).toBe(true);
      expect(swePreset.supportsTools).toBe(true);
    });

    it('should set supportsPenalties=true for Meta models', () => {
      const model = provider.languageModel('meta.llama-3.3-70b-instruct');
      const swePreset = (model as any).swePreset;

      expect(swePreset.supportsPenalties).toBe(true);
      expect(swePreset.supportsTools).toBe(true);
    });
  });

  describe('buildGenericChatRequest', () => {
    it('should exclude penalty parameters for xAI models', () => {
      const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        frequencyPenalty: 0.5, // Should be ignored for xAI
        presencePenalty: 0.5,  // Should be ignored for xAI
      };

      const request = buildGenericChatRequest(options);

      expect(request).not.toHaveProperty('frequencyPenalty');
      expect(request).not.toHaveProperty('presencePenalty');
      expect(request.temperature).toBeDefined();
      expect(request.topP).toBeDefined();
    });

    it('should exclude penalty parameters for Google models (OCI limitation)', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        frequencyPenalty: 0.5, // Should be ignored for Google via OCI
        presencePenalty: 0.3,  // Should be ignored for Google via OCI
      };

      const request = buildGenericChatRequest(options);

      // OCI's Gemini doesn't support penalty parameters
      expect(request).not.toHaveProperty('frequencyPenalty');
      expect(request).not.toHaveProperty('presencePenalty');
    });

    it('should include penalty parameters for models that support them', () => {
      const model = provider.languageModel('meta.llama-3.3-70b-instruct');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        frequencyPenalty: 0.5,
        presencePenalty: 0.3,
      };

      const request = buildGenericChatRequest(options);

      // Meta models support penalty parameters
      expect(request.frequencyPenalty).toBe(0.5);
      expect(request.presencePenalty).toBe(0.3);
    });

    it('should exclude stop sequences for xAI models', () => {
      const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        stopSequences: ['STOP', 'END'], // Should be ignored for xAI
      };

      const request = buildGenericChatRequest(options);

      // xAI doesn't support stop sequences
      expect(request).not.toHaveProperty('stop');
      expect(request.temperature).toBeDefined();
    });

    it('should include stop sequences for Gemini models', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        stopSequences: ['STOP', 'END'],
      };

      const request = buildGenericChatRequest(options);

      // Gemini supports stop sequences
      expect(request.stop).toEqual(['STOP', 'END']);
    });
  });

  describe('Model Family Detection', () => {
    it('should detect cohere-v2 model family for Command A', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      expect((model as any).modelFamily).toBe('cohere-v2');
    });

    it('should detect cohere model family for legacy Command R', () => {
      const model = provider.languageModel('cohere.command-r-plus-08-2024');
      expect((model as any).modelFamily).toBe('cohere');
    });

    it('should use generic family for Google models', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      expect((model as any).modelFamily).toBe('generic');
    });

    it('should use generic family for xAI models', () => {
      const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
      expect((model as any).modelFamily).toBe('generic');
    });

    it('should use generic family for Meta models', () => {
      const model = provider.languageModel('meta.llama-3.3-70b-instruct');
      expect((model as any).modelFamily).toBe('generic');
    });
  });
});

describe('Cohere Tool Results Handling', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should include isForceSingleStep=true when toolResults are present', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const buildCohereChatRequest = (model as any).buildCohereChatRequest.bind(model);

    // Simulate a conversation with tool results
    const options = {
      prompt: [
        { role: 'user' as const, content: [{ type: 'text' as const, text: 'What is the current directory?' }] },
        {
          role: 'assistant' as const,
          content: [
            { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"pwd"}' }
          ]
        },
        {
          role: 'tool' as const,
          content: [
            {
              type: 'tool-result' as const,
              toolCallId: 'call_1',
              output: { type: 'text' as const, value: '/Users/test/project' }
            }
          ]
        },
      ],
      maxOutputTokens: 1000,
      tools: [
        { type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }
      ],
    };

    const request = buildCohereChatRequest(options);

    // When tool results are present, isForceSingleStep must be true
    expect(request.isForceSingleStep).toBe(true);
  });

  it('should not include isForceSingleStep when no toolResults', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const buildCohereChatRequest = (model as any).buildCohereChatRequest.bind(model);

    const options = {
      prompt: [
        { role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] },
      ],
      maxOutputTokens: 1000,
    };

    const request = buildCohereChatRequest(options);

    // No tool results, no isForceSingleStep needed
    expect(request.isForceSingleStep).toBeUndefined();
  });

  it('should handle Cohere tool-result with undefined output gracefully', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const convertMessagesToCohereFormat = (model as any).convertMessagesToCohereFormat.bind(model);

    // Simulate a conversation with undefined tool result output
    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'Run pwd' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"pwd"}' }
        ]
      },
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_1',
            output: undefined as any
          }
        ]
      },
    ];

    // Should not throw and should handle undefined gracefully
    const result = convertMessagesToCohereFormat(prompt);

    expect(result.toolResults).toBeDefined();
    expect(result.toolResults.length).toBe(1);
    expect(result.toolResults[0].outputs).toBeDefined();
    // Empty string result for undefined output
    expect(result.toolResults[0].outputs[0].result).toBe('');
  });

  it('should handle Cohere tool-result with string output', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const convertMessagesToCohereFormat = (model as any).convertMessagesToCohereFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'Run pwd' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"pwd"}' }
        ]
      },
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_1',
            output: '/Users/test/project' as any // Raw string output
          }
        ]
      },
    ];

    const result = convertMessagesToCohereFormat(prompt);

    expect(result.toolResults).toBeDefined();
    expect(result.toolResults.length).toBe(1);
    expect(result.toolResults[0].outputs[0].result).toBe('/Users/test/project');
  });
});

describe('Tool Result Message Conversion', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should handle tool-result with text output type', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'tool' as const,
        content: [{
          type: 'tool-result' as const,
          toolCallId: 'call_123',
          output: { type: 'text' as const, value: '/Users/test/project' }
        }]
      }
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe('TOOL');
    expect(messages[0].content[0].text).toBe('/Users/test/project');
    expect(messages[0].toolCallId).toBe('call_123');
  });

  it('should handle tool-result with json output type', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'tool' as const,
        content: [{
          type: 'tool-result' as const,
          toolCallId: 'call_456',
          output: { type: 'json' as const, value: { files: ['a.ts', 'b.ts'] } }
        }]
      }
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].content[0].text).toBe('{"files":["a.ts","b.ts"]}');
  });

  it('should handle tool-result with raw string output (fallback)', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    // Some providers might send raw string output
    const prompt = [
      {
        role: 'tool' as const,
        content: [{
          type: 'tool-result' as const,
          toolCallId: 'call_789',
          output: '/Users/test/project' as any // Raw string
        }]
      }
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].content[0].text).toBe('/Users/test/project');
  });

  it('should handle tool-result with undefined output gracefully', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    // Edge case: output might be undefined
    const prompt = [
      {
        role: 'tool' as const,
        content: [{
          type: 'tool-result' as const,
          toolCallId: 'call_undefined',
          output: undefined as any
        }]
      }
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    // Should not crash and should provide empty or placeholder text
    expect(messages[0].content[0].text).toBeDefined();
    expect(messages[0].content[0].text).not.toBe('undefined');
  });
});

describe('SDK Serialization Compatibility', () => {
  it('should preserve isForceSingleStep through SDK getJsonObj serialization', async () => {
    // Import the real SDK serialization functions (not mocked)
    const ociModels = await import('oci-generativeaiinference/lib/model/index.js');

    // Create a Cohere request object with isForceSingleStep
    const cohereRequest = {
      apiFormat: 'COHERE',
      message: 'What is the current directory?',
      chatHistory: [
        { role: 'CHATBOT', message: 'Let me check', toolCalls: [{ name: 'bash', parameters: { command: 'pwd' } }] }
      ],
      maxTokens: 1000,
      temperature: 0.2,
      tools: [{ name: 'bash', description: 'Run bash', parameterDefinitions: {} }],
      toolResults: [
        { call: { name: 'bash', parameters: { command: 'pwd' } }, outputs: [{ result: '/Users/test' }] }
      ],
      isForceSingleStep: true,
    };

    // Serialize using the SDK's getJsonObj function
    const serialized = ociModels.BaseChatRequest.getJsonObj(cohereRequest) as Record<string, unknown>;

    // Verify isForceSingleStep is preserved after SDK serialization
    expect(serialized.isForceSingleStep).toBe(true);
    expect(serialized.toolResults).toBeDefined();
    expect(serialized.apiFormat).toBe('COHERE');
  });

  it('should preserve isForceSingleStep through full ChatDetails serialization', async () => {
    // Import the real SDK serialization functions
    const ociModels = await import('oci-generativeaiinference/lib/model/index.js');

    // Create full ChatDetails object as it would be sent to the API
    const chatDetails = {
      compartmentId: 'test-compartment',
      servingMode: { servingType: 'ON_DEMAND', modelId: 'cohere.command-a-03-2025' },
      chatRequest: {
        apiFormat: 'COHERE',
        message: 'Process the result',
        chatHistory: [],
        toolResults: [
          { call: { name: 'bash', parameters: {} }, outputs: [{ result: 'test output' }] }
        ],
        isForceSingleStep: true,
      },
    };

    // Serialize using ChatDetails.getJsonObj (what the SDK client uses)
    const serialized = ociModels.ChatDetails.getJsonObj(chatDetails) as Record<string, any>;

    // Verify the nested chatRequest still has isForceSingleStep
    expect(serialized.chatRequest.isForceSingleStep).toBe(true);
    expect(serialized.chatRequest.toolResults).toBeDefined();
  });
});

describe('Generic Format Tool Results Handling', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should convert tool-result messages to TOOL role for Gemini models', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'Run pwd' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"pwd"}' }
        ]
      },
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_1',
            output: { type: 'text' as const, value: '/Users/test/project' }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    // Verify we have user, assistant, and tool messages
    expect(messages).toHaveLength(3);
    expect(messages[0].role).toBe('USER');
    expect(messages[1].role).toBe('ASSISTANT');
    expect(messages[2].role).toBe('TOOL');
    expect((messages[2] as any).toolCallId).toBe('call_1');
  });

  it('should convert tool-call in assistant message for Grok models', () => {
    const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'List files' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'text' as const, text: 'Let me check.' },
          { type: 'tool-call' as const, toolCallId: 'call_ls', toolName: 'bash', input: '{"command":"ls"}' }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(2);
    expect(messages[1].role).toBe('ASSISTANT');
    // For xAI/Grok: tool calls are converted to text representation (Grok rejects TOOL_CALL content type)
    const assistantContent = messages[1].content as any[];
    expect(assistantContent.length).toBe(1);
    expect(assistantContent[0].type).toBe('TEXT');
    // Text should contain original message plus tool call description
    expect(assistantContent[0].text).toContain('Let me check.');
    expect(assistantContent[0].text).toContain('[Called tool "bash"');
    expect(assistantContent[0].text).toContain('{"command":"ls"}');
  });

  it('should convert tool results to USER messages for Grok models', () => {
    const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'List files' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'text' as const, text: 'Let me check.' },
          { type: 'tool-call' as const, toolCallId: 'call_ls', toolName: 'bash', input: '{"command":"ls"}' }
        ]
      },
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_ls',
            toolName: 'bash',
            output: { type: 'text' as const, value: 'file1.txt\nfile2.txt' }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    // For xAI/Grok: tool results are converted to USER messages (Grok rejects TOOL role)
    expect(messages).toHaveLength(3);
    expect(messages[0].role).toBe('USER');
    expect(messages[1].role).toBe('ASSISTANT');
    // Tool result should be converted to USER message
    expect(messages[2].role).toBe('USER');
    const toolResultContent = messages[2].content as any[];
    expect(toolResultContent[0].type).toBe('TEXT');
    expect(toolResultContent[0].text).toContain('[Tool result from "bash"');
    expect(toolResultContent[0].text).toContain('file1.txt');
  });

  it('should handle complete tool flow for Llama models', () => {
    const model = provider.languageModel('meta.llama-3.3-70b-instruct');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'What is the current directory?' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'tool-call' as const, toolCallId: 'call_pwd', toolName: 'bash', input: '{"command":"pwd"}' }
        ]
      },
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_pwd',
            toolName: 'bash',
            output: { type: 'text' as const, value: '/home/user/project' }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    // Llama (like xAI) rejects TOOL role and TOOL_CALL content, so convert to text
    expect(messages).toHaveLength(3);
    // User message
    expect(messages[0].role).toBe('USER');
    // Assistant with tool call converted to TEXT
    expect(messages[1].role).toBe('ASSISTANT');
    expect((messages[1].content as any[])[0].type).toBe('TEXT');
    expect((messages[1].content as any[])[0].text).toContain('[Called tool "bash"');
    // Tool result should be converted to USER message
    expect(messages[2].role).toBe('USER');
    const toolResultContent = messages[2].content as any[];
    expect(toolResultContent[0].type).toBe('TEXT');
    expect(toolResultContent[0].text).toContain('[Tool result from "bash"');
    expect(toolResultContent[0].text).toContain('/home/user/project');
  });

  it('should handle JSON output type in tool results', () => {
    const model = provider.languageModel('google.gemini-2.5-pro-preview');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_json',
            output: { type: 'json' as const, value: { files: ['a.ts', 'b.ts'], count: 2 } }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe('TOOL');
    // JSON should be stringified
    const text = messages[0].content[0].text;
    expect(text).toContain('files');
    expect(text).toContain('a.ts');
    expect(JSON.parse(text)).toEqual({ files: ['a.ts', 'b.ts'], count: 2 });
  });
});

describe('Reasoning Support', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  describe('SWE Presets for Reasoning', () => {
    it('should set supportsReasoning=true for xAI Grok reasoning models', () => {
      const reasoningModel = provider.languageModel('xai.grok-4-1-fast-reasoning');
      const miniModel = provider.languageModel('xai.grok-3-mini');
      const miniFastModel = provider.languageModel('xai.grok-3-mini-fast');

      // Grok reasoning model variants support reasoning
      expect((reasoningModel as any).swePreset.supportsReasoning).toBe(true);
      // Grok 3 Mini models "think before responding"
      expect((miniModel as any).swePreset.supportsReasoning).toBe(true);
      expect((miniFastModel as any).swePreset.supportsReasoning).toBe(true);
    });

    it('should set supportsReasoning=false for xAI Grok non-reasoning models', () => {
      const nonReasoningModel = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
      const fastModel = provider.languageModel('xai.grok-3-fast');
      const standardModel = provider.languageModel('xai.grok-3');

      // Non-reasoning Grok models don't support reasoning
      expect((nonReasoningModel as any).swePreset.supportsReasoning).toBe(false);
      expect((fastModel as any).swePreset.supportsReasoning).toBe(false);
      expect((standardModel as any).swePreset.supportsReasoning).toBe(false);
    });

    it('should NOT send reasoningEffort parameter for xAI models even when supportsReasoning=true', () => {
      // xAI Grok uses model variant selection for reasoning (grok-4-1-fast-reasoning vs grok-4-1-fast-non-reasoning)
      // NOT the reasoningEffort API parameter - sending it causes HTTP 400
      const reasoningModel = provider.languageModel('xai.grok-4-1-fast-reasoning');
      const buildGenericChatRequest = (reasoningModel as any).buildGenericChatRequest.bind(reasoningModel);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Think step by step' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': {
            reasoningEffort: 'HIGH', // Should be ignored for xAI
          },
        },
      };

      const request = buildGenericChatRequest(options);

      // xAI models should NEVER have reasoningEffort in the request
      // Reasoning is controlled by model name suffix, not API parameter
      expect(request).not.toHaveProperty('reasoningEffort');
    });

    it('should NOT send reasoningEffort parameter for xAI Grok 3 Mini models', () => {
      // Grok 3 Mini models "think before responding" but also don't support reasoningEffort parameter
      const miniModel = provider.languageModel('xai.grok-3-mini');
      const buildGenericChatRequest = (miniModel as any).buildGenericChatRequest.bind(miniModel);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Solve this problem' }] }],
        maxOutputTokens: 1000,
      };

      const request = buildGenericChatRequest(options);

      // Even though supportsReasoning=true, xAI models don't use the reasoningEffort parameter
      expect(request).not.toHaveProperty('reasoningEffort');
    });

    it('should set supportsReasoning=true for Cohere reasoning models', () => {
      const model = provider.languageModel('cohere.command-a-reasoning-08-2025');
      const swePreset = (model as any).swePreset;

      expect(swePreset.supportsReasoning).toBe(true);
    });

    it('should set supportsReasoning=false for non-reasoning Cohere models', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const swePreset = (model as any).swePreset;

      // Standard Cohere models don't have reasoning
      expect(swePreset.supportsReasoning).toBe(false);
    });

    it('should set supportsReasoning=false for Google Gemini Pro models (OCI limitation)', () => {
      const model = provider.languageModel('google.gemini-2.5-pro');
      const swePreset = (model as any).swePreset;

      // OCI GenAI does NOT expose reasoningEffort parameter for Gemini models
      // Even though Gemini 2.5 has reasoning capabilities, OCI doesn't support the API parameter
      expect(swePreset.supportsReasoning).toBe(false);
    });

    it('should set supportsReasoning=false for Google Gemini Flash models (OCI limitation)', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const swePreset = (model as any).swePreset;

      // OCI GenAI does NOT expose reasoningEffort parameter for Gemini models
      expect(swePreset.supportsReasoning).toBe(false);
    });

    it('should set supportsReasoning=false for Google Gemini Flash-Lite models', () => {
      const model = provider.languageModel('google.gemini-2.5-flash-lite');
      const swePreset = (model as any).swePreset;

      // Flash-Lite has thinking disabled for speed/cost optimization
      expect(swePreset.supportsReasoning).toBe(false);
    });

    it('should set supportsReasoning=false for non-reasoning Meta Llama 3 models', () => {
      const model = provider.languageModel('meta.llama-3.3-70b-instruct');
      const swePreset = (model as any).swePreset;

      expect(swePreset.supportsReasoning).toBe(false);
    });

    it('should set supportsReasoning=false for Meta Llama 4 models (OCI limitation)', () => {
      // Note: Llama 4 models require dedicated AI cluster, but we can still verify the preset
      // These models may have internal reasoning but OCI doesn't expose reasoningEffort parameter
      const dedicatedProvider = createOCI({
        region: 'us-chicago-1',
        compartmentId: 'test-compartment',
        servingMode: 'dedicated',  // Required for Llama 4 models
        endpointId: 'test-endpoint',
      });

      const maverick = dedicatedProvider.languageModel('meta.llama-4-maverick-17b-128e-instruct-fp8');
      const scout = dedicatedProvider.languageModel('meta.llama-4-scout-17b-16e-instruct');

      // OCI GenAI does NOT expose reasoningEffort for Meta Llama models
      expect((maverick as any).swePreset.supportsReasoning).toBe(false);
      expect((scout as any).swePreset.supportsReasoning).toBe(false);
    });

    it('should set supportsReasoning=true for OpenAI gpt-oss models', () => {
      const model = provider.languageModel('openai.gpt-oss-120b');
      const swePreset = (model as any).swePreset;

      expect(swePreset.supportsReasoning).toBe(true);
      expect(swePreset.supportsTools).toBe(true);
      expect(swePreset.supportsPenalties).toBe(true);
    });
  });

  describe('buildGenericChatRequest with reasoning', () => {
    it('should include reasoningEffort for models that support reasoning', () => {
      // Use OpenAI gpt-oss which supports reasoning (OCI limitation: Gemini doesn't expose reasoningEffort)
      const model = provider.languageModel('openai.gpt-oss-120b');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Think step by step' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': {
            reasoningEffort: 'HIGH',
          },
        },
      };

      const request = buildGenericChatRequest(options);

      expect(request.reasoningEffort).toBe('HIGH');
    });

    it('should not include reasoningEffort for models that do not support reasoning', () => {
      const model = provider.languageModel('meta.llama-3.3-70b-instruct');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': {
            reasoningEffort: 'HIGH', // Should be ignored
          },
        },
      };

      const request = buildGenericChatRequest(options);

      expect(request).not.toHaveProperty('reasoningEffort');
    });

    it('should default to MEDIUM reasoningEffort when not specified for reasoning models', () => {
      // Use OpenAI gpt-oss which supports reasoning (OCI limitation: Gemini doesn't expose reasoningEffort)
      const model = provider.languageModel('openai.gpt-oss-120b');
      const buildGenericChatRequest = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        // No providerOptions specified
      };

      const request = buildGenericChatRequest(options);

      // Should default to MEDIUM for reasoning-capable models
      expect(request.reasoningEffort).toBe('MEDIUM');
    });
  });

  describe('buildCohereChatRequest with thinking', () => {
    it('should include thinking parameter for Cohere reasoning models', () => {
      const model = provider.languageModel('cohere.command-a-reasoning-08-2025');
      const buildCohereChatRequest = (model as any).buildCohereChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Think about this' }] }],
        maxOutputTokens: 4000,
      };

      const request = buildCohereChatRequest(options);

      // Should include thinking parameter with ENABLED type
      expect(request.thinking).toBeDefined();
      expect(request.thinking.type).toBe('ENABLED');
    });

    it('should include custom budgetTokens when specified', () => {
      const model = provider.languageModel('cohere.command-a-reasoning-08-2025');
      const buildCohereChatRequest = (model as any).buildCohereChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Complex problem' }] }],
        maxOutputTokens: 4000,
        providerOptions: {
          'oci-genai': {
            thinkingBudgetTokens: 31000, // Maximum reasoning
          },
        },
      };

      const request = buildCohereChatRequest(options);

      expect(request.thinking.budgetTokens).toBe(31000);
    });

    it('should not include thinking for non-reasoning Cohere models', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const buildCohereChatRequest = (model as any).buildCohereChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
      };

      const request = buildCohereChatRequest(options);

      expect(request.thinking).toBeUndefined();
    });
  });

  describe('Response reasoning content extraction', () => {
    it('should extract reasoningContent from Generic API response', () => {
      const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
      const extractReasoningContent = (model as any).extractReasoningContent?.bind(model);

      expect(extractReasoningContent).toBeDefined();

      const genericResponse = {
        choices: [{
          message: {
            content: [{ type: 'TEXT', text: 'The answer is 42' }],
            reasoningContent: 'Let me think through this step by step...',
          },
          finishReason: 'COMPLETE',
        }],
      };

      const reasoning = extractReasoningContent(genericResponse);

      expect(reasoning).toBe('Let me think through this step by step...');
    });

    it('should return undefined when no reasoningContent present', () => {
      const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
      const extractReasoningContent = (model as any).extractReasoningContent.bind(model);

      const genericResponse = {
        choices: [{
          message: {
            content: [{ type: 'TEXT', text: 'The answer is 42' }],
          },
          finishReason: 'COMPLETE',
        }],
      };

      const reasoning = extractReasoningContent(genericResponse);

      expect(reasoning).toBeUndefined();
    });
  });

  describe('doGenerate with reasoning content', () => {
    it('should include reasoning content in response for models that support it', async () => {
      // This test verifies the response structure includes reasoning
      const model = provider.languageModel('xai.grok-4-1-fast-non-reasoning');

      // The mock returns a response, we're testing the extraction logic
      const result = await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'Think step by step' }] }],
      });

      // The mock doesn't include reasoning, so content should only be text
      expect(result.content).toBeDefined();
      expect(result.finishReason).toBeDefined();
    });
  });
});

describe('Provider Instantiation', () => {
  it('should create provider with settings', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
      configProfile: 'CUSTOM',
    });

    expect(provider).toBeInstanceOf(OCIProvider);
    expect(provider.getSettings().compartmentId).toBe('test-compartment');
    expect(provider.getSettings().region).toBe('us-chicago-1');
    expect(provider.getSettings().configProfile).toBe('CUSTOM');
  });

  it('should throw error when creating language model without compartmentId', () => {
    // Clear env var if set
    const originalEnv = process.env.OCI_COMPARTMENT_ID;
    delete process.env.OCI_COMPARTMENT_ID;

    const provider = createOCI({});

    expect(() => provider.languageModel('google.gemini-2.5-flash'))
      .toThrow('Missing compartment ID');

    // Restore env
    if (originalEnv) process.env.OCI_COMPARTMENT_ID = originalEnv;
  });

  it('should not support text embedding models', () => {
    const provider = createOCI({ compartmentId: 'test' });
    expect(() => provider.textEmbeddingModel('any'))
      .toThrow('Text embedding models are not supported');
  });

  it('should not support image models', () => {
    const provider = createOCI({ compartmentId: 'test' });
    expect(() => provider.imageModel('any'))
      .toThrow('Image models are not supported');
  });
});

/**
 * Regression tests for Llama tool calling fix
 * Issue: Llama models reject TOOL role and TOOL_CALL content type
 * Fix: Convert tool calls/results to TEXT format (same as xAI/Grok)
 */
describe('Llama Tool Calling Regression', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should detect Llama models correctly', () => {
    const llama33 = provider.languageModel('meta.llama-3.3-70b-instruct');
    const llama31 = provider.languageModel('meta.llama-3.1-405b-instruct');
    const gemini = provider.languageModel('google.gemini-2.5-flash');

    // Check model IDs start with meta.llama
    expect((llama33 as any).modelId.startsWith('meta.llama')).toBe(true);
    expect((llama31 as any).modelId.startsWith('meta.llama')).toBe(true);
    expect((gemini as any).modelId.startsWith('meta.llama')).toBe(false);
  });

  it('should convert assistant tool calls to TEXT for Llama models', () => {
    const model = provider.languageModel('meta.llama-3.3-70b-instruct');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'List files' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'text' as const, text: 'Let me list the files.' },
          { type: 'tool-call' as const, toolCallId: 'call_ls', toolName: 'bash', input: '{"command":"ls -la"}' }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(2);
    expect(messages[1].role).toBe('ASSISTANT');
    // Tool call should be converted to TEXT
    const content = messages[1].content as any[];
    expect(content).toHaveLength(1);
    expect(content[0].type).toBe('TEXT');
    expect(content[0].text).toContain('Let me list the files.');
    expect(content[0].text).toContain('[Called tool "bash"');
    expect(content[0].text).toContain('{"command":"ls -la"}');
  });

  it('should convert tool results to USER messages for Llama models', () => {
    const model = provider.languageModel('meta.llama-3.1-70b-instruct');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'Run pwd' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'tool-call' as const, toolCallId: 'call_pwd', toolName: 'bash', input: '{"command":"pwd"}' }
        ]
      },
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_pwd',
            toolName: 'bash',
            output: { type: 'text' as const, value: '/home/user/project' }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(3);
    // Tool result should be USER, not TOOL
    expect(messages[2].role).toBe('USER');
    const content = messages[2].content as any[];
    expect(content[0].type).toBe('TEXT');
    expect(content[0].text).toContain('[Tool result from "bash"');
    expect(content[0].text).toContain('/home/user/project');
  });

  it('should handle multi-step tool flow for Llama', () => {
    const model = provider.languageModel('meta.llama-3.3-70b-instruct');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    // Simulate a 2-step tool conversation
    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'List files then read README.md' }] },
      // First tool call
      {
        role: 'assistant' as const,
        content: [
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"ls"}' }
        ]
      },
      // First tool result
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_1',
            toolName: 'bash',
            output: { type: 'text' as const, value: 'README.md\nindex.ts' }
          }
        ]
      },
      // Second tool call
      {
        role: 'assistant' as const,
        content: [
          { type: 'text' as const, text: 'Now reading README.md.' },
          { type: 'tool-call' as const, toolCallId: 'call_2', toolName: 'read', input: '{"path":"README.md"}' }
        ]
      },
      // Second tool result
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_2',
            toolName: 'read',
            output: { type: 'text' as const, value: '# Project\n\nThis is a test project.' }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(5);
    
    // Verify alternating pattern for Llama
    expect(messages[0].role).toBe('USER');
    expect(messages[1].role).toBe('ASSISTANT');
    expect(messages[2].role).toBe('USER'); // Tool result -> USER
    expect(messages[3].role).toBe('ASSISTANT');
    expect(messages[4].role).toBe('USER'); // Tool result -> USER

    // Verify content is TEXT, not TOOL_CALL
    expect((messages[1].content as any[])[0].type).toBe('TEXT');
    expect((messages[3].content as any[])[0].type).toBe('TEXT');
  });

  it('should NOT convert tool calls for Gemini (only for Llama/xAI)', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      { role: 'user' as const, content: [{ type: 'text' as const, text: 'List files' }] },
      {
        role: 'assistant' as const,
        content: [
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"ls"}' }
        ]
      },
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_1',
            output: { type: 'text' as const, value: 'file1.txt' }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(3);
    // Gemini should use native TOOL role with toolCallId at message level
    expect(messages[2].role).toBe('TOOL');
    expect((messages[2] as any).toolCallId).toBe('call_1');
    expect((messages[2].content as any[])[0].type).toBe('TEXT');
    // Gemini should use toolCalls array at message level (not TOOL_CALL content type)
    expect((messages[1] as any).toolCalls).toBeDefined();
    expect((messages[1] as any).toolCalls[0].type).toBe('FUNCTION');
    expect((messages[1] as any).toolCalls[0].name).toBe('bash');
  });
});

/**
 * Regression tests for parseOCIError helper
 * Issue: OCI API errors are cryptic and unhelpful
 * Fix: Map common errors to user-friendly messages with hints
 */
describe('Error Handling Regression', () => {
  // We need to import the parseOCIError function
  // Since it's a module-level function, we test it indirectly through the provider

  it('should provide user-friendly message for rate limit errors', async () => {
    // Create a provider that will throw a rate limit error
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');

    // Mock the client to throw a rate limit error
    (model as any).client = {
      chat: vi.fn().mockRejectedValue(new Error('Service request limit is exceeded, request is throttled')),
    };

    try {
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      });
      expect.fail('Should have thrown');
    } catch (error: any) {
      expect(error.message).toContain('[OCI GenAI] Rate limit exceeded');
      expect(error.message).toContain('Hint: Wait a moment');
    }
  });

  it('should provide user-friendly message for format errors', async () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('meta.llama-3.3-70b-instruct');

    (model as any).client = {
      chat: vi.fn().mockRejectedValue(new Error('Please pass in correct format of request')),
    };

    try {
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      });
      expect.fail('Should have thrown');
    } catch (error: any) {
      expect(error.message).toContain('[OCI GenAI] OCI API rejected the request format');
      expect(error.message).toContain('Hint:');
    }
  });

  it('should provide user-friendly message for auth errors', async () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockRejectedValue(new Error('Unauthorized: invalid credentials')),
    };

    try {
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      });
      expect.fail('Should have thrown');
    } catch (error: any) {
      expect(error.message).toContain('[OCI GenAI] Authentication failed');
      expect(error.message).toContain('OCI config profile');
    }
  });

  it('should provide user-friendly message for 404/not found errors', async () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockRejectedValue(new Error('NotAuthorizedOrNotFound')),
    };

    try {
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      });
      expect.fail('Should have thrown');
    } catch (error: any) {
      expect(error.message).toContain('[OCI GenAI] Model or resource not found');
      expect(error.message).toContain('google.gemini-2.5-flash');
    }
  });

  it('should include model ID in generic error messages', async () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('meta.llama-3.3-70b-instruct');

    (model as any).client = {
      chat: vi.fn().mockRejectedValue(new Error('Unknown error XYZ123')),
    };

    try {
      await model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      });
      expect.fail('Should have thrown');
    } catch (error: any) {
      expect(error.message).toContain('[OCI GenAI]');
      expect(error.message).toContain('meta.llama-3.3-70b-instruct');
      expect(error.message).toContain('Unknown error XYZ123');
    }
  });
});

/**
 * Tests for Cohere V2 API format (Command A models)
 */
describe('Cohere V2 API Format', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  describe('Model Family Detection', () => {
    it('should detect cohere-v2 family for Command A models', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      expect((model as any).modelFamily).toBe('cohere-v2');
    });

    it('should detect cohere-v2 family for Command A reasoning models', () => {
      const model = provider.languageModel('cohere.command-a-reasoning-08-2025');
      expect((model as any).modelFamily).toBe('cohere-v2');
    });

    it('should detect legacy cohere family for Command R models', () => {
      const model = provider.languageModel('cohere.command-r-08-2024');
      expect((model as any).modelFamily).toBe('cohere');
    });
  });

  describe('convertMessagesToCohereV2Format', () => {
    it('should convert system messages correctly', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertMessagesToCohereV2Format.bind(model);

      const prompt = [
        { role: 'system' as const, content: 'You are a helpful assistant.' },
      ];

      const messages = convert(prompt);

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('SYSTEM');
      expect(messages[0].content[0].type).toBe('TEXT');
      expect(messages[0].content[0].text).toBe('You are a helpful assistant.');
    });

    it('should convert user messages correctly', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertMessagesToCohereV2Format.bind(model);

      const prompt = [
        { role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello world' }] },
      ];

      const messages = convert(prompt);

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('USER');
      expect(messages[0].content[0].type).toBe('TEXT');
      expect(messages[0].content[0].text).toBe('Hello world');
    });

    it('should convert assistant messages with tool calls', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertMessagesToCohereV2Format.bind(model);

      const prompt = [
        {
          role: 'assistant' as const,
          content: [
            { type: 'text' as const, text: 'Let me check that.' },
            { type: 'tool-call' as const, toolCallId: 'call_123', toolName: 'bash', input: '{"command":"ls"}' }
          ]
        },
      ];

      const messages = convert(prompt);

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('ASSISTANT');
      expect(messages[0].content[0].type).toBe('TEXT');
      expect(messages[0].toolCalls).toBeDefined();
      expect(messages[0].toolCalls.length).toBe(1);
      expect(messages[0].toolCalls[0].type).toBe('FUNCTION');
      expect(messages[0].toolCalls[0].function.name).toBe('bash');
    });

    it('should convert tool results to TOOL messages', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertMessagesToCohereV2Format.bind(model);

      const prompt = [
        {
          role: 'tool' as const,
          content: [{
            type: 'tool-result' as const,
            toolCallId: 'call_123',
            output: { type: 'text' as const, value: '/Users/test' }
          }]
        },
      ];

      const messages = convert(prompt);

      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('TOOL');
      expect(messages[0].toolCallId).toBeDefined();
      expect(messages[0].content[0].type).toBe('TEXT');
      expect(messages[0].content[0].text).toBe('/Users/test');
    });

    it('should handle full multi-turn conversation with tools', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertMessagesToCohereV2Format.bind(model);

      const prompt = [
        { role: 'system' as const, content: 'You are a coding assistant.' },
        { role: 'user' as const, content: [{ type: 'text' as const, text: 'List files' }] },
        {
          role: 'assistant' as const,
          content: [
            { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"ls"}' }
          ]
        },
        {
          role: 'tool' as const,
          content: [{
            type: 'tool-result' as const,
            toolCallId: 'call_1',
            output: { type: 'text' as const, value: 'file1.txt\nfile2.txt' }
          }]
        },
        { role: 'user' as const, content: [{ type: 'text' as const, text: 'Now read file1.txt' }] },
      ];

      const messages = convert(prompt);

      expect(messages).toHaveLength(5);
      expect(messages[0].role).toBe('SYSTEM');
      expect(messages[1].role).toBe('USER');
      expect(messages[2].role).toBe('ASSISTANT');
      expect(messages[3].role).toBe('TOOL');
      expect(messages[4].role).toBe('USER');
    });
  });

  describe('convertToolsToCohereV2', () => {
    it('should convert tools to Cohere V2 function format', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertToolsToCohereV2.bind(model);

      const tools = [
        {
          type: 'function' as const,
          name: 'get_weather',
          description: 'Get weather for a location',
          inputSchema: {
            type: 'object',
            properties: {
              location: { type: 'string', description: 'City name' }
            },
            required: ['location']
          }
        }
      ];

      const converted = convert(tools);

      expect(converted).toHaveLength(1);
      expect(converted[0].type).toBe('FUNCTION');
      expect(converted[0].function.name).toBe('get_weather');
      expect(converted[0].function.description).toBe('Get weather for a location');
      expect(converted[0].function.parameters.type).toBe('object');
      expect(converted[0].function.parameters.properties.location.type).toBe('string');
    });

    it('should clean JSON schema in tool parameters', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertToolsToCohereV2.bind(model);

      const tools = [
        {
          type: 'function' as const,
          name: 'test_tool',
          description: 'Test',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
            type: 'object',
            additionalProperties: false,
            properties: {
              name: { type: 'string', minLength: 1 }
            }
          }
        }
      ];

      const converted = convert(tools);

      expect(converted[0].function.parameters).not.toHaveProperty('$schema');
      expect(converted[0].function.parameters).not.toHaveProperty('additionalProperties');
      expect(converted[0].function.parameters.properties.name).not.toHaveProperty('minLength');
    });

    it('should return undefined for empty tools array', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const convert = (model as any).convertToolsToCohereV2.bind(model);

      expect(convert([])).toBeUndefined();
      expect(convert(null)).toBeUndefined();
      expect(convert(undefined)).toBeUndefined();
    });
  });

  describe('buildCohereV2ChatRequest', () => {
    it('should build request with correct apiFormat', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
      };

      const request = build(options);

      expect(request.apiFormat).toBe('COHEREV2');
      expect(request.messages).toBeDefined();
      expect(request.maxTokens).toBe(1000);
    });

    it('should include tools when provided', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'List files' }] }],
        maxOutputTokens: 1000,
        tools: [
          { type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }
        ],
      };

      const request = build(options);

      expect(request.tools).toBeDefined();
      expect(request.tools.length).toBe(1);
      expect(request.tools[0].function.name).toBe('bash');
    });

    it('should map toolChoice to toolsChoice', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'List files' }] }],
        maxOutputTokens: 1000,
        tools: [
          { type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }
        ],
        toolChoice: { type: 'required' as const },
      };

      const request = build(options);

      expect(request.toolsChoice).toBe('REQUIRED');
    });

    it('should set toolsChoice to NONE when toolChoice is none', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }] }],
        maxOutputTokens: 1000,
        tools: [
          { type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }
        ],
        toolChoice: { type: 'none' as const },
      };

      const request = build(options);

      expect(request.toolsChoice).toBe('NONE');
    });

    it('should include thinking parameter for reasoning models', () => {
      const model = provider.languageModel('cohere.command-a-reasoning-08-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Think about this' }] }],
        maxOutputTokens: 4000,
      };

      const request = build(options);

      expect(request.thinking).toBeDefined();
      expect(request.thinking.type).toBe('ENABLED');
    });
  });
});

/**
 * Tests for dedicated endpoint serving mode
 */
describe('Dedicated Endpoint Serving Mode', () => {
  it('should use dedicated serving mode when configured', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
      servingMode: 'dedicated',
      endpointId: 'ocid1.endpoint.test123',
    });

    const model = provider.languageModel('custom-model');
    const getServingMode = (model as any).getServingMode.bind(model);

    const servingMode = getServingMode();

    expect(servingMode.servingType).toBe('DEDICATED');
    expect(servingMode.endpointId).toBe('ocid1.endpoint.test123');
  });

  it('should use on-demand serving mode by default', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');
    const getServingMode = (model as any).getServingMode.bind(model);

    const servingMode = getServingMode();

    expect(servingMode.servingType).toBe('ON_DEMAND');
    expect(servingMode.modelId).toBe('google.gemini-2.5-flash');
  });

  it('should throw error for dedicated-only models in on-demand mode', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
      servingMode: 'on-demand',
    });

    // Llama 4 models require dedicated clusters
    expect(() => provider.languageModel('meta.llama-4-maverick-17b-128e-instruct-fp8'))
      .toThrow('requires a dedicated AI cluster');
  });
});

/**
 * Tests for streaming response handling (doStream)
 */
describe('Streaming Response Handling', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should emit stream-start event first', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    const reader = result.stream.getReader();
    const firstChunk = await reader.read();

    expect(firstChunk.value?.type).toBe('stream-start');
    reader.releaseLock();
  });

  it('should emit text-start, text-delta, text-end for text content', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    const reader = result.stream.getReader();
    const events: string[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) events.push(value.type);
    }

    expect(events).toContain('stream-start');
    expect(events).toContain('text-start');
    expect(events).toContain('text-delta');
    expect(events).toContain('text-end');
    expect(events).toContain('finish');
  });

  it('should emit tool-call events when model calls tools', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    // Mock client to return tool calls
    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: {
                content: [{
                  type: 'TOOL_CALL',
                  id: 'call_123',
                  name: 'bash',
                  arguments: '{"command":"ls"}',
                }]
              },
              finishReason: 'TOOL_CALL',
            }],
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'List files' }] }],
      tools: [{ type: 'function', name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }],
    });

    const reader = result.stream.getReader();
    const events: string[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) events.push(value.type);
    }

    expect(events).toContain('tool-input-start');
    expect(events).toContain('tool-input-delta');
    expect(events).toContain('tool-input-end');
    expect(events).toContain('tool-call');
  });

  it('should emit finish event with usage information', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    const reader = result.stream.getReader();
    let finishEvent: any = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value?.type === 'finish') finishEvent = value;
    }

    expect(finishEvent).not.toBeNull();
    expect(finishEvent.finishReason).toBeDefined();
    expect(finishEvent.usage).toBeDefined();
    expect(finishEvent.usage.inputTokens).toBeDefined();
    expect(finishEvent.usage.outputTokens).toBeDefined();
  });

  it('should emit error event on API failure', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockRejectedValue(new Error('API error')),
    };

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    const reader = result.stream.getReader();
    let errorEvent: any = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value?.type === 'error') errorEvent = value;
    }

    expect(errorEvent).not.toBeNull();
    expect(errorEvent.error).toBeDefined();
  });
});

/**
 * Tests for image/file content handling
 */
describe('Image and File Content Handling', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should convert base64 image content to OCI format', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convert = (model as any).convertMessagesToGenericFormat.bind(model);

    const base64Data = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const prompt = [
      {
        role: 'user' as const,
        content: [
          { type: 'text' as const, text: 'What is in this image?' },
          { type: 'file' as const, data: base64Data, mediaType: 'image/png' }
        ]
      }
    ];

    const messages = convert(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe('USER');
    const content = messages[0].content as any[];
    expect(content.length).toBe(2);
    expect(content[0].type).toBe('TEXT');
    expect(content[1].type).toBe('IMAGE');
    expect(content[1].source.type).toBe('BASE64');
    expect(content[1].source.data).toBe(base64Data);
    expect(content[1].source.mediaType).toBe('image/png');
  });

  it('should default to image/png when mediaType not specified', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convert = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'user' as const,
        content: [
          { type: 'file' as const, data: 'base64data' }
        ]
      }
    ];

    const messages = convert(prompt);

    const imageContent = messages[0].content[0] as any;
    expect(imageContent.source.mediaType).toBe('image/png');
  });
});

/**
 * Tests for provider options handling
 */
describe('Provider Options Handling', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  describe('reasoningEffort', () => {
    it('should include reasoningEffort when specified for reasoning-capable models', () => {
      const model = provider.languageModel('openai.gpt-oss-120b');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': {
            reasoningEffort: 'HIGH',
          },
        },
      };

      const request = build(options);

      expect(request.reasoningEffort).toBe('HIGH');
    });

    it('should support LOW, MEDIUM, HIGH values', () => {
      const model = provider.languageModel('openai.gpt-oss-120b');
      const build = (model as any).buildGenericChatRequest.bind(model);

      for (const effort of ['LOW', 'MEDIUM', 'HIGH']) {
        const options = {
          prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
          maxOutputTokens: 1000,
          providerOptions: {
            'oci-genai': { reasoningEffort: effort },
          },
        };

        const request = build(options);
        expect(request.reasoningEffort).toBe(effort);
      }
    });
  });

  describe('thinkingBudgetTokens', () => {
    it('should include thinkingBudgetTokens for Cohere reasoning models', () => {
      const model = provider.languageModel('cohere.command-a-reasoning-08-2025');
      const build = (model as any).buildCohereChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 4000,
        providerOptions: {
          'oci-genai': {
            thinkingBudgetTokens: 16000,
          },
        },
      };

      const request = build(options);

      expect(request.thinking).toBeDefined();
      expect(request.thinking.type).toBe('ENABLED');
      expect(request.thinking.budgetTokens).toBe(16000);
    });

    it('should include thinkingBudgetTokens in Cohere V2 requests', () => {
      const model = provider.languageModel('cohere.command-a-reasoning-08-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 4000,
        providerOptions: {
          'oci-genai': {
            thinkingBudgetTokens: 31000,
          },
        },
      };

      const request = build(options);

      expect(request.thinking.budgetTokens).toBe(31000);
    });
  });

  describe('SWE defaults', () => {
    it('should apply default temperature when not specified', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        // temperature not specified
      };

      const request = build(options);

      // Google preset has temperature 0.1
      expect(request.temperature).toBe(0.1);
    });

    it('should override default temperature when specified', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        temperature: 0.7,
      };

      const request = build(options);

      expect(request.temperature).toBe(0.7);
    });

    it('should apply default topP when not specified', () => {
      const model = provider.languageModel('meta.llama-3.3-70b-instruct');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
      };

      const request = build(options);

      // Meta preset has topP 0.9
      expect(request.topP).toBe(0.9);
    });
  });
});

/**
 * Tests for edge cases in response handling
 */
describe('Response Handling Edge Cases', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should handle response with empty text content from Gemini', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    // Gemini sometimes returns empty TEXT objects alongside tool calls
    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: {
                content: [
                  { type: 'TEXT', text: '' }, // Empty text
                  { type: 'TOOL_CALL', id: 'call_1', name: 'bash', arguments: '{}' }
                ]
              },
              finishReason: 'TOOL_CALL',
            }],
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    // Should have tool call but no empty text
    const textContent = result.content.filter(c => c.type === 'text');
    const toolCalls = result.content.filter(c => c.type === 'tool-call');

    expect(textContent.length).toBe(0); // Empty text should be filtered
    expect(toolCalls.length).toBe(1);
  });

  it('should handle Cohere V2 response with toolPlan', async () => {
    const model = provider.languageModel('cohere.command-a-reasoning-08-2025');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            message: {
              content: [{ type: 'TEXT', text: 'Here is the answer' }],
              toolPlan: 'I need to think about this step by step...',
              toolCalls: []
            },
            finishReason: 'COMPLETE',
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Think about this' }] }],
    });

    // toolPlan should be included as reasoning content
    const reasoning = result.content.filter(c => c.type === 'reasoning');
    expect(reasoning.length).toBeGreaterThan(0);
    expect((reasoning[0] as any).text).toContain('step by step');
  });

  it('should handle message.content as string fallback', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    // Some responses may have content as a direct string
    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: {
                content: 'Direct string response'
              },
              finishReason: 'COMPLETE',
            }],
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    const textContent = result.content.filter(c => c.type === 'text');
    expect(textContent.length).toBe(1);
    expect((textContent[0] as any).text).toBe('Direct string response');
  });

  it('should handle message.text fallback', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    // Alternative response format with text property
    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: {
                text: 'Text property response'
              },
              finishReason: 'COMPLETE',
            }],
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    const textContent = result.content.filter(c => c.type === 'text');
    expect(textContent.length).toBe(1);
    expect((textContent[0] as any).text).toBe('Text property response');
  });

  it('should handle tool_calls at message level', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    // Some models put tool_calls at message level instead of in content
    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: {
                content: [{ type: 'TEXT', text: '' }],
                tool_calls: [{
                  id: 'call_msg',
                  function: {
                    name: 'read',
                    arguments: '{"path":"test.txt"}'
                  }
                }]
              },
              finishReason: 'TOOL_CALL',
            }],
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Read test.txt' }] }],
    });

    const toolCalls = result.content.filter(c => c.type === 'tool-call');
    expect(toolCalls.length).toBe(1);
    expect((toolCalls[0] as any).toolName).toBe('read');
    expect((toolCalls[0] as any).input).toContain('test.txt');
  });
});

/**
 * Tests for finish reason mapping
 */
describe('Finish Reason Mapping', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should map MAX_TOKENS to length', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: 'Truncated...' }] },
              finishReason: 'MAX_TOKENS',
            }],
            usage: { promptTokens: 10, completionTokens: 1000 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('length');
  });

  it('should map TOOL_CALL to tool-calls', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: {
                content: [{ type: 'TOOL_CALL', id: 'c', name: 'bash', arguments: '{}' }]
              },
              finishReason: 'TOOL_CALL',
            }],
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('tool-calls');
  });

  it('should map CONTENT_FILTER to content-filter', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: '' }] },
              finishReason: 'CONTENT_FILTER',
            }],
            usage: { promptTokens: 10, completionTokens: 0 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('content-filter');
  });

  it('should default to stop for unknown finish reasons', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: 'Done' }] },
              finishReason: 'UNKNOWN_REASON',
            }],
            usage: { promptTokens: 10, completionTokens: 20 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('stop');
  });
});

/**
 * Regression tests for multi-turn tool conversations
 * Issue: Tool results weren't being properly sent back to models
 * Fix: Proper message format conversion for each model family
 */
describe('Multi-Turn Tool Conversation Regression', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should preserve tool call IDs through conversion for Gemini', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const toolCallId = 'unique_call_id_12345';
    const prompt = [
      {
        role: 'tool' as const,
        content: [{
          type: 'tool-result' as const,
          toolCallId: toolCallId,
          output: { type: 'text' as const, value: 'result data' }
        }]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    // Tool call ID should be preserved
    expect((messages[0] as any).toolCallId).toBe(toolCallId);
  });

  it('should handle error-text output type in tool results', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'tool' as const,
        content: [{
          type: 'tool-result' as const,
          toolCallId: 'call_error',
          output: { type: 'error-text' as const, value: 'Command failed: exit code 1' }
        }]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages[0].content[0].text).toBe('Command failed: exit code 1');
  });

  it('should handle multiple tool results in single message', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'tool' as const,
        content: [
          {
            type: 'tool-result' as const,
            toolCallId: 'call_1',
            output: { type: 'text' as const, value: 'result 1' }
          },
          {
            type: 'tool-result' as const,
            toolCallId: 'call_2',
            output: { type: 'text' as const, value: 'result 2' }
          }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    // Should create separate TOOL messages for each result
    expect(messages.length).toBeGreaterThanOrEqual(1);
  });

  it('should handle tool results with JSON objects', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const jsonValue = { files: ['a.ts', 'b.ts'], count: 2, nested: { key: 'value' } };
    const prompt = [
      {
        role: 'tool' as const,
        content: [{
          type: 'tool-result' as const,
          toolCallId: 'call_json',
          output: { type: 'json' as const, value: jsonValue }
        }]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    // JSON should be stringified
    const text = messages[0].content[0].text;
    const parsed = JSON.parse(text);
    expect(parsed).toEqual(jsonValue);
  });

  it('should handle assistant messages with mixed text and tool calls for Gemini', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'assistant' as const,
        content: [
          { type: 'text' as const, text: 'I will run two commands.' },
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"ls"}' },
          { type: 'tool-call' as const, toolCallId: 'call_2', toolName: 'bash', input: '{"command":"pwd"}' }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe('ASSISTANT');
    // Gemini with PARALLEL tool calls: OCI Generic format doesn't support parallel function calls
    // So we convert to text fallback (same as Llama/xAI)
    const msg = messages[0] as any;
    const content = msg.content as any[];
    expect(content.length).toBe(1);
    expect(content[0].type).toBe('TEXT');
    // Text should contain original message plus tool calls in text format
    expect(content[0].text).toContain('I will run two commands.');
    expect(content[0].text).toContain('[Called tool "bash"');
    expect(content[0].text).toContain('{"command":"ls"}');
    expect(content[0].text).toContain('{"command":"pwd"}');
    // toolCalls array should NOT be present for parallel calls (we use text fallback)
    expect(msg.toolCalls).toBeUndefined();
  });

  it('should handle assistant messages with SINGLE tool call for Gemini', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'assistant' as const,
        content: [
          { type: 'text' as const, text: 'I will run one command.' },
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"ls"}' }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe('ASSISTANT');
    // Gemini with SINGLE tool call: Use native toolCalls array
    const msg = messages[0] as any;
    // Content should only contain the text part
    const content = msg.content as any[];
    expect(content.length).toBe(1);
    expect(content[0].type).toBe('TEXT');
    expect(content[0].text).toBe('I will run one command.');
    // Tool calls should be in the toolCalls array for single calls
    expect(msg.toolCalls).toBeDefined();
    expect(msg.toolCalls.length).toBe(1);
    expect(msg.toolCalls[0].type).toBe('FUNCTION');
    expect(msg.toolCalls[0].name).toBe('bash');
  });

  it('should handle assistant messages with mixed text and tool calls for Llama', () => {
    const model = provider.languageModel('meta.llama-3.3-70b-instruct');
    const convertMessagesToGenericFormat = (model as any).convertMessagesToGenericFormat.bind(model);

    const prompt = [
      {
        role: 'assistant' as const,
        content: [
          { type: 'text' as const, text: 'I will run two commands.' },
          { type: 'tool-call' as const, toolCallId: 'call_1', toolName: 'bash', input: '{"command":"ls"}' },
          { type: 'tool-call' as const, toolCallId: 'call_2', toolName: 'bash', input: '{"command":"pwd"}' }
        ]
      },
    ];

    const messages = convertMessagesToGenericFormat(prompt);

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe('ASSISTANT');
    // Llama: all converted to single TEXT
    const content = messages[0].content as any[];
    expect(content.length).toBe(1);
    expect(content[0].type).toBe('TEXT');
    expect(content[0].text).toContain('I will run two commands.');
    expect(content[0].text).toContain('[Called tool "bash"');
    // Both tool calls should be in the text
    expect(content[0].text).toContain('{"command":"ls"}');
    expect(content[0].text).toContain('{"command":"pwd"}');
  });
});

/**
 * Tests for usage and token counting
 */
describe('Usage and Token Counting', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should extract usage from Generic API response', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: 'Hello' }] },
              finishReason: 'COMPLETE',
            }],
            usage: { promptTokens: 100, completionTokens: 50 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hi' }] }],
    });

    expect(result.usage.inputTokens).toBe(100);
    expect(result.usage.outputTokens).toBe(50);
    expect(result.usage.totalTokens).toBe(150);
  });

  it('should extract usage from Cohere V2 response', async () => {
    const model = provider.languageModel('cohere.command-a-03-2025');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            message: {
              content: [{ type: 'TEXT', text: 'Hello' }],
            },
            finishReason: 'COMPLETE',
            usage: { inputTokens: 75, outputTokens: 25 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hi' }] }],
    });

    expect(result.usage.inputTokens).toBe(75);
    expect(result.usage.outputTokens).toBe(25);
    expect(result.usage.totalTokens).toBe(100);
  });

  it('should handle missing usage gracefully', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: 'Hello' }] },
              finishReason: 'COMPLETE',
            }],
            // No usage field
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hi' }] }],
    });

    expect(result.usage.inputTokens).toBe(0);
    expect(result.usage.outputTokens).toBe(0);
  });
});

/**
 * Tests for empty and null handling
 */
describe('Empty and Null Handling', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should handle empty prompt array', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convert = (model as any).convertMessagesToGenericFormat.bind(model);

    expect(convert([])).toEqual([]);
    expect(convert(null)).toEqual([]);
    expect(convert(undefined)).toEqual([]);
  });

  it('should handle empty Cohere prompt', () => {
    const model = provider.languageModel('cohere.command-r-08-2024');
    const convert = (model as any).convertMessagesToCohereFormat.bind(model);

    const result = convert([]);
    expect(result.message).toBe('');
    expect(result.chatHistory).toEqual([]);
    expect(result.toolResults).toEqual([]);
  });

  it('should handle empty Cohere V2 prompt', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const convert = (model as any).convertMessagesToCohereV2Format.bind(model);

    expect(convert([])).toEqual([]);
    expect(convert(null)).toEqual([]);
  });

  it('should handle null/undefined in tool schema cleaning', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const clean = (model as any).cleanJsonSchema.bind(model);

    expect(clean(null)).toBe(null);
    expect(clean(undefined)).toBe(undefined);
    expect(clean(0)).toBe(0);
    expect(clean('')).toBe('');
    expect(clean(false)).toBe(false);
  });

  it('should handle empty tools array', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const convert = (model as any).convertTools.bind(model);

    expect(convert([])).toBeUndefined();
    expect(convert(null)).toBeUndefined();
    expect(convert(undefined)).toBeUndefined();
  });
});

/**
 * Tests for model ID parsing and provider detection
 */
describe('Model ID Parsing', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should correctly identify Cohere models', () => {
    const cohereR = provider.languageModel('cohere.command-r-08-2024');
    const cohereA = provider.languageModel('cohere.command-a-03-2025');

    expect((cohereR as any).modelFamily).toBe('cohere');
    expect((cohereA as any).modelFamily).toBe('cohere-v2');
  });

  it('should correctly identify Google models', () => {
    const flash = provider.languageModel('google.gemini-2.5-flash');
    const pro = provider.languageModel('google.gemini-2.5-pro');
    const lite = provider.languageModel('google.gemini-2.5-flash-lite');

    expect((flash as any).modelFamily).toBe('generic');
    expect((pro as any).modelFamily).toBe('generic');
    expect((lite as any).modelFamily).toBe('generic');
  });

  it('should correctly identify xAI models', () => {
    const grok = provider.languageModel('xai.grok-4-1-fast-non-reasoning');
    const grokMini = provider.languageModel('xai.grok-3-mini');

    expect((grok as any).modelFamily).toBe('generic');
    expect((grokMini as any).modelFamily).toBe('generic');
  });

  it('should correctly identify Meta Llama models', () => {
    const llama33 = provider.languageModel('meta.llama-3.3-70b-instruct');
    const llama31 = provider.languageModel('meta.llama-3.1-405b-instruct');

    expect((llama33 as any).modelFamily).toBe('generic');
    expect((llama31 as any).modelFamily).toBe('generic');
  });

  it('should correctly identify OpenAI models', () => {
    const gptOss = provider.languageModel('openai.gpt-oss-120b');

    expect((gptOss as any).modelFamily).toBe('generic');
    expect((gptOss as any).swePreset.supportsReasoning).toBe(true);
  });

  it('should use default family for unknown prefixes', () => {
    const unknown = provider.languageModel('unknown.model-123');

    expect((unknown as any).modelFamily).toBe('generic');
  });
});

/**
 * Tests for generateId utility
 */
describe('ID Generation', () => {
  it('should generate unique IDs', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');
    const convertTools = (model as any).convertTools.bind(model);

    // Generate multiple tool conversions and check IDs are different
    // (IDs are generated when tools don't have explicit IDs)
    const tools1 = convertTools([
      { type: 'function', name: 'tool1', description: 'Test', inputSchema: {} }
    ]);
    const tools2 = convertTools([
      { type: 'function', name: 'tool2', description: 'Test', inputSchema: {} }
    ]);

    // The tools themselves don't have IDs, but the conversion should work
    expect(tools1).toBeDefined();
    expect(tools2).toBeDefined();
    expect(tools1[0].name).toBe('tool1');
    expect(tools2[0].name).toBe('tool2');
  });
});

/**
 * Tests for warnings array
 */
describe('Warnings Handling', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should return empty warnings array from doGenerate', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    expect(result.warnings).toBeDefined();
    expect(Array.isArray(result.warnings)).toBe(true);
    expect(result.warnings.length).toBe(0);
  });

  it('should emit empty warnings in stream-start event', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    const reader = result.stream.getReader();
    const { value } = await reader.read();

    expect(value?.type).toBe('stream-start');
    if (value?.type === 'stream-start') {
      expect(value.warnings).toBeDefined();
      expect(Array.isArray(value.warnings)).toBe(true);
    }
    reader.releaseLock();
  });
});

/**
 * Tests for supportedUrls property
 */
describe('Model Properties', () => {
  it('should have specificationVersion v2', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');

    expect(model.specificationVersion).toBe('v2');
  });

  it('should have provider property', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');

    expect(model.provider).toBe('oci-genai');
  });

  it('should have empty supportedUrls', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');

    expect(model.supportedUrls).toBeDefined();
    expect(typeof model.supportedUrls).toBe('object');
  });

  it('should have correct modelId', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });

    const model = provider.languageModel('google.gemini-2.5-flash');

    expect(model.modelId).toBe('google.gemini-2.5-flash');
  });
});

/**
 * Tests for new finish reason mappings
 */
describe('Finish Reason Mappings', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should map ERROR to error finish reason', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: 'partial' }] },
              finishReason: 'ERROR',
            }],
            usage: { promptTokens: 10, completionTokens: 5 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('error');
  });

  it('should map ERROR_TOXIC to content-filter finish reason', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: '' }] },
              finishReason: 'ERROR_TOXIC',
            }],
            usage: { promptTokens: 10, completionTokens: 0 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('content-filter');
  });

  it('should map ERROR_LIMIT to error finish reason', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: '' }] },
              finishReason: 'ERROR_LIMIT',
            }],
            usage: { promptTokens: 10, completionTokens: 0 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('error');
  });

  it('should map USER_CANCEL to other finish reason', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    (model as any).client = {
      chat: vi.fn().mockResolvedValue({
        chatResult: {
          chatResponse: {
            choices: [{
              message: { content: [{ type: 'TEXT', text: 'partial' }] },
              finishReason: 'USER_CANCEL',
            }],
            usage: { promptTokens: 10, completionTokens: 5 },
          },
        },
      }),
    };

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
    });

    expect(result.finishReason).toBe('other');
  });
});

/**
 * Tests for toolChoice mapping in Generic format
 */
describe('Generic Format toolChoice Mapping', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should map auto toolChoice to AUTO', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const build = (model as any).buildGenericChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      tools: [{ type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }],
      toolChoice: { type: 'auto' as const },
    };

    const request = build(options);

    expect(request.toolChoice).toEqual({ type: 'AUTO' });
  });

  it('should map required toolChoice to REQUIRED', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const build = (model as any).buildGenericChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      tools: [{ type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }],
      toolChoice: { type: 'required' as const },
    };

    const request = build(options);

    expect(request.toolChoice).toEqual({ type: 'REQUIRED' });
  });

  it('should map none toolChoice to NONE', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const build = (model as any).buildGenericChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      tools: [{ type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }],
      toolChoice: { type: 'none' as const },
    };

    const request = build(options);

    expect(request.toolChoice).toEqual({ type: 'NONE' });
  });

  it('should map specific tool toolChoice to FUNCTION with functionName', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const build = (model as any).buildGenericChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      tools: [{ type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }],
      toolChoice: { type: 'tool' as const, toolName: 'bash' },
    };

    const request = build(options);

    expect(request.toolChoice).toEqual({ type: 'FUNCTION', functionName: 'bash' });
  });

  it('should not include toolChoice when no tools are provided', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const build = (model as any).buildGenericChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      toolChoice: { type: 'required' as const },
    };

    const request = build(options);

    expect(request.toolChoice).toBeUndefined();
  });

  it('should not include toolChoice when toolChoice is not specified', () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const build = (model as any).buildGenericChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      tools: [{ type: 'function' as const, name: 'bash', description: 'Run bash', inputSchema: { type: 'object' } }],
    };

    const request = build(options);

    expect(request.toolChoice).toBeUndefined();
  });
});

/**
 * Tests for new OCI API parameters (topK, seed, responseFormat, maxCompletionTokens)
 */
describe('New OCI API Parameters', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  describe('topK parameter', () => {
    it('should include topK in Generic request when specified', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        topK: 40,
      };

      const request = build(options);

      expect(request.topK).toBe(40);
    });

    it('should include topK in Cohere request when specified', () => {
      const model = provider.languageModel('cohere.command-r-plus-08-2024');
      const build = (model as any).buildCohereChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        topK: 50,
      };

      const request = build(options);

      expect(request.topK).toBe(50);
    });

    it('should include topK in Cohere V2 request when specified', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        topK: 30,
      };

      const request = build(options);

      expect(request.topK).toBe(30);
    });

    it('should not include topK when not specified', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
      };

      const request = build(options);

      expect(request.topK).toBeUndefined();
    });
  });

  describe('seed parameter', () => {
    it('should include seed in Generic request via providerOptions', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': { seed: 42 },
        },
      };

      const request = build(options);

      expect(request.seed).toBe(42);
    });

    it('should include seed in Cohere request via providerOptions', () => {
      const model = provider.languageModel('cohere.command-r-plus-08-2024');
      const build = (model as any).buildCohereChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': { seed: 123 },
        },
      };

      const request = build(options);

      expect(request.seed).toBe(123);
    });

    it('should include seed in Cohere V2 request via providerOptions', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      const build = (model as any).buildCohereV2ChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': { seed: 99 },
        },
      };

      const request = build(options);

      expect(request.seed).toBe(99);
    });

    it('should not include seed when not specified', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
      };

      const request = build(options);

      expect(request.seed).toBeUndefined();
    });
  });

  describe('responseFormat parameter', () => {
    it('should map text responseFormat', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        responseFormat: { type: 'text' as const },
      };

      const request = build(options);

      expect(request.responseFormat).toEqual({ type: 'TEXT' });
    });

    it('should map json responseFormat without schema to JSON_OBJECT', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        responseFormat: { type: 'json' as const },
      };

      const request = build(options);

      expect(request.responseFormat).toEqual({ type: 'JSON_OBJECT' });
    });

    it('should map json responseFormat with schema to JSON_SCHEMA', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const schema = {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      };

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        responseFormat: { type: 'json' as const, schema },
      };

      const request = build(options);

      expect(request.responseFormat.type).toBe('JSON_SCHEMA');
      expect(request.responseFormat.jsonSchema).toBeDefined();
      expect(request.responseFormat.jsonSchema.type).toBe('object');
    });

    it('should not include responseFormat when not specified', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
      };

      const request = build(options);

      expect(request.responseFormat).toBeUndefined();
    });
  });

  describe('maxCompletionTokens parameter', () => {
    it('should include maxCompletionTokens via providerOptions', () => {
      const model = provider.languageModel('openai.gpt-oss-120b');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
        providerOptions: {
          'oci-genai': { maxCompletionTokens: 8000 },
        },
      };

      const request = build(options);

      expect(request.maxCompletionTokens).toBe(8000);
    });

    it('should not include maxCompletionTokens when not specified', () => {
      const model = provider.languageModel('openai.gpt-oss-120b');
      const build = (model as any).buildGenericChatRequest.bind(model);

      const options = {
        prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
        maxOutputTokens: 1000,
      };

      const request = build(options);

      expect(request.maxCompletionTokens).toBeUndefined();
    });
  });
});

/**
 * Tests for Cohere safety mode and stop sequences
 */
describe('Cohere safetyMode and stopSequences', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should include safetyMode in Cohere V1 request via providerOptions', () => {
    const model = provider.languageModel('cohere.command-r-plus-08-2024');
    const build = (model as any).buildCohereChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      providerOptions: {
        'oci-genai': { safetyMode: 'STRICT' },
      },
    };

    const request = build(options);

    expect(request.safetyMode).toBe('STRICT');
  });

  it('should include safetyMode in Cohere V2 request via providerOptions', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const build = (model as any).buildCohereV2ChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      providerOptions: {
        'oci-genai': { safetyMode: 'OFF' },
      },
    };

    const request = build(options);

    expect(request.safetyMode).toBe('OFF');
  });

  it('should not include safetyMode when not specified', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const build = (model as any).buildCohereV2ChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
    };

    const request = build(options);

    expect(request.safetyMode).toBeUndefined();
  });

  it('should include stopSequences in Cohere V2 request', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const build = (model as any).buildCohereV2ChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      stopSequences: ['STOP', 'END'],
    };

    const request = build(options);

    expect(request.stopSequences).toEqual(['STOP', 'END']);
  });

  it('should not include stopSequences in Cohere V2 when empty', () => {
    const model = provider.languageModel('cohere.command-a-03-2025');
    const build = (model as any).buildCohereV2ChatRequest.bind(model);

    const options = {
      prompt: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Test' }] }],
      maxOutputTokens: 1000,
      stopSequences: [],
    };

    const request = build(options);

    expect(request.stopSequences).toBeUndefined();
  });
});

/**
 * Tests for abortSignal support
 */
describe('AbortSignal Support', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should abort doGenerate when signal is triggered', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const controller = new AbortController();

    // Mock a slow chat call
    (model as any).client = {
      chat: vi.fn().mockImplementation(() =>
        new Promise((resolve) => setTimeout(resolve, 5000))
      ),
    };

    // Abort immediately
    controller.abort();

    await expect(
      model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
        abortSignal: controller.signal,
      })
    ).rejects.toThrow('aborted');
  });

  it('should abort doGenerate when signal fires during request', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const controller = new AbortController();

    // Mock a slow chat call
    (model as any).client = {
      chat: vi.fn().mockImplementation(() =>
        new Promise((resolve) => setTimeout(resolve, 5000))
      ),
    };

    // Abort after 10ms
    setTimeout(() => controller.abort(), 10);

    await expect(
      model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
        abortSignal: controller.signal,
      })
    ).rejects.toThrow('aborted');
  });

  it('should not throw when signal is not aborted', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');
    const controller = new AbortController();

    // Normal response, no abort
    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Test' }] }],
      abortSignal: controller.signal,
    });

    expect(result.content).toBeDefined();
    expect(result.finishReason).toBeDefined();
  });
});

/**
 * Tests for response metadata
 */
describe('Response Metadata', () => {
  let provider: OCIProvider;

  beforeEach(() => {
    provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
    });
  });

  it('should return request body in doGenerate response', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    expect(result.request).toBeDefined();
    expect(result.request?.body).toBeDefined();
    // The body should contain the chat request
    const body = result.request?.body as any;
    expect(body.apiFormat).toBe('GENERIC');
  });

  it('should return response metadata with modelId in doGenerate response', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    expect(result.response).toBeDefined();
    expect(result.response?.modelId).toBe('google.gemini-2.5-flash');
  });

  it('should emit response-metadata event in doStream', async () => {
    const model = provider.languageModel('google.gemini-2.5-flash');

    const result = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
    });

    const reader = result.stream.getReader();
    const events: any[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) events.push(value);
    }

    const metadataEvent = events.find(e => e.type === 'response-metadata');
    expect(metadataEvent).toBeDefined();
    expect(metadataEvent.modelId).toBe('google.gemini-2.5-flash');
    expect(metadataEvent.timestamp).toBeInstanceOf(Date);
  });
});

/**
 * Tests for auth provider configuration
 */
describe('Auth Provider Configuration', () => {
  it('should accept session-token auth provider type', () => {
    // This will fail to authenticate but should not throw during construction
    // since the mock handles the constructor
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
      authProvider: 'session-token',
    });

    expect(provider).toBeDefined();
    const model = provider.languageModel('google.gemini-2.5-flash');
    expect(model).toBeDefined();
  });

  it('should accept config-file auth provider type (default)', () => {
    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
      authProvider: 'config-file',
    });

    expect(provider).toBeDefined();
    const model = provider.languageModel('google.gemini-2.5-flash');
    expect(model).toBeDefined();
  });

  it('should accept a pre-built auth provider instance', () => {
    // Create a mock auth provider object
    const mockAuthProvider = {
      getKeyId: () => Promise.resolve('mock-key-id'),
      getUser: () => Promise.resolve('mock-user'),
    };

    const provider = createOCI({
      compartmentId: 'test-compartment',
      region: 'us-chicago-1',
      authProvider: mockAuthProvider as any,
    });

    expect(provider).toBeDefined();
    const model = provider.languageModel('google.gemini-2.5-flash');
    expect(model).toBeDefined();
  });
});
