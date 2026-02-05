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
      const model = provider.languageModel('xai.grok-4-1-fast');
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
      const model = provider.languageModel('xai.grok-4-1-fast');
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
      const model = provider.languageModel('xai.grok-4-1-fast');
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
    it('should detect cohere model family', () => {
      const model = provider.languageModel('cohere.command-a-03-2025');
      expect((model as any).modelFamily).toBe('cohere');
    });

    it('should use generic family for Google models', () => {
      const model = provider.languageModel('google.gemini-2.5-flash');
      expect((model as any).modelFamily).toBe('generic');
    });

    it('should use generic family for xAI models', () => {
      const model = provider.languageModel('xai.grok-4-1-fast');
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
    const serialized = ociModels.BaseChatRequest.getJsonObj(cohereRequest);

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
    const serialized = ociModels.ChatDetails.getJsonObj(chatDetails);

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
    const model = provider.languageModel('xai.grok-4-1-fast');
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
    const model = provider.languageModel('xai.grok-4-1-fast');
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
    it('should set supportsReasoning=false for xAI Grok models', () => {
      const model = provider.languageModel('xai.grok-4-1-fast');
      const swePreset = (model as any).swePreset;

      // Grok models through OCI don't support reasoning_effort parameter
      expect(swePreset.supportsReasoning).toBe(false);
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
      const model = provider.languageModel('xai.grok-4-1-fast');
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
      const model = provider.languageModel('xai.grok-4-1-fast');
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
      const model = provider.languageModel('xai.grok-4-1-fast');

      // The mock returns a response, we're testing the extraction logic
      const result = await model.doGenerate({
        mode: { type: 'regular' },
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'Think step by step' }] }],
        inputFormat: 'messages',
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
