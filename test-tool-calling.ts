/**
 * Direct test of tool calling with OCI provider
 * This bypasses OpenCode to verify the provider itself handles tools correctly
 */
import { createOCI } from './dist/index.js';
import { streamText } from 'ai';
import { z } from 'zod';

async function testToolCalling() {
  console.log('🔧 Testing OCI Provider Tool Calling\n');

  const provider = createOCI({
    compartmentId: process.env.OCI_COMPARTMENT_ID!,
    region: process.env.OCI_REGION!,
    servingMode: 'on-demand',
  });

  // Test models
  const modelsToTest = [
    'cohere.command-a-03-2025',
    'google.gemini-2.5-flash',
    'xai.grok-4-1-fast',
    'meta.llama-3.3-70b-instruct',
  ];

  // Tool definition using AI SDK v5 format with Zod schema
  const tools = {
    getCurrentWeather: {
      description: 'Get the current weather in a location',
      inputSchema: z.object({
        location: z.string().describe('The city and state, e.g. San Francisco, CA'),
        unit: z.enum(['celsius', 'fahrenheit']).optional().describe('The unit of temperature'),
      }),
      execute: async ({ location, unit = 'celsius' }) => {
        console.log(`Tool called with location=${location}, unit=${unit}`);
        return { location, temperature: 72, unit };
      },
    },
  };

  for (const modelId of modelsToTest) {
    console.log(`\n📊 Testing: ${modelId}`);
    console.log('='.repeat(60));

    try {
      const model = provider.languageModel(modelId);

      const result = await streamText({
        model,
        prompt: 'What is the weather like in San Francisco?',
        tools,
        maxSteps: 3,
      });

      // Consume the stream
      let fullText = '';
      let toolCallsCount = 0;

      for await (const chunk of result.textStream) {
        fullText += chunk;
      }

      // Wait for completion
      const finalResult = await result.response;

      console.log('✅ Model responded successfully');
      console.log('Response:', fullText.slice(0, 100) + (fullText.length > 100 ? '...' : ''));

      // Check tool calls from the final result
      const toolResults = await result.toolResults;
      if (toolResults && toolResults.length > 0) {
        console.log('✅ TOOL CALLING WORKS! Called', toolResults.length, 'tool(s)');
      } else {
        console.log('❌ NO TOOLS CALLED - Model did not use tools');
      }

    } catch (error: any) {
      console.log(`❌ Error: ${error.message}`);
    }
  }
}

testToolCalling().catch(console.error);
