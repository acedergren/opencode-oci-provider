/**
 * Test reasoning/thinking output from OCI GenAI models
 * Run: npx tsx test-reasoning.ts
 */
import { OCIProvider } from './src/index';

const provider = new OCIProvider({
  compartmentId: process.env.OCI_COMPARTMENT_ID,
  region: 'us-chicago-1',
});

async function testReasoning(modelId: string) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing reasoning: ${modelId}`);

  const model = provider.languageModel(modelId);

  try {
    // Test streaming to see reasoning events
    const { stream } = await model.doStream({
      mode: { type: 'regular' },
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'What is 15 * 23? Think step by step.' }],
        },
      ],
      inputFormat: 'messages',
    });

    const reader = stream.getReader();
    let reasoning = '';
    let text = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      if (value.type === 'reasoning-start') {
        console.log('\n📝 Reasoning started...');
      } else if (value.type === 'reasoning-delta') {
        reasoning += value.delta;
        process.stdout.write('.');
      } else if (value.type === 'reasoning-end') {
        console.log('\n✅ Reasoning complete');
      } else if (value.type === 'text-delta') {
        text += value.delta;
      } else if (value.type === 'finish') {
        console.log(`\nFinish reason: ${value.finishReason}`);
      }
    }

    if (reasoning) {
      console.log('\n💭 REASONING CONTENT:');
      console.log(reasoning.substring(0, 500) + (reasoning.length > 500 ? '...' : ''));
    } else {
      console.log('\n(No reasoning content returned)');
    }

    console.log('\n📄 TEXT RESPONSE:');
    console.log(text.substring(0, 300) + (text.length > 300 ? '...' : ''));

    return { hasReasoning: !!reasoning, text };
  } catch (error: any) {
    console.error('❌ Error:', error.message);
    return { hasReasoning: false, error: error.message };
  }
}

async function main() {
  console.log('Testing OCI GenAI Reasoning Support');
  console.log('Region: us-chicago-1');

  // Test models that should have reasoning
  const modelsToTest = [
    'google.gemini-2.5-flash',  // Always has thinking
    'xai.grok-3-fast',          // Has reasoningEffort
  ];

  for (const model of modelsToTest) {
    await testReasoning(model);
  }
}

main();
