#!/usr/bin/env node
/**
 * OpenCode OCI Setup Wizard
 *
 * Interactive CLI to configure OCI GenAI provider for OpenCode
 * Features dynamic model discovery to show only available models
 */
import { select, input, confirm } from '@inquirer/prompts';
import chalk from 'chalk';
import * as fs from 'fs';
import * as path from 'path';
import * as common from 'oci-common';
import * as genai from 'oci-generativeai';
import * as inference from 'oci-generativeaiinference';
import { REGIONS, formatRegionChoice, getAllRegionIds } from './data/regions.js';

interface SetupConfig {
  profile: string;
  region: string;
  compartmentId: string;
  servingMode: 'on-demand' | 'dedicated';
  modelId?: string;
  endpointId?: string;
  customModelName?: string;
}

interface ModelInfo {
  id: string;
  name: string;
  vendor: string;
  capabilities: string[];
}

/**
 * Known foundation models - we'll probe these to see which are available
 */
const KNOWN_MODELS: ModelInfo[] = [
  // Cohere models
  { id: 'cohere.command-a-reasoning-08-2025', name: 'Cohere Command A Reasoning', vendor: 'cohere', capabilities: ['chat', 'reasoning'] },
  { id: 'cohere.command-a-vision-07-2025', name: 'Cohere Command A Vision', vendor: 'cohere', capabilities: ['chat', 'vision'] },
  { id: 'cohere.command-a-03-2025', name: 'Cohere Command A', vendor: 'cohere', capabilities: ['chat'] },
  { id: 'cohere.command-r-08-2024', name: 'Cohere Command R', vendor: 'cohere', capabilities: ['chat'] },
  { id: 'cohere.command-r-plus-08-2024', name: 'Cohere Command R+', vendor: 'cohere', capabilities: ['chat'] },

  // Google models
  { id: 'google.gemini-2.5-pro', name: 'Google Gemini 2.5 Pro', vendor: 'google', capabilities: ['chat', 'vision'] },
  { id: 'google.gemini-2.5-flash', name: 'Google Gemini 2.5 Flash', vendor: 'google', capabilities: ['chat', 'vision'] },
  { id: 'google.gemini-2.5-flash-lite', name: 'Google Gemini 2.5 Flash Lite', vendor: 'google', capabilities: ['chat'] },
  { id: 'google.gemini-2.0-flash-001', name: 'Google Gemini 2.0 Flash', vendor: 'google', capabilities: ['chat'] },
  { id: 'google.gemini-1.5-pro-002', name: 'Google Gemini 1.5 Pro', vendor: 'google', capabilities: ['chat', 'vision'] },

  // Meta models
  { id: 'meta.llama-4-maverick-17b-128e-instruct-fp8', name: 'Meta Llama 4 Maverick 17B', vendor: 'meta', capabilities: ['chat'] },
  { id: 'meta.llama-4-scout-17b-16e-instruct', name: 'Meta Llama 4 Scout 17B', vendor: 'meta', capabilities: ['chat'] },
  { id: 'meta.llama-3.3-70b-instruct', name: 'Meta Llama 3.3 70B', vendor: 'meta', capabilities: ['chat'] },
  { id: 'meta.llama-3.2-90b-vision-instruct', name: 'Meta Llama 3.2 90B Vision', vendor: 'meta', capabilities: ['chat', 'vision'] },
  { id: 'meta.llama-3.2-11b-vision-instruct', name: 'Meta Llama 3.2 11B Vision', vendor: 'meta', capabilities: ['chat', 'vision'] },
  { id: 'meta.llama-3.1-405b-instruct', name: 'Meta Llama 3.1 405B', vendor: 'meta', capabilities: ['chat'] },

  // xAI models (US regions only: Ashburn, Chicago, Phoenix)
  { id: 'xai.grok-4-1-fast-reasoning', name: 'xAI Grok 4.1 Fast (Reasoning)', vendor: 'xai', capabilities: ['chat', 'vision', 'reasoning'] },
  { id: 'xai.grok-4-1-fast-non-reasoning', name: 'xAI Grok 4.1 Fast', vendor: 'xai', capabilities: ['chat', 'vision'] },
  { id: 'xai.grok-4-fast', name: 'xAI Grok 4 Fast', vendor: 'xai', capabilities: ['chat'] },
  { id: 'xai.grok-4', name: 'xAI Grok 4', vendor: 'xai', capabilities: ['chat'] },
  { id: 'xai.grok-3-fast', name: 'xAI Grok 3 Fast', vendor: 'xai', capabilities: ['chat'] },
  { id: 'xai.grok-3', name: 'xAI Grok 3', vendor: 'xai', capabilities: ['chat'] },
  { id: 'xai.grok-3-mini', name: 'xAI Grok 3 Mini (Reasoning)', vendor: 'xai', capabilities: ['chat', 'reasoning'] },
  { id: 'xai.grok-3-mini-fast', name: 'xAI Grok 3 Mini Fast (Reasoning)', vendor: 'xai', capabilities: ['chat', 'reasoning'] },
  { id: 'xai.grok-code-fast-1', name: 'xAI Grok Code Fast', vendor: 'xai', capabilities: ['chat', 'code'] },

  // OpenAI models
  { id: 'openai.gpt-oss-120b', name: 'OpenAI GPT OSS 120B', vendor: 'openai', capabilities: ['chat'] },
  { id: 'openai.gpt-oss-20b', name: 'OpenAI GPT OSS 20B', vendor: 'openai', capabilities: ['chat'] },
];

async function main() {
  console.log(chalk.bold.cyan('\n🚀 OpenCode OCI GenAI Setup\n'));
  console.log(chalk.gray('This wizard will configure OCI GenAI for use with OpenCode.\n'));

  const config: Partial<SetupConfig> = {};

  // Step 1: Select OCI config profile
  config.profile = await selectProfile();

  // Step 2: Select region
  config.region = await selectRegion();

  // Step 3: Get compartment ID
  config.compartmentId = await getCompartmentId();

  // Step 4: Select serving mode
  config.servingMode = await selectServingMode();

  if (config.servingMode === 'on-demand') {
    // Step 5a: Discover and select model for on-demand
    config.modelId = await discoverAndSelectModel(config as SetupConfig);
  } else {
    // Step 5b: Select endpoint for dedicated
    const endpoint = await selectDedicatedEndpoint(config as SetupConfig);
    if (endpoint) {
      config.endpointId = endpoint.id;
      config.customModelName = endpoint.name;
    }
  }

  // Step 6: Test configuration
  const testPassed = await testConfiguration(config as SetupConfig);

  if (!testPassed) {
    console.log(chalk.yellow('\n⚠️  Configuration test failed. Please check your settings.'));
    const proceed = await confirm({
      message: 'Save configuration anyway?',
      default: false,
    });
    if (!proceed) {
      console.log(chalk.gray('Setup cancelled.'));
      process.exit(1);
    }
  }

  // Step 7: Save configuration
  await saveConfiguration(config as SetupConfig);

  console.log(chalk.bold.green('\n✅ Setup complete!\n'));
  console.log(chalk.gray('Configuration saved. You can now use OCI GenAI with OpenCode.\n'));
}

async function selectProfile(): Promise<string> {
  const ociConfigPath = path.join(process.env.HOME || '', '.oci', 'config');

  let profiles = ['DEFAULT'];

  if (fs.existsSync(ociConfigPath)) {
    const content = fs.readFileSync(ociConfigPath, 'utf-8');
    const profileMatches = content.match(/^\[([^\]]+)\]/gm);
    if (profileMatches) {
      profiles = profileMatches.map(m => m.slice(1, -1));
    }
  }

  if (profiles.length === 1) {
    console.log(chalk.gray(`Using OCI profile: ${profiles[0]}`));
    return profiles[0];
  }

  return select({
    message: 'Select OCI config profile:',
    choices: profiles.map(p => ({ value: p, name: p })),
  });
}

async function selectRegion(): Promise<string> {
  const regionIds = getAllRegionIds();

  return select({
    message: 'Select OCI region:',
    choices: regionIds.map(id => ({
      value: id,
      name: formatRegionChoice(id),
    })),
    pageSize: 15,
  });
}

async function getCompartmentId(): Promise<string> {
  const envCompartmentId = process.env.OCI_COMPARTMENT_ID;

  if (envCompartmentId) {
    const useEnv = await confirm({
      message: `Use compartment ID from environment? (${envCompartmentId.slice(0, 30)}...)`,
      default: true,
    });
    if (useEnv) return envCompartmentId;
  }

  return input({
    message: 'Enter compartment OCID:',
    validate: (value) => {
      if (!value.startsWith('ocid1.compartment.') && !value.startsWith('ocid1.tenancy.')) {
        return 'Invalid OCID. Must start with ocid1.compartment. or ocid1.tenancy.';
      }
      return true;
    },
  });
}

async function selectServingMode(): Promise<'on-demand' | 'dedicated'> {
  return select({
    message: 'Select serving mode:',
    choices: [
      {
        value: 'on-demand' as const,
        name: 'On-Demand (pay-per-token, instant access)',
        description: 'Best for development and variable workloads',
      },
      {
        value: 'dedicated' as const,
        name: 'Dedicated AI Cluster (custom endpoints)',
        description: 'For production workloads with consistent traffic',
      },
    ],
  });
}

/**
 * Probe the OCI GenAI API to discover which models are actually available
 */
async function discoverAvailableModels(config: SetupConfig): Promise<ModelInfo[]> {
  console.log(chalk.gray('\n🔍 Discovering available models in your region...'));

  const provider = new common.ConfigFileAuthenticationDetailsProvider(
    undefined,
    config.profile
  );

  const client = new inference.GenerativeAiInferenceClient({
    authenticationDetailsProvider: provider,
  });

  if (config.region) {
    client.region = common.Region.fromRegionId(config.region);
  }

  const availableModels: ModelInfo[] = [];
  const checkPromises: Promise<void>[] = [];

  // Check models in parallel with concurrency limit
  const concurrencyLimit = 5;
  let activeChecks = 0;

  for (const model of KNOWN_MODELS) {
    const checkModel = async () => {
      try {
        // Send a minimal request to see if the model exists
        await client.chat({
          chatDetails: {
            compartmentId: config.compartmentId,
            servingMode: {
              servingType: 'ON_DEMAND',
              modelId: model.id,
            },
            chatRequest: {
              apiFormat: 'GENERIC',
              messages: [
                {
                  role: 'USER',
                  content: [{ type: 'TEXT', text: 'hi' } as inference.models.TextContent],
                },
              ],
              maxTokens: 1,
            } as inference.models.GenericChatRequest,
          },
        });

        // If we get here, model is available
        availableModels.push(model);
        process.stdout.write(chalk.green('✓'));
      } catch (error: any) {
        const message = error.message || '';
        // Model not found errors
        if (message.includes('not found') || message.includes('Entity with key')) {
          process.stdout.write(chalk.gray('·'));
        } else {
          // Other errors might still mean the model exists but something else failed
          // For now, mark as unavailable
          process.stdout.write(chalk.yellow('?'));
        }
      }
    };

    checkPromises.push(checkModel());

    // Rate limit
    if (checkPromises.length >= concurrencyLimit) {
      await Promise.all(checkPromises);
      checkPromises.length = 0;
    }
  }

  // Wait for remaining checks
  if (checkPromises.length > 0) {
    await Promise.all(checkPromises);
  }

  console.log('\n');

  return availableModels;
}

/**
 * Discover available models and let user select one
 */
async function discoverAndSelectModel(config: SetupConfig): Promise<string> {
  const availableModels = await discoverAvailableModels(config);

  if (availableModels.length === 0) {
    console.log(chalk.yellow('No foundation models found in this region.'));
    console.log(chalk.gray('You may need to check your region or use a custom model ID.\n'));

    return input({
      message: 'Enter model ID manually:',
      validate: (value) => {
        if (!value.includes('.')) {
          return 'Model ID should be in format: provider.model-name';
        }
        return true;
      },
    });
  }

  console.log(chalk.green(`Found ${availableModels.length} available model(s)\n`));

  // Group by vendor
  const byVendor = availableModels.reduce((acc, m) => {
    if (!acc[m.vendor]) acc[m.vendor] = [];
    acc[m.vendor].push(m);
    return acc;
  }, {} as Record<string, ModelInfo[]>);

  // Build choices grouped by vendor
  const choices: { value: string; name: string }[] = [];

  for (const [vendor, models] of Object.entries(byVendor)) {
    const vendorName = vendor.charAt(0).toUpperCase() + vendor.slice(1);
    choices.push({ value: `header-${vendor}`, name: chalk.bold.blue(`── ${vendorName} ──`) });
    for (const model of models) {
      const caps = model.capabilities.join(', ');
      choices.push({
        value: model.id,
        name: `  ${model.name} ${chalk.gray(`[${caps}]`)}`,
      });
    }
  }

  choices.push({ value: 'custom', name: chalk.gray('Enter custom model ID...') });

  const modelChoice = await select({
    message: 'Select model:',
    choices,
    pageSize: 20,
  });

  // Skip header selections
  if (modelChoice.startsWith('header-')) {
    return discoverAndSelectModel(config);
  }

  if (modelChoice === 'custom') {
    return input({
      message: 'Enter model ID:',
      validate: (value) => {
        if (!value.includes('.')) {
          return 'Model ID should be in format: provider.model-name';
        }
        return true;
      },
    });
  }

  return modelChoice;
}

async function selectDedicatedEndpoint(
  config: SetupConfig
): Promise<{ id: string; name: string } | null> {
  console.log(chalk.gray('\nFetching dedicated AI clusters...'));

  try {
    const provider = new common.ConfigFileAuthenticationDetailsProvider(
      undefined,
      config.profile
    );
    const client = new genai.GenerativeAiClient({
      authenticationDetailsProvider: provider,
    });

    if (config.region) {
      client.region = common.Region.fromRegionId(config.region);
    }

    // List clusters
    const clustersResponse = await client.listDedicatedAiClusters({
      compartmentId: config.compartmentId,
      lifecycleState: genai.models.DedicatedAiCluster.LifecycleState.Active,
    });

    const clusters = clustersResponse.dedicatedAiClusterCollection?.items || [];

    if (clusters.length === 0) {
      console.log(chalk.yellow('No active dedicated AI clusters found.'));
      console.log(chalk.gray('You can create one in the OCI Console.'));
      return null;
    }

    const clusterId = await select({
      message: 'Select dedicated AI cluster:',
      choices: clusters.map(c => ({
        value: c.id!,
        name: `${c.displayName} (${c.unitCount} units, ${c.unitShape})`,
      })),
    });

    // List endpoints for selected cluster
    const endpointsResponse = await client.listEndpoints({
      compartmentId: config.compartmentId,
      lifecycleState: genai.models.Endpoint.LifecycleState.Active,
    });

    // Filter endpoints by cluster ID manually since SDK may not support the parameter
    const allEndpoints = endpointsResponse.endpointCollection?.items || [];
    const endpoints = allEndpoints.filter(e => e.dedicatedAiClusterId === clusterId);

    if (endpoints.length === 0) {
      console.log(chalk.yellow('No active endpoints found on this cluster.'));
      return null;
    }

    const endpoint = await select({
      message: 'Select endpoint:',
      choices: endpoints.map(e => ({
        value: { id: e.id!, name: e.displayName || 'custom-endpoint' },
        name: `${e.displayName} (${e.modelId})`,
      })),
    });

    return endpoint;
  } catch (error) {
    console.log(chalk.red('Failed to fetch clusters/endpoints:'), error);
    return null;
  }
}

async function testConfiguration(config: SetupConfig): Promise<boolean> {
  console.log(chalk.gray('\nTesting configuration...'));

  try {
    const provider = new common.ConfigFileAuthenticationDetailsProvider(
      undefined,
      config.profile
    );

    const client = new inference.GenerativeAiInferenceClient({
      authenticationDetailsProvider: provider,
    });

    if (config.region) {
      client.region = common.Region.fromRegionId(config.region);
    }

    // Test with actual model
    const modelId = config.servingMode === 'on-demand' ? config.modelId! : undefined;
    const endpointId = config.servingMode === 'dedicated' ? config.endpointId : undefined;

    const servingMode = endpointId
      ? { servingType: 'DEDICATED' as const, endpointId }
      : { servingType: 'ON_DEMAND' as const, modelId: modelId! };

    await client.chat({
      chatDetails: {
        compartmentId: config.compartmentId,
        servingMode,
        chatRequest: {
          apiFormat: 'GENERIC',
          messages: [
            {
              role: 'USER',
              content: [{ type: 'TEXT', text: 'Say "test successful" in 3 words or less.' } as inference.models.TextContent],
            },
          ],
          maxTokens: 10,
        } as inference.models.GenericChatRequest,
      },
    });

    console.log(chalk.green('✓ Configuration valid - model responded successfully'));
    return true;
  } catch (error: any) {
    console.log(chalk.red('✗ Configuration test failed:'), error.message || error);
    return false;
  }
}

async function saveConfiguration(config: SetupConfig): Promise<void> {
  // Save .env file
  const envContent = `# OCI GenAI Configuration (generated by opencode-oci-setup)
OCI_REGION=${config.region}
OCI_COMPARTMENT_ID=${config.compartmentId}
OCI_CONFIG_PROFILE=${config.profile}
${config.servingMode === 'dedicated' && config.endpointId ? `OCI_GENAI_ENDPOINT_ID=${config.endpointId}` : '# OCI_GENAI_ENDPOINT_ID='}
`;

  const envPath = '.env.oci-genai';
  fs.writeFileSync(envPath, envContent);
  console.log(chalk.gray(`Saved: ${envPath}`));

  // Get model info for display name
  const modelInfo = KNOWN_MODELS.find(m => m.id === config.modelId);
  const modelName = config.servingMode === 'on-demand'
    ? modelInfo?.name || config.modelId!
    : config.customModelName || 'Custom Endpoint';

  const modelKey = config.servingMode === 'on-demand'
    ? config.modelId!
    : config.customModelName || 'custom-endpoint';

  const opencodeConfig = {
    provider: {
      oci: {
        npm: 'opencode-oci-provider',
        name: 'Oracle Cloud Infrastructure',
        options: {
          region: config.region,
          compartmentId: '${OCI_COMPARTMENT_ID}',
          ...(config.servingMode === 'dedicated' && config.endpointId
            ? {
                servingMode: 'dedicated',
                endpointId: config.endpointId,
              }
            : {}),
        },
        models: {
          [modelKey]: {
            name: modelName,
            type: 'chat',
            capabilities: {
              streaming: true,
              toolCalling: true,
            },
          },
        },
      },
    },
  };

  // Try to merge with existing opencode.json
  const opencodeJsonPath = 'opencode.json';
  let existingConfig: any = {};

  if (fs.existsSync(opencodeJsonPath)) {
    try {
      existingConfig = JSON.parse(fs.readFileSync(opencodeJsonPath, 'utf-8'));
    } catch {
      // Ignore parse errors
    }
  }

  const mergedConfig = {
    ...existingConfig,
    provider: {
      ...existingConfig.provider,
      ...opencodeConfig.provider,
    },
  };

  fs.writeFileSync(opencodeJsonPath, JSON.stringify(mergedConfig, null, 2) + '\n');
  console.log(chalk.gray(`Saved: ${opencodeJsonPath}`));
}

main().catch((error) => {
  console.error(chalk.red('Setup failed:'), error);
  process.exit(1);
});
