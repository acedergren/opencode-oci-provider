#!/usr/bin/env node
/**
 * OpenCode OCI Setup Wizard
 *
 * Interactive CLI to configure OCI GenAI provider for OpenCode
 */
import { select, input, confirm } from '@inquirer/prompts';
import chalk from 'chalk';
import * as fs from 'fs';
import * as path from 'path';
import * as common from 'oci-common';
import * as genai from 'oci-generativeai';
import { REGIONS, formatRegionChoice, getAllRegionIds, supportsXAI } from './data/regions.js';

interface SetupConfig {
  profile: string;
  region: string;
  compartmentId: string;
  servingMode: 'on-demand' | 'dedicated';
  modelId?: string;
  endpointId?: string;
  customModelName?: string;
}

const POPULAR_MODELS = [
  { value: 'cohere.command-r-plus-08-2024', name: 'Cohere Command R+ (Best quality)' },
  { value: 'cohere.command-r-08-2024', name: 'Cohere Command R (Balanced)' },
  { value: 'google.gemini-2.0-flash-001', name: 'Google Gemini 2.0 Flash (Fast)' },
  { value: 'google.gemini-1.5-pro-002', name: 'Google Gemini 1.5 Pro' },
  { value: 'xai.grok-2-1212', name: 'xAI Grok 2 (US only)' },
  { value: 'meta.llama-3.1-405b-instruct', name: 'Meta Llama 3.1 405B' },
  { value: 'meta.llama-3.1-70b-instruct', name: 'Meta Llama 3.1 70B' },
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
    // Step 5a: Select model for on-demand
    config.modelId = await selectModel(config.region);
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
      if (!value.startsWith('ocid1.compartment.')) {
        return 'Invalid compartment OCID. Must start with ocid1.compartment.';
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

async function selectModel(region: string): Promise<string> {
  const isXAIRegion = supportsXAI(region);
  const availableModels = POPULAR_MODELS.filter(m => {
    if (m.value.startsWith('xai.') && !isXAIRegion) {
      return false;
    }
    return true;
  });

  const modelChoice = await select({
    message: 'Select model:',
    choices: [
      ...availableModels.map(m => ({ value: m.value, name: m.name })),
      { value: 'custom', name: 'Enter custom model ID...' },
    ],
  });

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

    // Just verify we can create the client and region is valid
    const client = new genai.GenerativeAiClient({
      authenticationDetailsProvider: provider,
    });

    if (config.region) {
      client.region = common.Region.fromRegionId(config.region);
    }

    // Verify compartment exists by listing models (lightweight call)
    await client.listModels({
      compartmentId: config.compartmentId,
      limit: 1,
    });

    console.log(chalk.green('✓ Configuration valid'));
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

  // Generate opencode.json snippet
  const modelKey = config.servingMode === 'on-demand'
    ? config.modelId!
    : config.customModelName || 'custom-endpoint';

  const modelName = config.servingMode === 'on-demand'
    ? POPULAR_MODELS.find(m => m.value === config.modelId)?.name || config.modelId!
    : config.customModelName || 'Custom Endpoint';

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
