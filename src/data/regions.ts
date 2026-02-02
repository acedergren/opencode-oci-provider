/**
 * OCI GenAI region availability data
 *
 * xAI models (Grok) are only available in US regions.
 * Other providers (Cohere, Google, Meta) have broader availability.
 */

export type Provider = 'cohere' | 'google' | 'xai' | 'meta';

export interface RegionInfo {
  name: string;
  providers: Provider[];
}

/**
 * Region availability map
 * Source: https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm
 */
export const REGIONS: Record<string, RegionInfo> = {
  // US Regions (full xAI support)
  'us-chicago-1': {
    name: 'US Midwest (Chicago)',
    providers: ['cohere', 'google', 'xai', 'meta'],
  },
  'us-ashburn-1': {
    name: 'US East (Ashburn)',
    providers: ['cohere', 'google', 'xai', 'meta'],
  },
  'us-phoenix-1': {
    name: 'US West (Phoenix)',
    providers: ['cohere', 'google', 'xai', 'meta'],
  },
  'us-sanjose-1': {
    name: 'US West (San Jose)',
    providers: ['cohere', 'google', 'xai', 'meta'],
  },

  // EU Regions
  'eu-frankfurt-1': {
    name: 'Germany Central (Frankfurt)',
    providers: ['cohere', 'google', 'meta'],
  },
  'uk-london-1': {
    name: 'UK South (London)',
    providers: ['cohere', 'google', 'meta'],
  },
  'eu-amsterdam-1': {
    name: 'Netherlands Northwest (Amsterdam)',
    providers: ['cohere', 'google', 'meta'],
  },

  // Asia Pacific Regions
  'ap-osaka-1': {
    name: 'Japan Central (Osaka)',
    providers: ['cohere', 'google', 'meta'],
  },
  'ap-tokyo-1': {
    name: 'Japan East (Tokyo)',
    providers: ['cohere', 'google', 'meta'],
  },
  'ap-sydney-1': {
    name: 'Australia East (Sydney)',
    providers: ['cohere', 'google', 'meta'],
  },
  'ap-melbourne-1': {
    name: 'Australia Southeast (Melbourne)',
    providers: ['cohere', 'google', 'meta'],
  },
  'ap-singapore-1': {
    name: 'Singapore',
    providers: ['cohere', 'google', 'meta'],
  },
  'ap-hyderabad-1': {
    name: 'India South (Hyderabad)',
    providers: ['cohere', 'google', 'meta'],
  },
  'ap-mumbai-1': {
    name: 'India West (Mumbai)',
    providers: ['cohere', 'google', 'meta'],
  },
  'ap-seoul-1': {
    name: 'South Korea Central (Seoul)',
    providers: ['cohere', 'google', 'meta'],
  },

  // South America
  'sa-saopaulo-1': {
    name: 'Brazil East (Sao Paulo)',
    providers: ['cohere', 'google', 'meta'],
  },
  'sa-santiago-1': {
    name: 'Chile (Santiago)',
    providers: ['cohere', 'meta'],
  },

  // Middle East
  'me-dubai-1': {
    name: 'UAE East (Dubai)',
    providers: ['cohere', 'meta'],
  },
  'me-jeddah-1': {
    name: 'Saudi Arabia West (Jeddah)',
    providers: ['cohere', 'meta'],
  },

  // Canada
  'ca-toronto-1': {
    name: 'Canada Southeast (Toronto)',
    providers: ['cohere', 'google', 'meta'],
  },
  'ca-montreal-1': {
    name: 'Canada Southeast (Montreal)',
    providers: ['cohere', 'google', 'meta'],
  },
};

/**
 * Models that require dedicated AI clusters (not available on-demand)
 * Source: https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm
 * Last updated: 2026-02-02
 */
export const DEDICATED_ONLY_MODELS = [
  // Meta Llama 4 models (dedicated only)
  'meta.llama-4-maverick',
  'meta.llama-4-scout',
  // Meta Llama 3.2 11B Vision (dedicated only)
  'meta.llama-3.2-11b-vision',
];

/**
 * Get regions that support a specific provider
 */
export function getRegionsForProvider(provider: Provider): string[] {
  return Object.entries(REGIONS)
    .filter(([_, info]) => info.providers.includes(provider))
    .map(([regionId]) => regionId);
}

/**
 * Check if a region supports xAI models
 */
export function supportsXAI(regionId: string): boolean {
  const region = REGIONS[regionId];
  return region?.providers.includes('xai') ?? false;
}

/**
 * Format region choice for display in prompts
 */
export function formatRegionChoice(regionId: string): string {
  const info = REGIONS[regionId];
  if (!info) return regionId;

  const providerHints = info.providers.includes('xai')
    ? `[xAI, Google, Cohere, Meta]`
    : `[${info.providers.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(', ')}]`;

  return `${regionId} - ${info.name} ${providerHints}`;
}

/**
 * Get all region IDs sorted by name
 */
export function getAllRegionIds(): string[] {
  return Object.keys(REGIONS).sort((a, b) => {
    const aInfo = REGIONS[a];
    const bInfo = REGIONS[b];
    return aInfo.name.localeCompare(bInfo.name);
  });
}

/**
 * Check if a model requires a dedicated AI cluster
 */
export function isDedicatedOnly(modelId: string): boolean {
  return DEDICATED_ONLY_MODELS.includes(modelId);
}

/**
 * Get user-friendly model name for error messages
 */
export function getModelDisplayName(modelId: string): string {
  const parts = modelId.split('.');
  if (parts.length < 2) return modelId;

  const provider = parts[0];
  const model = parts.slice(1).join('.');

  // Format provider name
  const providerNames: Record<string, string> = {
    'meta': 'Meta',
    'cohere': 'Cohere',
    'google': 'Google',
    'xai': 'xAI',
  };

  const providerName = providerNames[provider] || provider;

  // Format model name
  const modelName = model
    .split('-')
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

  return `${providerName} ${modelName}`;
}
