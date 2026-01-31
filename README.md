# opencode-oci-provider

OCI GenAI provider for [OpenCode](https://opencode.ai) with interactive setup wizard.

## Quick Start

### 1. Run Setup Wizard

```bash
npx opencode-oci-provider
# or after installation:
opencode-oci-setup
```

The wizard will:
- Detect your OCI config profiles
- Let you select a region (with xAI availability indicators)
- Configure compartment ID
- Choose between On-Demand or Dedicated AI Cluster
- Select a model or endpoint
- Test the configuration
- Generate `opencode.json` and `.env.oci-genai` files

### 2. Use with OpenCode

After setup, your `opencode.json` will be configured. Start OpenCode:

```bash
opencode
```

## Manual Configuration

If you prefer manual setup:

### Environment Variables

```bash
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxxxx
export OCI_CONFIG_PROFILE=DEFAULT
# For dedicated mode:
# export OCI_GENAI_ENDPOINT_ID=ocid1.generativeaiendpoint.oc1..xxxxx
```

### opencode.json

```json
{
  "provider": {
    "oci": {
      "npm": "opencode-oci-provider",
      "name": "Oracle Cloud Infrastructure",
      "options": {
        "region": "us-chicago-1",
        "compartmentId": "${OCI_COMPARTMENT_ID}"
      },
      "models": {
        "cohere.command-r-plus-08-2024": {
          "name": "Cohere Command R+",
          "type": "chat"
        }
      }
    }
  }
}
```

## Available Models

### On-Demand Models

| Provider | Model ID | Notes |
|----------|----------|-------|
| Cohere | `cohere.command-r-plus-08-2024` | Best quality |
| Cohere | `cohere.command-r-08-2024` | Balanced |
| Google | `google.gemini-2.0-flash-001` | Fast |
| Google | `google.gemini-1.5-pro-002` | Advanced |
| xAI | `xai.grok-2-1212` | **US regions only** |
| Meta | `meta.llama-3.1-405b-instruct` | Large |
| Meta | `meta.llama-3.1-70b-instruct` | Medium |

### Region Availability

- **US regions** (us-chicago-1, us-ashburn-1, us-phoenix-1, us-sanjose-1): All providers including xAI
- **Other regions**: Cohere, Google, Meta (no xAI)

## Dedicated AI Clusters

For production workloads, you can use Dedicated AI Clusters:

1. Create a cluster in the OCI Console
2. Create an endpoint on the cluster
3. Run `opencode-oci-setup` and select "Dedicated AI Cluster"
4. Choose your cluster and endpoint

## Prerequisites

- [OCI CLI configured](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm) (`~/.oci/config`)
- Node.js 18+
- OCI compartment with GenAI access

## Troubleshooting

### "Missing compartment ID"

Set `OCI_COMPARTMENT_ID` environment variable or pass it in options.

### "Authentication failed"

Verify your OCI CLI is configured: `oci iam region list`

### "Model not available in region"

xAI models are only available in US regions. Use Cohere, Google, or Meta models in other regions.

## License

MIT
