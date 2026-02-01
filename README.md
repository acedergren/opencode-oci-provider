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

For the latest list of available models and their regional availability, see the [Oracle Cloud Infrastructure Generative AI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm).

**Note:** The setup wizard will automatically show you only the models available in your selected region.

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

Check the [OCI GenAI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm) for model availability by region.

## Legal

**Independent Project** — This is a community project with no affiliation to Oracle Corporation or the OpenCode team. "OCI" and "Oracle Cloud Infrastructure" refer to compatibility with Oracle's services, not endorsement by Oracle. "OpenCode" refers to compatibility with the OpenCode CLI tool.

**License** — MIT

**Disclaimer** — This software is provided "as is" without warranty. The authors and Oracle Corporation bear no liability for damages arising from its use. You are responsible for compliance with all applicable laws and Oracle's terms of service.

---

**Created by** [Alexander Cedergren](https://github.com/acedergren)
