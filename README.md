<div align="center">

```
 ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗ ██████╗ ██████╗ ███████╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║   ██║██║  ██║█████╗
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║   ██║██║  ██║██╔══╝
╚██████╔╝██║     ███████╗██║ ╚████║╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
 ██████╗  ██████╗██╗    ██████╗ ██████╗  ██████╗ ██╗   ██╗██╗██████╗ ███████╗██████╗
██╔═══██╗██╔════╝██║    ██╔══██╗██╔══██╗██╔═══██╗██║   ██║██║██╔══██╗██╔════╝██╔══██╗
██║   ██║██║     ██║    ██████╔╝██████╔╝██║   ██║██║   ██║██║██║  ██║█████╗  ██████╔╝
██║   ██║██║     ██║    ██╔═══╝ ██╔══██╗██║   ██║╚██╗ ██╔╝██║██║  ██║██╔══╝  ██╔══██╗
╚██████╔╝╚██████╗██║    ██║     ██║  ██║╚██████╔╝ ╚████╔╝ ██║██████╔╝███████╗██║  ██║
 ╚═════╝  ╚═════╝╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═══╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

**OCI GenAI provider for OpenCode with interactive setup wizard**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCode Compatible](https://img.shields.io/badge/OpenCode-Compatible-blue)](https://opencode.ai)
[![Node 18+](https://img.shields.io/badge/Node-18%2B-brightgreen)](https://nodejs.org)
[![Community Project](https://img.shields.io/badge/Community-Maintained-success)](https://github.com/acedergren/opencode-oci-provider)

</div>

> **⚠️ Independent Community Project** — This project has **no official affiliation** with Oracle Corporation or the OpenCode team. It is a community-built integration that enables OpenCode to work with OCI GenAI services.

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
| xAI | `xai.grok-4.1-fast-non-reasoning` | **US regions only** |
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

## Legal

**Independent Project** — This is a community project with no affiliation to Oracle Corporation or the OpenCode team. "OCI" and "Oracle Cloud Infrastructure" refer to compatibility with Oracle's services, not endorsement by Oracle. "OpenCode" refers to compatibility with the OpenCode CLI tool.

**License** — MIT

**Disclaimer** — This software is provided "as is" without warranty. The authors and Oracle Corporation bear no liability for damages arising from its use. You are responsible for compliance with all applicable laws and Oracle's terms of service.

---

**Created by** [Alexander Cedergren](https://github.com/acedergren)
