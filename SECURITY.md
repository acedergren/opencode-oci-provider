# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

We support only the latest version. Please upgrade to the most recent release for security updates.

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, report security issues by:

1. **Email**: Create a private security advisory on GitHub
2. **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature

### What to include

- **Description**: Clear description of the vulnerability
- **Impact**: What can an attacker do?
- **Reproduction**: Step-by-step instructions
- **Affected versions**: Which versions are vulnerable?
- **Suggested fix**: If you have one

### Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity (critical: days, high: weeks, medium/low: best effort)

## Security Best Practices

When using this provider:

### OCI Credentials

- **Never commit** OCI credentials or config files to git
- **Use environment variables** or OCI config profiles
- **Restrict compartment permissions** to minimum required
- **Rotate credentials** regularly

### Environment Variables

Add to `.gitignore`:
```
.env*
*.local
~/.oci/config
```

### Compartment Security

- Use **dedicated compartments** for different environments
- Apply **IAM policies** with least-privilege access
- Enable **audit logging** for GenAI API calls
- Monitor **usage and costs** for anomalies

## Known Security Considerations

### API Keys

This provider uses OCI API keys for authentication. Ensure:
- Keys are stored securely in `~/.oci/config`
- File permissions are restrictive (`chmod 600 ~/.oci/config`)
- Keys are not logged or exposed in error messages

### Network Security

- OCI GenAI API calls use HTTPS
- Validate SSL certificates (do not disable verification)
- Be aware of network policies in your OCI tenancy

### Data Privacy

- GenAI API calls send prompts to Oracle's infrastructure
- Review Oracle's data handling policies
- Consider data residency requirements for your region
- Do not send sensitive/confidential data without proper review

## Disclosure Policy

- Vulnerabilities will be disclosed after a fix is available
- Credit will be given to reporters (unless they wish to remain anonymous)
- A security advisory will be published on GitHub

## Comments

This is a community project with no official Oracle affiliation. Security issues related to OCI services should be reported to Oracle through their official channels.
