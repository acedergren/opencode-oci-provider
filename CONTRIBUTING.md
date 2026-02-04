# Contributing to opencode-oci-provider

Thank you for your interest in contributing! This is a community project, and contributions are welcome.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** with clear commit messages
5. **Test thoroughly** before submitting
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/opencode-oci-provider.git
cd opencode-oci-provider

# Install dependencies
npm install

# Run development mode
npm run dev

# Run tests
npm run test

# Build distribution
npm run build
```

## Code Standards

- **Language**: TypeScript (ES2020, strict mode)
- **Runtime**: Node.js 18+
- **Style**: Follow existing code conventions
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/)

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Examples**:
```
feat(cli): add region auto-detection
fix(provider): handle missing compartment ID
docs: update installation instructions
```

## Before Submitting

- [ ] Code builds successfully (`npm run build`)
- [ ] Tests pass (`npm run test`)
- [ ] No linting errors
- [ ] Commit messages follow conventions
- [ ] Changes are documented in commit body

## Pull Request Process

1. **Update documentation** if adding features
2. **Add tests** for new functionality
3. **Ensure CI passes** (all checks green)
4. **Request review** from maintainers
5. **Address feedback** promptly

## Reporting Issues

### Bug Reports

Include:
- **Description**: What happened vs. what you expected
- **Steps to reproduce**: Minimal example
- **Environment**: Node version, OS, OCI region
- **Logs**: Relevant error messages (redact sensitive info)

### Feature Requests

Include:
- **Problem**: What problem does this solve?
- **Proposed solution**: How would it work?
- **Alternatives**: What else have you considered?
- **Impact**: Who would benefit?

## Code Review Guidelines

**For contributors:**
- Keep PRs focused (one feature/fix per PR)
- Respond to feedback constructively
- Be patient with review process

**For reviewers:**
- Be respectful and constructive
- Focus on code quality and correctness
- Suggest improvements, don't demand perfection

## Questions?

- **Documentation**: See [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/acedergren/opencode-oci-provider/issues)
- **Discussions**: [GitHub Discussions](https://github.com/acedergren/opencode-oci-provider/discussions)

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

## Community Standards

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards.
