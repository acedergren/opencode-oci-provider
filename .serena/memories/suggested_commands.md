# Suggested Commands for Development

## Build Commands
```bash
npm run build          # Build distribution (CJS + ESM + types)
```

## Development Commands
```bash
npm run dev            # Run CLI in development (useful for testing setup wizard)
```

## Testing
```bash
npm run test           # Run all tests with Vitest
npm run test -- --watch  # Run tests in watch mode
```

## Package Management
```bash
npm install            # Install dependencies
npm ci                 # Clean install (preferred for CI/CD)
```

## Publishing (automatic on npm publish)
```bash
npm run prepublishOnly # Automatically runs before publish
npm publish            # Publish to npm registry
```

## Useful Git Commands (Darwin system)
```bash
git status             # Check current branch and changes
git add <file>         # Stage specific file
git commit -m "msg"    # Create commit with message
git log --oneline -n10 # View recent commits
git diff               # View unstaged changes
git branch -a          # List all branches
```

## File System Commands (Darwin)
```bash
ls -la                 # List files with details
find . -name "*.ts"    # Find TypeScript files
grep -r "pattern" src/ # Search in source files
cat <file>             # View file contents
```

## Pre-Commit Checklist
1. Run `npm run build` to verify build succeeds
2. Run `npm run test` to verify tests pass
3. Verify no console.log or debug code remains
4. Check TypeScript has no errors
