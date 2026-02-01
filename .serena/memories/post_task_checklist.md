# Post-Task Completion Checklist

When a task/feature is complete, follow these steps:

## 1. Code Quality
- [ ] Run `npm run build` - verify build succeeds without errors
- [ ] Run `npm run test` - all tests pass
- [ ] TypeScript compilation has no errors
- [ ] No `console.log()` or debugging code remains
- [ ] Code follows established naming conventions
- [ ] Type safety is maintained (no `any` types unless justified)

## 2. Code Review
- [ ] Code is self-documented with clear naming
- [ ] Logic is straightforward and maintainable
- [ ] Error handling is explicit and meaningful
- [ ] No commented-out code left behind

## 3. Testing
- [ ] Unit tests cover core functionality
- [ ] Edge cases are tested where applicable
- [ ] Tests use descriptive names
- [ ] For CLI features: manually test the wizard flow

## 4. Documentation
- [ ] README.md is updated if behavior changed
- [ ] Code comments added for non-obvious logic
- [ ] Types/interfaces are properly documented
- [ ] New commands/options are documented

## 5. Git Commit
- [ ] Changes are logically grouped
- [ ] Commit message is descriptive
- [ ] Only relevant files are committed (no dist/, node_modules/, etc.)

## 6. Before Creating PR
- [ ] Feature branch is up to date with main
- [ ] No merge conflicts
- [ ] All local tests pass
- [ ] Build produces clean output
