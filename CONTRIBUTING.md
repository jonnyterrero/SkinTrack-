# Contributing to SkinTrack+

Thank you for your interest in contributing to SkinTrack+. All contributions — bug reports, feature suggestions, documentation improvements, and code changes — are welcome.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. **Install dependencies** (from the `frontend/` directory): `npm install` or `pnpm install`
3. **Run the dev server**: `npm run dev` or `pnpm dev`
4. Make your changes on a dedicated branch.

## Branching Convention

| Type | Branch format |
|------|---------------|
| New feature | `feature/short-description` |
| Bug fix | `fix/short-description` |
| Documentation | `docs/short-description` |
| Refactor | `refactor/short-description` |

## Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add melanoma progression chart
fix: correct lesion area calculation for cm² conversion
docs: document LAB color space analysis pipeline
refactor: extract body map component
```

## Pull Request Checklist

Before opening a PR, confirm:

- [ ] `npm run lint` passes with no errors
- [ ] `npm run build` completes successfully
- [ ] New features include a brief description in the PR body
- [ ] Image analysis changes are tested against sample dermatological images
- [ ] No patient or personally identifiable data is included in test assets
- [ ] Sensitive data (API keys, secrets) is not committed

## Reporting Bugs

Open a [GitHub Issue](https://github.com/jonnyterrero/SkinTrack-/issues) and include:
- Steps to reproduce
- Expected vs. actual behavior
- Browser/OS/Node version
- Relevant screenshots or console logs

## Code Style

- TypeScript strict mode is enabled — no implicit `any`
- Prefer named exports over default exports for components
- Keep components under `components/`, image analysis logic under `lib/`
- Tailwind utility classes only — no inline `style` props unless unavoidable
- Image analysis modules should be documented with inline comments explaining the algorithm (LAB color space, contour detection, etc.)

## Important Notes

- **This is not a diagnostic tool.** No contribution should imply clinical diagnosis capability.
- All healthcare-adjacent features must include appropriate disclaimers.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
