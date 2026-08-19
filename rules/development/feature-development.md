# Feature development

1. Search the relevant rules and inspect the existing contract before writing code.
2. Define the behavior and write a failing focused test before implementation.
3. Implement the smallest change that satisfies the test; refactor only with green tests.
4. Update schemas, Skill references, docs, and release checks whenever a portable contract moves.
5. Run the applicable canonical checks and report exact results.

Keep v0.1 source-read-only and offline. Do not turn an unavailable workflow into a hidden
fallback or a provider request.
