# Contributing

Thanks for your interest. universal-bot is in **Phase 0** (scaffolding) — we aren't ready for code contributions yet, but design discussion and issues are welcome.

## Quick links
- [Phase 0 checklist](docs/phase-0-checklist.md)
- [Roadmap](README.md#roadmap)
- [Open an issue](../../issues/new/choose)

## Sign-off (DCO)

Every commit must be signed off:

```
git commit -s -m "your message"
```

This adds a `Signed-off-by:` trailer certifying you wrote the change or have permission to contribute it, per the [Developer Certificate of Origin](https://developercertificate.org/). A CI check enforces this on every PR.

If you forget to sign off, amend with:

```
git commit --amend -s --no-edit
```

## Plugin authoring

The plugin SDK lands in **Phase 2**. You'll be able to add:

- A **connector** (data source — URL, DB, SaaS app, etc.)
- An **indexer** (retrieval strategy)
- A **provider** (LLM, embedding, or reranker)

...by dropping a single Python file in `app/connectors/`, `app/indexers/`, or `app/providers/`. Authoring docs will live here when ready.

## License

By contributing you agree your contributions are licensed under the project's [Apache-2.0 license](LICENSE). Do not contribute code under a copyleft license (GPL / AGPL / LGPL / SSPL).
