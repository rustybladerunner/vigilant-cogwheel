# Public training experiment

Read the README's validation limits. Run `python -m pytest -q` and the documented
validation-only CLI before publishing code changes. CPU contract tests do not
prove GPU training or model quality. Do not silently drop messages or represent
an unimplemented detector as a successful safety check.

Write plainly and distinguish measurements from hypotheses. Before publishing,
review exact files, dataset examples, caches, model artifacts, commit metadata,
links, and reachable history for personal data and secrets. Keep private training
data and local logs out of the repo. Use a GitHub noreply commit address. Review
pattern matches in context; a scan alone cannot prove privacy.
