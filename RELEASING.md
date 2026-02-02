# Releasing

## Pre-release checklist

1. Bump `version` in `pyproject.toml`.
2. Update `version` and `date-released` in `CITATION.cff`.
3. Commit the version bump, then create an annotated tag:

```sh
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

## Build and verify

```sh
make build
```

This runs `uv build` to produce a wheel and sdist in `dist/`, then smoke-tests
both artifacts in isolated environments.

## Test on TestPyPI (optional but recommended)

Upload to TestPyPI first to catch packaging issues before the real release:

```sh
make publish-test
```

Then verify the upload:

```sh
uv pip install --index-url https://test.pypi.org/simple/ mjlab
```

Note: TestPyPI requires a separate account and API token from the real PyPI.
Set the token via `UV_PUBLISH_TOKEN` or pass `--token`.

## Publish to PyPI

```sh
make publish
```

Set the token via `UV_PUBLISH_TOKEN` or pass `--token`.

## Post-release

Verify the release installs correctly:

```sh
uv pip install mjlab==X.Y.Z
```

## Releasing from a past tag

If the tag has already been created and HEAD has moved ahead, check out the
tag before building:

```sh
git checkout vX.Y.Z
make build
make publish
git checkout main
```
