# Publishing the documentation

The site is built and deployed by `.github/workflows/docs.yaml`. On every pull
request and push to `main` it only builds the site (and fails on any warning).
It deploys to GitHub Pages only when a release is published, or when someone
runs the workflow by hand. These are the steps to publish it the first time,
and afterwards.

## One-time setup

1. **Merge the pull request that adds the site into `main`.** GitHub offers the
   "Run workflow" button only for workflows on the default branch, so the docs
   workflow must be on `main` first.
2. **Switch the Pages source to GitHub Actions.**
   1. Open the repository on GitHub and click **Settings** (under the **…**
      menu if it is hidden).
   2. In the sidebar, under "Code and automation", click **Pages**.
   3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

   GitHub then creates the deployment environment `github-pages` if it does not
   exist. The old `gh-pages` branch is no longer used once this is set; the old
   template pages stop being served.
3. **Allow release tags to deploy (only if needed).** If a deploy started by a
   release later fails with a message like
   `Tag "v1.4.0" is not allowed to deploy to github-pages due to environment protection rules`,
   open **Settings → Environments → github-pages → Deployment branches and
   tags**, click **Add deployment branch or tag rule**, choose **Tag**, enter
   `v*`, and save. Deploys started by hand from `main` do not need this rule.

## Publish the site without making a release

Use this for the first publication, and whenever the site should change between
releases.

1. Open the **Actions** tab of the repository.
2. In the list of workflows, click **docs**.
3. Click **Run workflow**, keep the branch `main`, and click the green
   **Run workflow** button.
4. Wait for both jobs, **build** and **deploy**, to finish with a green check
   (a few minutes; the build installs PyTorch).
5. Open <https://khegazy.github.io/padaquad/>. If the old template still
   shows, reload without the cache (Shift + reload).

## Publish the site with a release

Publishing a GitHub release deploys the site automatically. **It also uploads
the package to PyPI**: `.github/workflows/release.yaml` runs on every new
release. Only take this route when you mean to release a new version.

1. Make sure `version` in `pyproject.toml` and the CHANGELOG heading name the
   new version, and that they are merged into `main`.
2. Open **Releases → Draft a new release**, create a tag such as `v1.4.0` on
   `main`, write the notes, and click **Publish release**.
3. Check the **Actions** tab: the **docs** workflow (build and deploy) and the
   PyPI workflow both run.

## If the build fails

The **build** job runs the same commands as a local build:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

Run them locally to see the warning. A warning fails the build on purpose; fix
the page or docstring it names. A link-check failure from a publisher that
blocks automated requests (it answers 403) can be excluded in `docs/conf.py`
(`linkcheck_ignore`), after checking the link by hand.
