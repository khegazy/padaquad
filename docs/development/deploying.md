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
   exist. The `gh-pages` branch is no longer used for new deploys, but the site
   it last published may stay online until the first deploy from the workflow
   replaces it.
3. **Allow release tags to deploy.** A deploy started by a release runs on the
   release's tag (for example `v1.4.0`), not on `main`, and the `github-pages`
   environment may allow only the default branch to deploy. GitHub's
   documentation does not say what it allows by default, and the rule is
   harmless if it is not needed, so add it now: open **Settings → Environments →
   github-pages**, under **Deployment branches and tags** click **Add deployment
   branch or tag rule**, choose **Tag**, enter `v*`, and save. Without it, the
   deploy of a release fails with a message like
   `Tag "v1.4.0" is not allowed to deploy to github-pages due to environment protection rules`.
   Deploys started by hand from `main` do not need this rule.

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
the package to PyPI**: `.github/workflows/release.yaml` runs on every published
release. Only take this route when you mean to release a new version. Both
workflows run when the release is published, not when a draft is saved, so a
draft release starts neither.

1. Make sure the new version and its release date are in all three places, and
   that they are merged into `main`: `version` in `pyproject.toml`, the heading
   `## [<version>] - <date>` in `CHANGELOG.md`, and `version` and
   `date-released` in `CITATION.cff`. A test checks that the three agree.
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
