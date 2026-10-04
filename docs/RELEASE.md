# Aftermeal 1.0 release verification

This is a locally verified source release and public-service preview. Public hosting, real-user adoption and new model accuracy are not claimed.

## Scope

- Three dish-named gallery samples, before/after photos, optional blue highlights and rounded remaining-fraction estimates.
- Recorded fractions and prediction errors stay visible. The fourth, difficult rice pair is accessible from Accuracy's optional failure analysis.
- Technical explanations are collapsed or in Project notes. No review form appears on the fixed demo samples.
- Upload analysis, the 524-record review lab and weighed capture remain separate local capabilities.
- A sample-only WSGI service, pinned Gunicorn dependency, Docker configuration, CI workflow, source packaging and deployment instructions.

## Verification performed

A fresh Python 3.12 virtual environment installed only `requirements-public.txt`: NumPy 2.3.5, Pillow 12.3.0, threadpoolctl 3.7.0, Gunicorn 26.2.0. No encoder installation or GPU was needed.

- **86 Python tests passed**, including seven new public-service tests. The tests exercise WSGI response validity, bounded bodies, upload/private-route rejection, simultaneous sample requests, request validation and security headers, alongside the existing model, split, image and transactional-storage checks.
- All **16 original input/model assets** passed their unchanged checksum manifest.
- The feature benchmark refit every candidate and reproduced all ten outer episodes with its existing 1e-10 reference tolerance. LeFood Change MAE remains **9.6330 pp**; ACETADA **9.4585 pp**. Accuracy has not changed in this release.
- Gunicorn configuration validated. The actual two-worker service passed black-box HTTP checks for health, all four samples, project notes, private-route isolation and request limits.
- JavaScript syntax checked. Browser verification covered the shortened gallery, highlight on/off, normal sample switching, Accuracy → failure pair navigation, absence of sample-review controls and narrow-screen layout without horizontal overflow.

The Docker CLI is installed but its daemon was not running. **Docker build and container execution have not been verified locally.** The CI workflow includes both. The GitHub workflow verifies each pushed release; its status is available in the repository Actions tab. Host-level HTTPS, traffic limits and actual hosted behavior remain publication checks, not local test results.

## Data handling

Model parameters, source weights and benchmark folds are unchanged. No real capture records were added. Development-only browser review tests were run on the separate port-8080 preview before that unnecessary feature was removed; they were marked synthetic, were not submitted anywhere, and are not included in the release.

The source ZIP excludes local databases, environment directories, downloaded model weights, extraction caches, and unrelated workspace projects. `RELEASE-MANIFEST.json` identifies all packaged content and explicitly records redaction of an absolute workbook path from the packaged source-audit report. Original evidence files remain intact in the workspace.

## 1.0.1 upload simplification

Removed the manual starting-portion dropdown and its blocking missing/uncertain-observation gate. Valid photo pairs now proceed directly to experimental inference. The API preserves absent observations as `not_provided`, never as user-confirmed food presence. Duplicate, constant-image, size and known source-flag checks remain. Shortened upload instructions and removed redundant success/help messages. All 87 Python tests pass after this change. The earlier 1.0.0 ZIP is retained as a historical build.

Browser verification of 1.0.1 uploaded the original L492 before/after JPEGs through the file inputs, selected Analyze pair without any observation field, and received a 29% rounded estimate plus both live highlights.  This verifies the flow, not new model accuracy.

## 1.0.2 navigation cleanup

The product uses one persistent Analyze / Accuracy / About shell. Removed research-lab, capture and setup links from product navigation. The difficult example has its own photos and result inside an Accuracy disclosure. Switching views preserves the analysis state; switching to sample pairs and back also retains uploaded photos and their completed estimate. Old `/case-study` URLs enter the new About view. Developer tools remain available by direct local URL and in repository documentation.

Navigation verification: an actual uploaded L492 pair retained identical displayed image hashes and its 29% result after visiting Accuracy (with the difficult example expanded), About, and Analyze, and after switching to sample mode and back. Verified old notes URLs, browser Back, keyboard Home navigation and a 390-pixel viewport with no horizontal overflow in Analyze or Accuracy. All 87 Python tests and the JavaScript syntax check pass. Reloading the document still clears in-memory uploads; these checks concern in-app navigation.

## Repository checks

The standalone repository runs 89 Python checks, including missing/modified input detection and EXIF orientation. Bundled model/data verification covers the original 16 assets. The ten benchmark episodes reproduce their saved forecasts within 1e-10.
