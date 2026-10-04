# Release checks

The Render website runs the CPU ONNX encoder and regression head. It accepts photo uploads and serves four sample pairs.

The release checks cover:

- 95 Python tests for model selection, image validation, request limits, concurrency and storage.
- Checksums for 16 bundled data/model assets.
- Ten benchmark episodes matching reference predictions within `1e-10`.
- JavaScript syntax and Gunicorn configuration.
- A Docker build and photo-upload smoke test with 512 MB of memory and 0.1 CPU.
- Encoder-export parity on all four samples, with remaining-fraction differences below `0.0001`.
- A live browser upload of L492 returning the expected rounded 29% result, retained while switching between Analyze, Accuracy and About.

[Run these checks](DEVELOPMENT.md). CI repeats the automated checks on pushes and pull requests.

`RELEASE-MANIFEST.json` hashes the packaged source. The source ZIP excludes environments, encoder weights, caches, databases and uploads. [Deployment](DEPLOY.md) · [Data licenses](DATA.md).
