# Release verification

The source release includes cached benchmark inputs, the compact sample estimator,
local analysis and collection, and a sample-only WSGI service.

Run checksum verification, Python tests, the feature benchmark, JavaScript syntax
checks and the service smoke script before publishing. Container execution must
be verified separately. Model downloads and private local databases are excluded.
