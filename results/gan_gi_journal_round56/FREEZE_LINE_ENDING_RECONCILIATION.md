# Freeze-manifest line-ending reconciliation

The held-out method, artifacts, test-set policy, and decision rule are unchanged.
This record reconciles two raw SHA-256 values for the same parsed freeze
manifest caused solely by Git checkout newline conversion:

- Windows working-tree bytes (CRLF), used by the local aggregator:
  `1504562f3a5144e7e1d38d684849d5e3b5ccb8cb119ab366bd19965076f8f033`.
- Git blob and Colab checkout bytes (LF), used by all three held-out lanes:
  `c52c0c8f44f02ee2b4c2fc2a37e51dd0988799ab0e15d2d46948c12e31286bca`.
- Canonical parsed-JSON SHA-256 (sorted keys, compact UTF-8):
  `3cef6b6c0daff1dd87d40056ac26689449b4d07ae1072bd0c1024decfd769c4c`.

The decoded files have identical line sequences and parse to equal JSON
objects.  The LF release copy is byte-identical to the blob at commit
`ef149605b797c9f3241a005e0a7fe22f2e970599`.  All six released metric-vector,
summary, cache-manifest, and completion-receipt hashes match the held-out
decision.  Re-aggregation from the three extracted releases reproduces every
scientific value, interval, lane sign, projection certificate, and decision
gate.  No method parameter, checkpoint, operator, image, metric, or quality
result is changed by this reconciliation.
