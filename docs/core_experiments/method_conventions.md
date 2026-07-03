# Method Convention Audit

- RelMeasErr: canonical paper convention is unclipped float64 against recorded `y`.
- PSNR/SSIM: clipped/display metrics with data range 1.0.
- `x_data`: write as `D(y)`, the deterministic pipeline anchor. Do not universally define it as `B_lambda y`.
- Rad-5/Rad-10: ridge-pinv anchor numerically matches `B_lambda y`; exported exact A tensors are canonical; entries are `+/-1/sqrt(m)`.
- Scr-5/Scr-10: deterministic Hadamard zero-filled anchor; rows are orthonormal, DC row included; finite-lambda `B_lambda y=A^T y/(1+lambda)` is not the universal `x_data`.
- `lambda_solver`: 0.001 in the audited pipeline.
- Hard audit/POCS: use for boundary/feasibility experiments; noisy-data reporting should acknowledge Morozov/discrepancy calibration.
- GAN certificate: the GAN is not the certificate. `Pi_y^lambda` is the measurement audit/certificate path.
