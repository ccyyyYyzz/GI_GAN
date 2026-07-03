## 8. Robustness and Reach

The locked claim of Section 7 fixes one operating point: the balanced dial $B$, at a single $5.0\%$ operator, on one $64\times64$ resolution, under noiseless acquisition. This section asks how far that operating point reaches — across sampling rate, across measurement noise, and across the image population. All results here are **development-level**: they reuse the frozen system, re-select nothing on the locked split, and constitute supplementary evidence rather than a second pre-registered claim. We label them as such throughout and treat none of them as certified. The measurement-consistency guarantee, however, is not development-level: because $A P_0 = 0$ holds for every operator by construction, $A\hat{x}_B = y$ remains exact at every rate and every noise level below.

### 8.1 Cross-rate generalization (development-level)

To test whether the balanced-fusion advantage is specific to the $5\%$ operator, we reuse the rate-agnostic priors unchanged and retrain only the lightweight anchor refiner at $2\%$ ($m=82$) and $10\%$ ($m=410$), then run the identical pipeline: select the global scalar $B$ on validation under the same tolerance rule, and score on the held-out development split (3 seeds per rate). This does not re-touch the frozen $5\%$ locked result, which appears only as the anchor point.

The advantage holds at every rate. Balanced fusion lowers LPIPS relative to the VQAE branch by $-0.116$ ($29.3\%$) at $2\%$, $-0.098$ ($32.6\%$) at $5\%$, and $-0.076$ ($34.2\%$) at $10\%$, with all $3/3$ seeds agreeing in direction at each rate and a PSNR cost of $-0.39$, $-0.45$, and $-0.43$ dB respectively — below the pre-registered $0.5$ dB validation tolerance throughout. The relative gain grows mildly with sampling rate: at higher rates more genuine detail is recoverable for the fusion to exploit, which is consistent with the fusion mechanism rather than an artifact of a single operator.

| Sampling rate | $\Delta$LPIPS (balanced $-$ VQAE) | Relative gain | $\Delta$PSNR (dB) | Seeds same-direction |
|---|---|---|---|---|
| $2\%$ ($m=82$) | $-0.116$ | $29.3\%$ | $-0.39$ | $3/3$ |
| $5\%$ ($m=205$, *locked*) | $-0.098$ | $32.6\%$ | $-0.45$ | $3/3$ |
| $10\%$ ($m=410$) | $-0.076$ | $34.2\%$ | $-0.43$ | $3/3$ |

### 8.2 Noise robustness (development-level)

The locked result is noiseless. To probe robustness we add i.i.d. Gaussian noise of standard deviation $\sigma$ to the bucket measurements and re-run the frozen system with the balanced $B$ unchanged (3 seeds, locked split). Two behaviors emerge. First, VQAE is the most noise-stable branch but always the least perceptual. Second, and more instructive, full VQGAN degrades sharply as noise grows — its fine synthesized detail amplifies the measurement noise — so that balanced fusion, which the noiseless ladder places *above* full VQGAN, **overtakes it at $\sigma = 0.02$** (LPIPS $0.197$ for balanced versus $0.204$ for full VQGAN) and beats it decisively at $\sigma = 0.05$ ($0.250$ versus $0.293$).

| Bucket noise $\sigma$ | VQAE | Balanced | Full VQGAN |
|---|---|---|---|
| $0.000$ | $0.300$ | $0.202$ | $0.172$ |
| $0.005$ | $0.299$ | $0.199$ | $0.172$ |
| $0.010$ | $0.297$ | $0.195$ | $0.176$ |
| $0.020$ | $0.295$ | $0.197$ | $0.204$ |
| $0.050$ | $0.304$ | $0.250$ | $0.293$ |

This is the behavior a controlled interior operating point should have: balanced fusion keeps most of VQGAN's perceptual benefit at low noise while degrading far more gracefully as the measurement becomes unreliable. The crossover is not evidence that noise certifies anything about the null space — the guarantee $A\hat{x}_B = y$ concerns only the row space at every $\sigma$ — but it does show that the recommended dial position is also the robust one.

### 8.3 Breadth of improvement and its failure mode

The gain is broad, not anecdotal. Balanced fusion improves LPIPS on $97.5\%$ of locked images for one seed and $99.2\%$ for another, worsening it on at most $13$ of $512$ images, with a worst-case regression of only $+0.07$ LPIPS and most regressions far smaller. The failures are not random: they concentrate on **man-made periodic and edge structure** — fences, vehicle body panels, airplane fuselages — where the natural-image VQGAN prior is a mismatch and synthesizes texture that conflicts with the regular geometry the scene actually contains. This is the expected and interpretable failure mode of a natural-image prior applied to structured man-made content, and it delimits where the dial should be advanced with care: the reach of responsible injection is bounded by the support of the prior, not by the measurement.
