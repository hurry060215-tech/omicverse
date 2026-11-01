# scran numerical portability follow-up — diagnostic only

The production adapter, four reviewed wheels, strict `atol=1e-8, rtol=0` gate,
and default algorithms remain unchanged. Ordinary PyPI is still **113/116**;
the exact qualified Linux numerical profile remains **116/116**.

Tracing the retained-QR nonlinear fit found the first execution divergence at
zero-based `ddot` call32 with byte-identical inputs. The ordinary result was
0.2790912220520218 and the qualified result0.27909122205202186: one ULP. Subsequent
finite-difference steps amplify this difference. Changing OpenBLAS CPU dispatch
alone did not remove the difference. Source/binary inspection identified the
qualified short-vector grouping/FMA rounding profile.

A separately isolated, unaccepted compatibility prototype reproduced that
profile for finite contiguous float64 vectors of lengths1–15. It restores116/116
on both tested environments without changing thresholds or frozen references.
This is NOT evidence of a generally more accurate optimizer: ordinary sequential
summation passed111/116, while accurate `fsum` passed105/116. The gate is sensitive
to the reference execution trajectory, not just abstract real arithmetic.

Independent checks covered15,000 random short dots, another15,000 finite edge
cases, and11,728 FMA comparisons. No production global BLAS monkeypatch, new wheel,
or installation was made. Long vectors, other BLAS operations, hardware/compiler
profiles, fresh R oracle cases and biological validation remain outside this
bounded diagnosis. The candidate must not be presented as a portable production
fix. The supported strict-replay path is still the documented locked Linux stack.

Official source grounding:
https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/v0.3.34/kernel/x86_64/ddot.c
Compiler FMA contraction is additionally established from the qualified binary,
not dictated by source alone. Detailed result identities and hashes are in
`roadmap989-scran-portability-diagnostic.json`; those traces are research
evidence, not part of the runtime API.
