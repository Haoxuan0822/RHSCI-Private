# Sources

RHSCI builds on [PSI0](https://github.com/physical-superintelligence-lab/Psi0). Native packages retain their original names, copyright notices and component licenses. RHSCI integrations live inside their native packages (`psi.rsci`, `openpi.rsci`, `gr00t.rsci`, `human_plan.rsci`, `lambda0.rsci`, `act.rsci`); shared tools live in `src/rsci`. The [2026-10-31 layout record](NATIVE_PACKAGE_LAYOUT_20261031.md) records the original relocation; the [current flat package layout](DIRECTORY_STRUCTURE.md#package-depth-2026-11-01) and component manifests track later filesystem moves without changing upstream provenance.

ACT, Diffusion Policy, InternVLA-M1, H-RDT, DreamZero and the EgoDex/EgoMimic data helpers were imported from PSI0 commit [`cda3fdfafe47cb6fa75400d013de1bd36e3cea05`](https://github.com/physical-superintelligence-lab/Psi0/tree/cda3fdfafe47cb6fa75400d013de1bd36e3cea05). Their native source is included. ACT additionally has an RHSCI adapter under `src/act/rsci`.

Being-H0.5 in `src/beingh05` is the `Being-H05` directory of [BeingBeyond/Being-H](https://github.com/BeingBeyond/Being-H/tree/27ec67b84ac6c8e55f95408205e1ad7ba8bec253/Being-H05), revision `27ec67b84ac6c8e55f95408205e1ad7ba8bec253`. Official source, docs and license are retained; `configs/rng_state.pkl` is omitted as runtime state. Its launcher supports native training only. Diffusion Policy, InternVLA-M1, H-RDT, DreamZero and Being-H0.5 have no RHSCI adapters.

The λ₀ model package in `src/lambda0` and project resources in `src/openlambda` is based on upstream commit `858b3837d42946ed90291d8da7ce6ce720335cac`. [The integration patch](../baselines/lambda0/upstream.patch) records changes to its native interfaces; [the component manifest](../src/openlambda/SOURCE_MANIFEST.json) identifies the vendored files.

GR00T N1.6, EgoVLA/VILA and DreamZero were also imported through the same PSI0 revision. Their `SOURCE_MANIFEST.json` files identify the immediate source, license reference revisions and RHSCI modifications. The original checkout revisions before PSI0's modifications were not recorded by PSI0 and are not invented here. Historical component license texts have been restored separately from that source provenance: [GR00T N1.6](../src/gr00t/LICENSE), [VILA](../src/egovla/VILA/LICENSE) and [DreamZero](../src/dreamzero/LICENSE). GR00T N1.6's non-commercial research restriction also applies to derivatives; a later upstream Apache license is not substituted.

The root [LICENSE](../LICENSE) is retained from PSI0; the [full Apache 2.0 text](../licenses/Apache-2.0.txt) and [third-party notices](../THIRD_PARTY_NOTICES.md) describe component scope. Component license and notice files apply to their respective source trees. Upstream documentation describes the native implementations; use [Training](TRAINING.md) and the per-backbone guides for the RHSCI entries.


## 2026-10-30 PSI0-RAW integration

Latest input-cache and training-recovery changes are documented in the [integration record](PSI0_RAW_INTEGRATION_20261030.md). It lists source paths, compatibility limits and CPU validation. Existing inference entry points and backbone recipes remain documented above.

The subsequent local [training structure refactor](TRAINING_STRUCTURE_20261030.md) uses integration commit `7c39b9ac9b0285f64596b0ea00e96210dd2009ba` as its base. The integration JSON retains historical target hashes for that commit; the export manifest describes current files.

## Official SIMPLE (2026-10-30)

External SIMPLE now uses official commit `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`, including PR #29. Its MIT license and component notices remain in that checkout. The [source lock](../configs/simple/official.json) records file SHA256 and official submodules; see the [migration and validation record](SIMPLE_OFFICIAL_20261030.md). Runtime assets and environments are not bundled or certified by the source check.
