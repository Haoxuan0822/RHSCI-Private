# Sources

RHSCI builds on [PSI0](https://github.com/physical-superintelligence-lab/Psi0). Native packages retain their original names, copyright notices and component licenses. RHSCI integrations live in `src/rsci/backbones` and include changes to the native model and training interfaces where needed.

ACT, Diffusion Policy, InternVLA-M1, H-RDT, DreamZero and the EgoDex/EgoMimic data helpers were imported from PSI0 commit [`cda3fdfafe47cb6fa75400d013de1bd36e3cea05`](https://github.com/physical-superintelligence-lab/Psi0/tree/cda3fdfafe47cb6fa75400d013de1bd36e3cea05). Their native source is included. ACT additionally has an RHSCI adapter under `src/rsci/backbones/act`.

Being-H0.5 in `src/beingh05` is the `Being-H05` directory of [BeingBeyond/Being-H](https://github.com/BeingBeyond/Being-H/tree/27ec67b84ac6c8e55f95408205e1ad7ba8bec253/Being-H05), revision `27ec67b84ac6c8e55f95408205e1ad7ba8bec253`. Official source, docs and license are retained; `configs/rng_state.pkl` is omitted as runtime state. Its launcher supports native training only. Diffusion Policy, InternVLA-M1, H-RDT, DreamZero and Being-H0.5 have no RHSCI adapters.

The λ₀ source in `src/openlambda` is based on upstream commit `858b3837d42946ed90291d8da7ce6ce720335cac`. [The integration patch](../baselines/lambda0/upstream.patch) records changes to its native interfaces; [the component manifest](../src/openlambda/SOURCE_MANIFEST.json) identifies the vendored files.

GR00T N1.6, EgoVLA/VILA and DreamZero were also imported through the same PSI0 revision. Their `SOURCE_MANIFEST.json` files identify the immediate source, license reference revisions and RHSCI modifications. The original checkout revisions before PSI0's modifications were not recorded by PSI0 and are not invented here. Historical component license texts have been restored separately from that source provenance: [GR00T N1.6](../src/gr00t/LICENSE), [VILA](../src/egovla/VILA/LICENSE) and [DreamZero](../src/dreamzero/LICENSE). GR00T N1.6's non-commercial research restriction also applies to derivatives; a later upstream Apache license is not substituted.

The root [LICENSE](../LICENSE) is retained from PSI0; the [full Apache 2.0 text](../licenses/Apache-2.0.txt) and [third-party notices](../THIRD_PARTY_NOTICES.md) describe component scope. Component license and notice files apply to their respective source trees. Upstream documentation describes the native implementations; use [Training](TRAINING.md) and the per-backbone guides for the RHSCI entries.
