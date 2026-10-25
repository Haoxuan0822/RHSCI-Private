# Third-party notices

The root `LICENSE` is the Apache 2.0 notice retained from PSI0. The complete Apache 2.0 terms are in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). Component terms below apply to their respective code and derivatives; the root license does not replace them. Model weights, tokenizers and datasets are not distributed here and have separate provider terms.

| Component and scope | Source and applicable terms |
| --- | --- |
| `src/gr00t`, including the N1.6 implementation and modifications derived from it | NVIDIA Isaac-GR00T N1.6; [NVIDIA License](src/gr00t/LICENSE), copied from `ead52833afbbf4243f8cd5e7664f48a94de03b19` (`n1.6-release` snapshot). Section 3.3 restricts use of the work and derivatives to non-commercial research and excludes military, surveillance, nuclear-technology service and biometric-processing purposes. Section 3.1 requires retention of the complete license and attribution notices. This is not the current main branch's Apache license. |
| `src/rsci/backbones/gr00t` | RHSCI integration with the N1.6 implementation. The N1.6 terms continue to apply to derived code; the root Apache notice grants no broader rights to that code. |
| `src/egovla/VILA` | VILA / NVIDIA CORPORATION & AFFILIATES and other authors identified in file headers; [Apache 2.0](src/egovla/VILA/LICENSE), copied from the historical NVILA snapshot `36f6adcd11a10be1580caeb7e647e1b6f8517f89`. VILA's license history records its introduction at `1c88211d779fc46e58640637034a6d68d0a43f08`; the reference is not an assertion that this modified VILA tree exactly matches that NVILA commit. |
| `src/dreamzero` | DreamZero / NVIDIA Corporation; [LICENSE](src/dreamzero/LICENSE) and [COPYRIGHT](src/dreamzero/COPYRIGHT), from historical snapshot `2fcf79eb7d8615153304bbed33f589999a2526f8`. File-level notices for incorporated components remain applicable. |
| `src/openlambda` | λ₀; [LICENSE](src/openlambda/LICENSE), [NOTICE](src/openlambda/NOTICE) and [source manifest](src/openlambda/SOURCE_MANIFEST.json). |
| `src/openpi`, `src/h_rdt`, `src/beingh05` | The license files retained in each component directory and any file-level notices apply. |
| PSI0 source and other native packages imported through PSI0 | PSI0's root Apache 2.0 notice and retained file-level third-party notices; see [Sources](docs/SOURCES.md). |

The immediate source of GR00T, VILA and DreamZero is the vendored, modified code in PSI0 revision `cda3fdfafe47cb6fa75400d013de1bd36e3cea05`. PSI0 did not record the original GR00T/VILA/DreamZero checkout commits alongside these imports. Each component's `SOURCE_MANIFEST.json` distinguishes this known import revision from the historical revision supplying missing license text. These are not claimed to be byte-identical upstream checkouts. Do not infer a later relicensing from newer upstream branches.

Original copyright and license headers are retained. RHSCI modifications to native GR00T files are listed in its component source manifest. Keep these notices, the component manifests and component license files when redistributing.

Within `src/gr00t/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2`, the NVIDIA 2025 headers on `modeling_eagle3_vl.py` and `image_processing_eagle3_vl_fast.py` explicitly identify the MIT License. The [local MIT text](src/gr00t/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/LICENSE) supplies those terms and copyright attribution. The Hugging Face processing/SigLIP files in the same directory retain their own Apache 2.0 headers and use the complete Apache terms above. This records file-level exceptions, not a relicensing of the GR00T N1.6 work or of any pretrained weights.
