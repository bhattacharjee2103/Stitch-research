# Stitch cross-check — 6 November 2026

The report, presentation, saved research results, code, dependencies and source packages were reviewed and corrected.

## Corrections

- Descriptor indexes now check the encoder checkpoint hash, image resize limit, preprocessing and index version. Incomplete fingerprints and invalid descriptor matrices are rejected.
- Each content polygon belongs to one reference photograph. The optional `reference` field defaults to the first photograph; another view cannot reuse its polygon automatically.
- Rendering selects the reference by both place ID and reference path. Grayscale posts are supported.
- Configured LightGlue confidence settings reach both the application and evaluation engines. Invalid methods and empty shortlists are rejected.
- The CLI reports failed image writes and videos with no decodable frames.
- The standard OpenCV package replaces the contrib distribution at the same 4.11.0 version, satisfying LightGlue without overlapping OpenCV installations.
- Report builds refresh metadata, artwork and source listings, and stop on LaTeX errors. Bundled artwork can be reused when original external asset paths are absent.
- Artifact generation uses the saved evaluation resize setting when computing homography accuracy.
- `make package` rebuilds, checks and refreshes all deliverables. `make check-documents` audits the research and documents, checks manifest hashes and archive freshness, and compiles both ZIPs in newly created temporary directories. Checks inspect the final LaTeX log after reference resolution.

## Additional corrections from the second pass

- Homographies now preserve projection, validity and rendering under small or large nonzero homogeneous scale. Numerically unrepresentable maps are rejected without an SVD failure.
- Crossed, collapsed or nonfinite content-anchor quadrilaterals are rejected. Normalized database anchors must remain within the reference image.
- Parallel document builds finish the report before generating slides, avoiding stale generated figures and numbers. Explicit interpreter paths are resolved correctly across nested Make invocations.
- Demo and profiling indexes honor the configured model, descriptor size and image resize setting.
- Document validation uses the bundled original-layout measurements and reference PDF hash, and works when the original PDF is no longer present at its former folder location.

## Verification

- 31 regression tests passed. Python syntax checks passed. Pip reports no broken requirements.
- 768 dataset, generated-image and checkpoint files verified by SHA256. Model-source revisions and benchmark hashes match.
- All 2,226 prediction decisions independently reconstructed from saved descriptors and pair evidence; all 612 oracle pairs cross-checked. CSV aggregates and ground-truth homography success agree. Reported results required no numerical corrections.
- All 15,624 candidate acceptance decisions recomputed from stored support and the current geometry checks.
- All six saved projection heads reproduce the 72 recorded training-result rows. Float and INT8 ONNX exports each reproduce their retrieval scores on 96 actual Tokyo images; model parity errors agree with the recorded values.
- All four runtime-profile aggregates recomputed from the stored timing samples.
- Parallel `make -j2 PYTHON=.venv/bin/python package` completed successfully. Document verification also passed with the external reference PDF absent.
- Training and held-out place identities are disjoint.
- Actual model-backed CLI image retrieval and a two-frame video execution passed. Demo videos decode with their expected counts: 48 comparison frames, 24 day frames and 24 night frames.
- Rebuilt report: 254 pages, including 176 main-chapter pages. The build enforces a minimum of 200 PDF pages. Presentation PDF and PowerPoint: 30 slides each.
- Document builds have no LaTeX errors, missing citations/references, overfull boxes or oversized floats.
- Author, enrollment 25565017, Dr. R. Balasubramanian and 27 copied assets/listings verified. All slides and the report cover visually inspected.
- Both ZIPs compile independently from fresh extraction directories. Delivered PDFs/PPTX match current canonical outputs; bundled sources and manifest hashes match.
- Original `tep.py`, `tep2.py` and `tep3.py` remain byte-for-byte identical to the original Git revision. Supplied reference PDFs were not edited.

The complete A--D benchmark, training optimisation and runtime timings were not rerun. Saved decisions and model outputs were independently checked; fresh ONNX inference and functional CLI checks were executed. PowerPoint remains a rendered-slide export; edit the supplied LaTeX master for content changes.

Machine-readable evidence is in `results/cross_check_summary.json`, `results/research_audit.json`, `results/final_verification.json`, `results/iitr_documents_verification.json`, and `results/standalone_packages_verification.json`, `results/model_artifacts_audit.json`, and `results/document_portability.json`.

## Expanded-report review on 6 November 2026

The report adds five technical chapters covering projective derivation, decision analysis, failure diagnosis, software reliability and proposed persistent-AR transactions. Two further chapters provide an exhaustive scene atlas for all 24 selected HPatches sequences and 24 Tokyo sites, with photographs and numerical tables generated from the frozen experiment outputs. A complete 612-row oracle-pair register retains failures alongside successes.

The same IITR margins, 11-point type, line spacing, cover artwork and supervisor metadata are retained. No blank pages or font-size changes were introduced to meet the length requirement. The new figures, generated chapters and their inputs have SHA256 provenance. Reported experiment scores and their denominators remain unchanged; design proposals are explicitly labelled.

Representative derivations, HPatches scenes, Tokyo scenes and register pages were rendered and visually reviewed. PDF text was checked for unresolved or replacement characters, and all 48 scene entries were confirmed. Final package build evidence is recorded in the machine-readable summary and standalone-package verification files.
