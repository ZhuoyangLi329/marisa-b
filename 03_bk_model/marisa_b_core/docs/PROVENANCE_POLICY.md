# Provenance policy

## Input classes

Every file used by this project belongs to exactly one class.

1. **External requirement**: task, scope, or acceptance text supplied by the
   user.  Private task attachments are not distributed in the public source
   tree; release-facing requirements are restated in current documentation.
2. **Third-party source**: upstream code or reference data with an upstream
   identity, acquisition method, immutable commit or tree digest, license, and
   local SHA256 inventory.
3. **Read-only scientific input**: Quijote/JAXRecon measurements, realization
   membership, coordinates, power spectra, catalog summaries, or production
   configuration.  The production-data manifest records a root-relative URI,
   content hash, size, semantics, and sample membership without publishing a
   machine-local absolute path.
4. **New repository source**: code authored in this repository and reviewed
   through Git history.
5. **First-party current import**: current MARISA-B source selected from the
   user's active portable handoff, admitted with source path, destination path,
   byte-preserving source and imported-blob SHA256, import commit, timestamp,
   and role in `provenance/MARISA_B_IMPORT.tsv`.

A file with unknown or mixed ancestry is observation-only and cannot be linked,
included, imported, or copied into a formal build.

The canonical native core, native CLI, adapters, and production runners are
first-party current imports at the exact commits recorded in
`provenance/MARISA_B_IMPORT.tsv`, not independent reimplementations.  This
statement supersedes the older blanket description of every active
implementation file as independently written inside the restart repository.
Their authoritative acquisition source is the active portable handoff
recorded in that ledger.

## Historical material

Superseded task attachments, data contracts, figures, logs, and forensic
archives are excluded from the public source tree.  They are prohibited
scientific inputs.  A numerical value from a private archive may appear in a
regression report only when it is explicitly labelled historical and its exact
source hash is recorded.

Byte equality between a first-party current import and a file that also exists
in a historical archive does not make the archive its acquisition source.  The
current-import ledger is admissible only when the active portable source was
selected directly and both sides of the byte-preserving transfer were hashed.
Historical archives remain prohibited acquisition paths.

## Third-party admission

Before third-party code enters the repository, the source ledger must contain:

- project and upstream URL;
- upstream commit, release, or an explicitly justified content-addressed tree
  digest;
- acquisition date and method;
- local destination and complete inventory hash;
- license identifier and license-file hash;
- a statement of which files are compiled.

Copying a dependency from a prior MARISA-B worktree is forbidden even if it
appears identical to upstream.  It must be reacquired from the independently
identified source and verified against the ledger.

The repository root `LICENSE` governs MARISA-B first-party source.  Every
compiled, linked, vendored, or source-generating third-party dependency must
also appear in `provenance/third_party_sources.tsv` and
`THIRD_PARTY_NOTICES.md`.  A generated source file must record the immutable
upstream identity, exact input hashes, transformation, modification date, and
output hashes.  The third-party verifier must validate the recorded bytes and
Git identities; checking only that ledger fields are non-empty is insufficient
for a formal release.

## Formal build contract

A formal build is valid only when:

- the Git index and worktree are clean;
- `HEAD` is a named commit;
- third-party sources, the source boundary, and any selected external-data
  manifest verify;
- first-party current-import source and imported-blob hashes verify against the
  recorded import commit;
- the compiler path and version are captured;
- the complete compile and link commands are captured without shell elision;
- every compiled or generated input is hashed;
- the binary embeds the Git commit, tracked-tree digest, program identity, and
  build kind;
- `readelf`, `ldd`, binary SHA256, and environment/thread settings are recorded;
- the output directory is content-addressed and never overwritten.

Builds made from a dirty tree are diagnostic only and must not be promoted.

## Formal run contract

Every formal run bundle records the verified build-manifest hash, data-contract
hash, full arguments, environment, host/job identity, start/end times, stdout,
stderr, exit status, output hashes, numerical tolerances, integration errors,
and evaluation counts.  Re-running the same bundle must not mutate the first
bundle.

Scientific-gate reports must distinguish accepted evidence from diagnostic
checks.  A failed report is immutable evidence of failure and cannot be
silently replaced by a report with relaxed criteria.
