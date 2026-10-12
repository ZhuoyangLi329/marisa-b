# Validation record

- Migration was performed by copy from the original paths. The migration script rechecked every source SHA256 after writing and recorded `source_files_unchanged=true` in `MIGRATION_MANIFEST.json`.
- Python AST parsing and `python -m py_compile` pass for every migrated Python file.
- Shell syntax (`bash -n`) passes for both Slurm templates.
- Active Python and Slurm files contain no old repository path and no old archive dependency.
- The shared cosmodesi environment contains numpy, h5py, mpi4py, pmesh, jax, jaxrecon, jaxpower, clustering_statistics and cosmoprimo.
- A direct login-node import of `mpi4py` is not a valid production test: this login shell did not expose `libmpi.so`. Production must use the supplied Slurm `srun` template, where the NERSC MPI runtime is loaded for the job.
- No simulation, result directory, figure, Fisher, MCMC, or batch production job was run in this migration.

The temporary initial bulk manifest copy was moved to `/tmp/reconstruction_png_v2_migration_20260914/initial_bulk_manifests` on the NERSC shared filesystem; no source file was deleted or moved.
