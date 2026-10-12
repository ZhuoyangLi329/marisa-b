# Historical boundary

The public source tree does not carry the superseded R0 task attachments,
machine-specific data contracts, covariance-of-the-mean reports, or their
figures.  They remain recoverable from the private integration ancestry but
are not scientific inputs to the current model.

One historical compatibility module remains under `scripts/legacy/`: it
supplies frozen template and parameter containers used by the migrated
production adapters.  Its executable mean-covariance likelihood is disabled
unless a caller supplies an explicit acknowledgement flag.
