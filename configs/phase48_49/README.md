# Phase 48/49 Colab Ablation Config Notes

These YAML files are declarative wrappers for the Colab runners. The actual run config is rebuilt in Colab from the strict no-leak `resolved_config.yaml` in `noleak_bundle_phase48_49.zip`, then the listed ablation switches are applied.

Do not run these files directly as standalone training configs unless you first merge them with the corresponding no-leak resolved config.
