# Kestrel deployment profile

The bundled `hpca/config/platform.yaml` is currently configured for NREL Kestrel and contains
site paths for Python environments, VASP/POTCAR data, LAMMPS, PACKMOL, MACE models, MPI, SLURM
accounts, partitions, and the repository-local daemon inbox.

Before deployment, verify:

- All executable and environment paths exist on login, daemon, and compute nodes as needed.
- Scheduler accounts and partitions are valid for CPU, H100, and long-running daemon jobs.
- VASP licensing and POTCAR access are authorized.
- The daemon's allowed roots contain every intended project directory.
- Shared storage semantics support atomic rename and advisory locking used by state/leases.
- Wall-time and resource defaults comply with current site policy.

Never copy user credentials, tokens, private structure data, or licensed POTCAR content into
the documentation or repository. Document symbolic configuration keys; keep deployment values
in controlled configuration.
