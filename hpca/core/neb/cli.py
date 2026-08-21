"""hpca/core/neb/cli.py — Command-line entry point for the NEB chain generator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .image_tools import generate_neb_chain
from hpca.registry.submission import write_submission


def _write_neb_submissions(mech_dir: Path) -> list[Path]:
    """Write endpoint/image scripts through the canonical submission registry."""
    metadata = json.loads((mech_dir / "migration_metadata.yaml").read_text())
    n_images = int(metadata["n_images"])
    endpoint_tags = ["00", f"{n_images + 1:02d}"]
    scripts = [write_submission(
        mech_dir / "sub_endpoints.sh", "vasp_neb_endpoints",
        f"{mech_dir.name}_endpoints", endpoint_tags=endpoint_tags,
        time_key="neb_endpoint",
    )]
    tags = [f"{idx:02d}" for idx in range(1, n_images + 1)]
    for chunk_index, start in enumerate(range(0, len(tags), 6), start=1):
        chunk = tags[start:start + 6]
        scripts.append(write_submission(
            mech_dir / f"sub_images_{chunk_index}.sh", "vasp_neb_images",
            f"{mech_dir.name}_images_{chunk_index}", image_tags=chunk,
            time_key="neb_image",
        ))
    write_submission(
        mech_dir / "submit_all.sh", "submit_fanout", f"{mech_dir.name}_all",
        scripts=[path.name for path in scripts],
    )
    return scripts


def main() -> None:
    """
    Command‑line entry point.

    Parses user arguments, determines which mechanisms to generate, and
    calls generate_neb_chain() for each. Writes master submission scripts
    and prints a summary. Submission files are produced by the canonical registry.

    The script does NOT run VASP; it only generates input files.
    """
    parser = argparse.ArgumentParser(
        description="Generate constrained NEB input files (POSCARs, SLURM scripts). "
                    "Uses a non‑linear path predictor. No VASP NEB is run."
    )
    parser.add_argument("poscar", type=Path, help="Path to initial relaxed POSCAR")
    parser.add_argument("--template", type=Path, default=None,
                        help="Directory containing INCAR, KPOINTS, POTCAR to copy into each image dir. "
                             "If not given, tries to find them in the POSCAR directory.")
    parser.add_argument("--mobile", default="Li", help="Mobile ion element (for native species)")
    parser.add_argument("--mechanism",
                        choices=["vacancy", "interstitial", "dopant", "both", "all"],
                        default="all",
                        help="Which mechanism(s) to generate. 'both' = vacancy + native interstitial; 'all' = vacancy + native + dopant (default).")
    parser.add_argument("--dopant_element", default=None,
                        help="Element for dopant (only used when generating dopant mechanism)")
    parser.add_argument("--directions", nargs="+",
                        default=["[100]", "[010]", "[001]", "[110]", "[101]", "[011]", "[111]"],
                        help="Crystallographic directions for anchor building.")
    parser.add_argument("--n_images", type=int, default=None,
                        help="Override automatic number of images")
    parser.add_argument("--spacing", type=float, default=0.2,
                        help="Target spacing between images (Å) – used when --n_images not given (default: 0.2)")
    parser.add_argument("--hop_distance", type=float, default=2.8,
                        help="Step distance along each direction (Å)")
    parser.add_argument("--neighbor_radius", type=float, default=4.0,
                        help="Selective dynamics sphere radius (Å)")
    parser.add_argument("--void_cutoff", type=float, default=2.5,
                        help="Void detection cutoff (Å)")
    parser.add_argument("--anchors", nargs="+", type=float, default=None,
                        help="Explicit anchor coordinates (3-tuples: x y z x y z ...)")
    parser.add_argument("--output", "-o", type=Path, default=Path("./neb"),
                        help="Root output directory (default: ./neb)")
    parser.add_argument("--perturb", action="store_true",
                        help="Add small random perturbation to break symmetry")
    parser.add_argument("--perturb_strength", type=float, default=0.02,
                        help="Perturbation strength (Å)")
    parser.add_argument("--cutoff_radius", type=float, default=2.0,
                        help="Repulsion cutoff radius for non‑linear path predictor (Å).")
    parser.add_argument("--spring_constant", type=float, default=1.0,
                        help="Spring constant for smoothness of non‑linear path.")
    parser.add_argument("--repulsion_scale", type=float, default=100.0,
                        help="Repulsion strength for non‑linear path.")

    args = parser.parse_args()

    anchors = None
    if args.anchors is not None:
        if len(args.anchors) % 3 != 0:
            parser.error("--anchors must be a list of 3-tuples (x y z x y z ...)")
        anchors = [args.anchors[i:i+3] for i in range(0, len(args.anchors), 3)]

    # Determine which mechanisms to run
    if args.mechanism == "all":
        mechanisms = ["vacancy", "interstitial", "dopant"]
    elif args.mechanism == "both":
        mechanisms = ["vacancy", "interstitial"]
    else:
        mechanisms = [args.mechanism]

    print(f"\n[INFO] Generating: {', '.join(mechanisms)}\n")

    generated_dirs = []
    generated_scripts: list[Path] = []
    for mech in mechanisms:
        try:
            dir_path = generate_neb_chain(
                poscar_path=args.poscar,
                output_dir=args.output,
                template_dir=args.template,
                mobile_element=args.mobile,
                mechanism=mech,
                dopant_element=args.dopant_element,
                directions=args.directions,
                n_images=args.n_images,
                spacing=args.spacing,
                hop_distance=args.hop_distance,
                neighbor_radius=args.neighbor_radius,
                void_cutoff=args.void_cutoff,
                anchors=anchors,
                perturb=args.perturb,
                perturb_strength=args.perturb_strength,
                cutoff_radius=args.cutoff_radius,
                spring_constant=args.spring_constant,
                repulsion_scale=args.repulsion_scale,
            )
            generated_dirs.append(dir_path)
            generated_scripts.extend(_write_neb_submissions(dir_path))
        except RuntimeError as e:
            print(f"\n[WARNING] Skipping {mech}: {e}\n")

    if generated_scripts:
        relative_scripts = [str(path.relative_to(args.output)) for path in generated_scripts]
        write_submission(
            args.output / "submit_all.sh", "submit_fanout", "all_neb_mechanisms",
            scripts=relative_scripts,
        )
        print(f"\nSubmit all generated jobs with: cd {args.output} && bash submit_all.sh")
    else:
        print("\nNo mechanisms were generated.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for mech in mechanisms:
        traj = args.output / f"{mech}_trajectory.xyz"
        if traj.exists():
            print(f"✓ {mech}: {traj}")
            master = args.output / mech / "submit_all.sh"
            if master.exists():
                print(f"  -> Submission master: {master}")
        else:
            print(f"✗ {mech}: not generated (check warnings above)")
    top_master = args.output / "submit_all.sh"
    if top_master.exists():
        print(f"\nTop‑level master script: {top_master}")
        print("Run: cd neb && bash submit_all.sh  (or submit each registered script separately)")

    print("\nOpen .xyz files in OVITO to view migration as a movie.")


if __name__ == "__main__":
    main()
