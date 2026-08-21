"""
viz — NREL-styled Plotly and matplotlib visualisation package for the HPCA
Battery Materials Pipeline.

Usage
-----
    from hpca.viz import apply_nrel_theme, NREL_COLORS
    from hpca.viz.transport import plot_arrhenius_multi, plot_msd_by_temperature
    from hpca.viz.publication import PublicationFigure

Always consult Section 11 of the Pipeline Technical Report before creating
any new figure:
    Primary:  /path/to/workspace/manuscripts/v2/Pipeline_Technical_Report_v2_20260612.docx
    Fallback: /path/to/workspace/manuscripts/Pipeline_Technical_Report_20260608.docx

Mandatory export rule
---------------------
Every figure must be saved as BOTH an image file AND a CSV data companion:
    fig.write_image(f"{output_dir}/{name}.png", scale=2)   # Plotly
    fig.write_html(f"{output_dir}/{name}.html")            # Plotly
    np.savetxt(f"{data_dir}/{name}.csv", data, ...)        # data companion

Output directory convention:
    {PROJECT}/Analysis/continuum_figures/   <- PNG + HTML
    {PROJECT}/Analysis/continuum_data/      <- CSV
    {PROJECT}/Analysis/continuum_model.py   <- script (version-controlled)

Public re-exports
-----------------
theme          : NREL_COLORS, NREL_LIGHT_COLORS, NREL_TEMPLATE,
                 NREL_SEQUENTIAL_COLORSCALE, NREL_DIVERGING_COLORSCALE,
                 apply_nrel_theme, save_figure, multi_panel, add_annotation_box
transport      : plot_arrhenius_multi, plot_msd_multi, plot_diffusivity_bar,
                 plot_conductivity_vtf, plot_haven_ratio,
                 plot_nernst_planck_profile, plot_msd_by_temperature
comparison     : plot_mlip_benchmark_heatmap, plot_benchmark_radar,
                 plot_crossproject_diffusivity, plot_sei_comparison,
                 plot_continuum_summary_dashboard, build_benchmark_html_report
structure      : plot_rdf, plot_coordination_histogram, plot_bond_angle,
                 plot_vanhove, plot_non_gaussian, plot_displacement_distribution
continuum_viz  : plot_concentration_profile, plot_phase_field,
                 plot_stress_profile, plot_sei_growth, plot_kjma
dos_band       : plot_dos, plot_bader_charges
publication    : PublicationFigure, JOURNAL_WIDTHS, ELEMENT_COLORS,
                 quick_transport_figure, quick_characterization_figure
report         : ProjectReport, CharacterizationResult,
                 generate_full_report
"""

# ---------------------------------------------------------------------------
# Plotting guide — quick reference for all available functions
# ---------------------------------------------------------------------------

PLOTTING_GUIDE: str = """
hpca.viz PLOTTING GUIDE
=======================
Always consult Section 11 of the Pipeline Technical Report first.
Python env: hpc.python_cladue in hpca/config/platform.yaml

THEME & UTILITIES (theme.py)
  apply_nrel_theme(fig, title, xlabel, ylabel, width, height)
      Apply NREL colour scheme and typography to a go.Figure.
  save_figure(fig, path_stem, formats=['png','html'])
      Save Plotly figure as PNG (scale=2) + HTML in one call.
  multi_panel(rows, cols, **kwargs) -> go.Figure
      Create a multi-panel subplot figure with NREL theme pre-applied.
  add_annotation_box(fig, text, x=0.02, y=0.98)
      Add a floating text box annotation to a figure.
  NREL_COLORS          : list of 8 hex colour strings (blue, gold, green, …)
  NREL_LIGHT_COLORS    : corresponding pastel versions
  NREL_TEMPLATE        : raw Plotly layout dict for manual template use
  NREL_SEQUENTIAL_COLORSCALE, NREL_DIVERGING_COLORSCALE : Plotly colorscales

TRANSPORT (transport.py)
  plot_arrhenius_multi(data, title) -> go.Figure
      Overlay Arrhenius fits for multiple MLIPs / projects.
      data = {label: {T_K: D_m2s, ...}}
  plot_msd_multi(msd_results, labels, T_K, project) -> go.Figure
      Overlay MSD curves + linear fit from multiple runs.
      msd_results = [{"time_ps": arr, "msd_angsq": arr}, ...]
  plot_msd_by_temperature(positions_by_temp, dt_ps, species) -> go.Figure
      Compute MSD from raw position arrays at multiple temperatures,
      plot MSD vs time subplot grid + Arrhenius summary panel.
      positions_by_temp = {T_K: positions_array (n_frames, n_atoms, 3)}
  plot_diffusivity_bar(projects, D_values, mlips, T_K) -> go.Figure
      Grouped bar chart comparing D across projects and MLIPs.
  plot_conductivity_vtf(vtf_params, T_range, exp_point) -> go.Figure
      Vogel-Tammann-Fulcher ionic conductivity vs temperature.
  plot_haven_ratio(D_tracer, D_conductivity, project) -> go.Figure
      Haven ratio H_R = D_cond / D_tracer vs temperature.
  plot_nernst_planck_profile(NP_data, project) -> go.Figure
      Concentration and electric potential profiles from NP model.

STRUCTURE (structure.py)
  plot_rdf(rdf_data, labels, project) -> go.Figure
      Radial distribution function overlay for multiple species pairs.
  plot_coordination_histogram(coord_data, species, project) -> go.Figure
      Coordination number distribution as a histogram.
  plot_bond_angle(angle_data, labels, project) -> go.Figure
      Bond angle distribution.
  plot_vanhove(vanhove_data, times_ps, project) -> go.Figure
      Van Hove self-correlation function G_s(r, t).
  plot_non_gaussian(alpha2_data, times_ps, project) -> go.Figure
      Non-Gaussian parameter α₂(t) for dynamic heterogeneity.
  plot_displacement_distribution(disp_data, T_K, project) -> go.Figure
      Atomic displacement magnitude distribution.

CONTINUUM (continuum_viz.py)
  plot_concentration_profile(x_nm, c_profiles, times, project) -> go.Figure
      1D Li concentration profiles from Fick/phase-field model.
  plot_phase_field(x_nm, phi_profiles, times, project) -> go.Figure
      Phase-field order parameter φ(x, t).
  plot_stress_profile(x_nm, sigma_profiles, times, project) -> go.Figure
      Vegard stress σ(x, t) profiles.
  plot_sei_growth(times_s, L_nm, fit_params, project) -> go.Figure
      SEI/interphase layer thickness L(t) = A·t^n with power-law fit.
  plot_kjma(times_s, X_frac, kjma_params, project) -> go.Figure
      KJMA transformation kinetics X(t) = 1 − exp(−(kt)^n).

DOS / ELECTRONIC (dos_band.py)
  plot_dos(doscar_data, species_labels, project) -> go.Figure
      Density of states and projected DOS (PDOS) from VASP DOSCAR.
  plot_bader_charges(acf_data, species, project) -> go.Figure
      Bader charge per atom from ACF.dat, grouped by species.

COMPARISON / BENCHMARK (comparison.py)
  plot_mlip_benchmark_heatmap(benchmark_df, metric) -> go.Figure
      Heatmap of MLIP benchmark scores (E-RMSE, F-RMSE) per project.
  plot_benchmark_radar(scores_dict, categories) -> go.Figure
      Radar chart comparing multiple MLIPs across performance categories.
  plot_crossproject_diffusivity(D_table, temps_K) -> go.Figure
      D vs temperature across all projects on one Arrhenius plot.
  plot_sei_comparison(sei_data, projects) -> go.Figure
      SEI thickness + growth rate comparison across projects.
  plot_continuum_summary_dashboard(summary_data) -> go.Figure
      Multi-panel dashboard: D, Ea, σ, SEI across all projects.
  build_benchmark_html_report(benchmark_dict, output_path)
      Write a self-contained HTML report with all benchmark figures.

PUBLICATION / MATPLOTLIB (publication.py)
  PublicationFigure(journal, n_cols, n_rows, **kwargs)
      Context-managed matplotlib figure sized for journal submission.
      Journals: 'nature', 'acs', 'rsc'  (single/double column widths).
  quick_transport_figure(data, output_stem, journal) -> PublicationFigure
      One-call transport summary: Arrhenius + MSD panels, saves PDF+PNG+CSV.
  quick_characterization_figure(data, output_stem, journal) -> PublicationFigure
      One-call characterization: DOS + RDF + Bader panels.
  JOURNAL_WIDTHS    : dict mapping journal names to (single_col, double_col) inches
  ELEMENT_COLORS    : dict mapping element symbols to publication-standard hex colours

REPORTS (report.py)
  ProjectReport(project_name, output_dir)
      Collects all analysis results and generates a consolidated PDF report.
  CharacterizationResult(species, D_m2s, Ea_eV, sigma_Scm, ...)
      Dataclass for storing per-project characterization results.
  generate_full_report(results_list, output_path)
      Generate a cross-project comparison report from a list of
      CharacterizationResult objects.
"""

from .theme import (
    NREL_COLORS,
    NREL_LIGHT_COLORS,
    NREL_TEMPLATE,
    NREL_SEQUENTIAL_COLORSCALE,
    NREL_DIVERGING_COLORSCALE,
    apply_nrel_theme,
    save_figure,
    multi_panel,
    add_annotation_box,
)

from .transport import (
    plot_arrhenius_multi,
    plot_msd_multi,
    plot_diffusivity_bar,
    plot_conductivity_vtf,
    plot_haven_ratio,
    plot_nernst_planck_profile,
    plot_msd_by_temperature,
)

from .comparison import (
    plot_mlip_benchmark_heatmap,
    plot_benchmark_radar,
    plot_crossproject_diffusivity,
    plot_sei_comparison,
    plot_continuum_summary_dashboard,
    build_benchmark_html_report,
)

from .structure import (
    plot_rdf,
    plot_coordination_histogram,
    plot_bond_angle,
    plot_vanhove,
    plot_non_gaussian,
    plot_displacement_distribution,
)

from .continuum_viz import (
    plot_concentration_profile,
    plot_phase_field,
    plot_stress_profile,
    plot_sei_growth,
    plot_kjma,
)

from .dos_band import (
    plot_dos,
    plot_bader_charges,
)

from .publication import (
    PublicationFigure,
    JOURNAL_WIDTHS,
    ELEMENT_COLORS,
    quick_transport_figure,
    quick_characterization_figure,
)

from .report import (
    ProjectReport,
    CharacterizationResult,
    generate_full_report,
)

__all__ = [
    # guide
    "PLOTTING_GUIDE",
    # theme
    "NREL_COLORS",
    "NREL_LIGHT_COLORS",
    "NREL_TEMPLATE",
    "NREL_SEQUENTIAL_COLORSCALE",
    "NREL_DIVERGING_COLORSCALE",
    "apply_nrel_theme",
    "save_figure",
    "multi_panel",
    "add_annotation_box",
    # transport
    "plot_arrhenius_multi",
    "plot_msd_multi",
    "plot_diffusivity_bar",
    "plot_conductivity_vtf",
    "plot_haven_ratio",
    "plot_nernst_planck_profile",
    "plot_msd_by_temperature",
    # comparison
    "plot_mlip_benchmark_heatmap",
    "plot_benchmark_radar",
    "plot_crossproject_diffusivity",
    "plot_sei_comparison",
    "plot_continuum_summary_dashboard",
    "build_benchmark_html_report",
    # structure
    "plot_rdf",
    "plot_coordination_histogram",
    "plot_bond_angle",
    "plot_vanhove",
    "plot_non_gaussian",
    "plot_displacement_distribution",
    # continuum_viz
    "plot_concentration_profile",
    "plot_phase_field",
    "plot_stress_profile",
    "plot_sei_growth",
    "plot_kjma",
    # dos_band
    "plot_dos",
    "plot_bader_charges",
    # publication
    "PublicationFigure",
    "JOURNAL_WIDTHS",
    "ELEMENT_COLORS",
    "quick_transport_figure",
    "quick_characterization_figure",
    # report
    "ProjectReport",
    "CharacterizationResult",
    "generate_full_report",
]
