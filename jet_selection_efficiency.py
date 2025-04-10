# jet_selection_efficiency.py
import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import hist
from hist import Hist, axis, intervals
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

# Import functions from jet_selection.py
from jet_selection import (
    process_events,
    select_and_define_leading_jets,
    match_gen_taus,
    flatten_gen_tau_vars,
)

# Import functions from jet_plotting.py
from jet_plotting import (
    get_ratio_histogram,
    plot_efficiency,
    plot_dxy_efficiency,
    plot_pt_efficiency,
    plot_2d_histogram,
    plot_numJets_histogram,
    plot_sample_grid,
    plot_matched_vs_unmatched_jets
)

# Load the file
filenames = {
    'Stau_100_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
}

samples = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()

# Create output folder for histograms
output_dir = "histograms"
os.makedirs(output_dir, exist_ok=True)

# Create dictionaries to store efficiency histograms for overlay
dxy_eff_data = {}
pt_eff_data = {}      # keys: sample names, values: pt efficiency histogram
pt_zoom_eff_data = {} # keys: sample names, values: zoom pt efficiency histogram

num_hist_pt_dict = {}  # Stores numerator histograms for total pt efficiency per sample
den_hist_pt_dict = {}  # Stores denominator histograms for total pt efficiency per sample

# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        
        # Process events: select staus, taus, and apply event filters
        cut_filtered_events = process_events(events)
        
        # Select jets and define leading jets (using the filtered events)
        jets, leading_pt_jets, leading_score_jets, numJets = select_and_define_leading_jets(cut_filtered_events)
        
        # Match gen taus to jets using both pt-based and leading-score methods
        (gen_taus,
         gen_taus_matched_by_pt, jet_matched_gen_taus_pt,
         gen_vis_taus_matched_by_score, jet_matched_gen_taus_score,
         matched_leading_jets_flat, all_unmatched_jets_pt) = match_gen_taus(cut_filtered_events, leading_pt_jets, leading_score_jets, jets)
        
        # Flatten variables for histogram filling 
        (gen_taus_flat_dxy, gen_taus_flat_pt,
         gen_taus_matched_by_pt_flat_dxy, gen_taus_matched_by_pt_flat_pt) = flatten_gen_tau_vars(gen_taus, gen_taus_matched_by_pt)

        # Plot dxy efficiency (prompt and overall)
        dxy_eff_raw = plot_dxy_efficiency(gen_taus_flat_dxy, gen_taus_matched_by_pt_flat_dxy, output_dir, sample_name)
        
        # Store first two elements of dxy_eff_raw (hist_pt_num, hist_pt_den) in dictionary:
        dxy_eff_data[sample_name] = dxy_eff_raw[:2]

        # Plot pt efficiency (overall and zoom)
        pt_eff_raw = plot_pt_efficiency(gen_taus_flat_pt, gen_taus_matched_by_pt_flat_pt, output_dir, sample_name)

        # Store in dictionary:
        pt_eff_data[sample_name] = pt_eff_raw[:2]
        pt_zoom_eff_data[sample_name] = pt_eff_raw[2:]
        
        # 2D histogram of gen_tau_dxy vs. gen_tau_pT
        plot_2d_histogram(gen_taus_flat_pt, gen_taus_flat_dxy, output_dir, sample_name)

        plot_numJets_histogram(numJets, output_dir, sample_name)

        plot_matched_vs_unmatched_jets(matched_leading_jets_flat, all_unmatched_jets_pt, output_dir, sample_name)

        # Store numerator and denominator histograms for sample grid plot
        num_hist_pt_dict[sample_name] = pt_eff_raw[1]  # Numerator histogram (matched pT)
        den_hist_pt_dict[sample_name] = pt_eff_raw[0]  # Denominator histogram (all gen pT)

    output_file_pt = os.path.join(output_dir, "eff_grid_vs_stau_mass.pdf")
    plot_sample_grid(num_hist_pt_dict, den_hist_pt_dict, "Total Efficiency vs Stau Mass", output_file_pt)

    # Overlay dxy efficiency (overall)
    fig, ax = plt.subplots(figsize=(8, 6))
    for sample_name, (num_hist, den_hist) in dxy_eff_data.items():
        ratio_hist, yerr = get_ratio_histogram(num_hist, den_hist)
        ratio_hist.plot1d(ax=ax, histtype="errorbar", yerr=yerr, xerr=True, flow="none",
                          label=sample_name)
    ax.set_xlabel("dxy [cm]")
    ax.set_ylabel("Efficiency")
    ax.set_title("Overlay of dxy Efficiencies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay_eff_dxy.pdf"))
    plt.close(fig)

    # Overlay pt efficiency 
    fig, ax = plt.subplots(figsize=(8, 6))
    for sample_name, (num_hist, den_hist) in pt_eff_data.items():
        ratio_hist, yerr = get_ratio_histogram(num_hist, den_hist)
        ratio_hist.plot1d(ax=ax, histtype="errorbar", yerr=yerr, xerr=True, flow="none",
                          label=sample_name)
    ax.set_xlabel("pT [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_title("Overlay of pT Efficiencies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay_eff_pt.pdf"))
    plt.close(fig)

    # Overlay pt efficiency (zoomed)
    fig, ax = plt.subplots(figsize=(8, 6))
    for sample_name, (num_hist, den_hist) in pt_zoom_eff_data.items():
        ratio_hist, yerr = get_ratio_histogram(num_hist, den_hist)
        ratio_hist.plot1d(ax=ax, histtype="errorbar", yerr=yerr, xerr=True, flow="none",
                          label=sample_name)
    ax.set_xlabel("pT [GeV] (Zoomed)")
    ax.set_ylabel("Efficiency")
    ax.set_title("Overlay of pt Efficiencies (Zoomed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay_eff_pt_zoom.pdf"))
    plt.close(fig)