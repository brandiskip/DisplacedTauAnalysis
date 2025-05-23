import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import hist
import vector
from hist import Hist, axis, intervals
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import coffea.nanoevents.methods
import json
np.set_printoptions(precision=6, suppress=False, threshold=np.inf)

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

PFNanoAODSchema.mixins["DisMuon"] = "Muon"
samples = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()
'''
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array: 
    mval = first.metric_table(second) 
    return ak.all(mval > threshold, axis=-1)
'''
# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        # add dxy to jet fields
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)
        events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")
        '''
        # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Electron, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Muon, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.DisMuon, 0.4)]
        '''
        ## find staus and their tau children
        gpart = events.GenPart
        events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] 

        events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                          (events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                        (events.staus.distinctChildren.hasFlags("fromHardProcess")) \
                                                         ]
        events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & (events.GenVisTau.parent.hasFlags("fromHardProcess"))]

        events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
        staus_taus = events['staus_taus']

        mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        mask_tauh = ~mask_taul

        one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
        one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

        filtered_events = events[one_tauh_evt & one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau
    
        tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus.eta) < 2.4), axis=-1)
        num_taus = ak.num(filtered_events.staus_taus[tau_selections])
        num_tau_mask = num_taus > 1
        cut_filtered_events = filtered_events[num_tau_mask]

        # Select GenVisStauTaus with |eta| < 2.4 and pt > 20
        cut_filtered_events.GenVisStauTaus = cut_filtered_events.GenVisStauTaus[(cut_filtered_events.GenVisStauTaus.pt > 20) & (abs(cut_filtered_events.GenVisStauTaus.eta) < 2.4)]

        # Select jets with |eta| < 2.4 and pt > 20
        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
        
        # Sort the selected jets by pt (descending) and take the first jet per event
        sorted_by_pt = jets[ak.argsort(jets.pt, ascending=False)]
        leading_pt_jets = ak.singletons(ak.firsts(sorted_by_pt))

        # Sort the selected jets by dxy (descending) and take the first jet per event
        sorted_by_dxy = jets[ak.argsort(abs(jets.dxy), ascending=False)]
        highest_dxy_jets = ak.singletons(ak.firsts(sorted_by_dxy))

        # Make cut on highest score jets
        jets = jets[jets.disTauTag_score1 > 0.90]

        # Sort the selected jets by disTauTag_score1 (descending) and take the first jet per event
        sorted_by_score = jets[ak.argsort(jets.disTauTag_score1, ascending=False)]
        highest_score_jets = ak.singletons(ak.firsts(sorted_by_score))

        ##########################################################################################################
        # GenVisTau matched_leading_jets
        ##########################################################################################################
        num_vis_gen_taus = ak.sum(ak.num(cut_filtered_events.GenVisStauTaus))

        gen_vis_taus_matched_by_pt = leading_pt_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        gen_vis_taus_matched_by_pt = ak.drop_none(gen_vis_taus_matched_by_pt)
        nMatched_gen_vis_taus_highest_pt_jet = ak.sum(ak.num(gen_vis_taus_matched_by_pt))

        # Compute pt efficiency
        efficiency = (nMatched_gen_vis_taus_highest_pt_jet / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0

        # Matching using the dxy leading jets
        gen_vis_taus_matched_highest_dxy_jet = highest_dxy_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        gen_vis_taus_matched_highest_dxy_jet = ak.drop_none(gen_vis_taus_matched_highest_dxy_jet)
        nMatched_gen_vis_taus_highest_dxy_jet = ak.sum(ak.num(gen_vis_taus_matched_highest_dxy_jet))

        # Compute dxy efficiency
        #efficiency = (nMatched_gen_vis_taus_highest_dxy_jet / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0

        # Matching using the leading-score jets.
        gen_vis_taus_matched_by_score = highest_score_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        gen_vis_taus_matched_by_score = ak.drop_none(gen_vis_taus_matched_by_score)
        jet_matched_gen_vis_taus_score = cut_filtered_events.GenVisStauTaus.nearest(highest_score_jets, threshold=0.4)
        jet_matched_gen_vis_taus_score = ak.drop_none(jet_matched_gen_vis_taus_score)
        nMatched_jets_matched_to_gen_vis_tau_highest_score_jet = ak.sum(ak.num(jet_matched_gen_vis_taus_score))

        # Compute score efficiency
        #efficiency = (nMatched_jets_matched_to_gen_vis_tau_highest_score_jet / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0
        
        # Get mass and lifetime from sample_name (e.g., "Stau_100_1mm")
        parts = sample_name.split('_')
        mass = int(parts[1]) 
        lifetime = int(parts[2].replace('mm', '')) # remove mm from 1mm 

        # Create a dictionary with keys: mass, lifetime, efficiency
        efficiency_data = {
            "mass": mass,
            "lifetime": lifetime,
            "efficiency": efficiency
        }

        # Save to JSON file
        json_filename = "pt_efficiency_results.json"

        # Ensure JSON file exists and is not empty before loading
        if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
            try:
                with open(json_filename, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {json_filename} is corrupted. Overwriting with new data.")
                existing_data = []  # Reset the JSON file if it's corrupted
        else:
            print(f"Creating new JSON file: {json_filename}")
            existing_data = []  # If the file doesn't exist or is empty, initialize as an empty list

        existing_data.append(efficiency_data)

        with open(json_filename, "w") as f:
            json.dump(existing_data, f, indent=4)

    # Read the JSON file
    with open("pt_efficiency_results.json", "r") as f:
        efficiency_data = json.load(f)

    # Extract unique masses and lifetimes
    masses = sorted(set(entry["mass"] for entry in efficiency_data))
    lifetimes = sorted(set(entry["lifetime"] for entry in efficiency_data))  # Still in mm

    # Create a mapping from mass/lifetime to an array index
    mass_idx = {m: i for i, m in enumerate(masses)}
    lifetime_idx = {lt: i for i, lt in enumerate(lifetimes)}

    # Build efficiency grid (rows=lifetimes, columns=masses)
    Z = np.zeros((len(lifetimes), len(masses)))

    for entry in efficiency_data:
        m = entry["mass"]
        lt = entry["lifetime"]
        Z[lifetime_idx[lt], mass_idx[m]] = entry["efficiency"]

    # Reverse the y-axis order so lifetimes go from smallest to largest
    Z = Z[::-1]
    lifetimes = lifetimes[::-1]

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the grid as an image
    cmap = cm.get_cmap("plasma")  # Use a color map where efficiency determines shade
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Efficiency is in range [0,1]
    im = ax.imshow(Z, cmap=cmap, norm=norm)

    # Set x-axis (Mass) and y-axis (Lifetime)
    ax.set_xticks(range(len(masses)))
    ax.set_xticklabels([str(m) for m in masses])  # Mass values as labels

    ax.set_yticks(range(len(lifetimes)))
    ax.set_yticklabels([f"{lt} mm" for lt in lifetimes])  # Lifetimes in mm
    
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("Lifetime [mm]")
    plt.title("(nMatched_gen_vis_taus_highest_pt_jet)/(num_vis_gen_taus)[s, $a_{vis , j}, v$]", fontsize=10, pad=15)

    # Loop over data dimensions and create text annotations.
    for i in range(len(lifetimes)):
        for j in range(len(masses)):
            efficiency_value = Z[i, j]
            if efficiency_value > 0:  # Only display values where efficiency is nonzero
                text_color = "white" if efficiency_value < 0.5 else "black"
                ax.text(j, i, f"{efficiency_value:.3f}", ha="center", va="center", 
                        color=text_color, fontsize=9)

    # Add colorbar to indicate efficiency scale
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Efficiency")
    output_file = "pt_efficiency_grid.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved efficiency plot to {output_file}")





