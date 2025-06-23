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
    #eventsnotselected_samples[sample_name] = NanoEventsFactory.from_root(
        #{files: "EventsNotSelected"},
        #schemaclass=NanoAODSchema,
    #).events()

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array: 
    mval = first.metric_table(second) 
    return ak.all(mval > threshold, axis=-1)

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
        vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.parent.vx
        vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.parent.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        parent_with_Lxy = ak.with_field(events.GenVisTau.parent, Lxy, where="Lxy")
        events['GenVisTau'] = ak.with_field(events.GenVisTau, parent_with_Lxy, where="parent")

        events.Muon = events.Muon[(events.Muon.pt > 20) & (abs(events.Muon.eta) < 2.4) & (events.Muon.looseId == 1)]
        events.DisMuon = events.DisMuon[(events.DisMuon.pt > 20) & (abs(events.DisMuon.eta) < 2.4) & (events.DisMuon.looseId == 1)]
        events.Electron = events.Electron[(events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.4) & (events.Electron.convVeto)]
        events.Photon = events.Photon[(events.Photon.pt > 20) & (abs(events.Photon.eta) < 2.4) & (events.Photon.electronVeto)]
        
        # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
        '''
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
        #events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & \
                                                        #(abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & \
                                                        #(events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & \
                                                        #(events.GenVisTau.parent.hasFlags("fromHardProcess")) & \
                                                        #(events.GenVisTau.parent.Lxy < 10.0)]

        events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & \
                                                        (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & \
                                                        (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & \
                                                        (events.GenVisTau.parent.hasFlags("fromHardProcess"))]

        events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
        staus_taus = events['staus_taus']

        mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        mask_tauh = ~mask_taul

        one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
        one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

        filtered_events = events[one_tauh_evt & one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau
    
        tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus_taus.eta) < 2.4), axis=-1)
        num_taus = ak.num(filtered_events.staus_taus[tau_selections])
        num_tau_mask = num_taus > 1
        cut_filtered_events = filtered_events[num_tau_mask]

        # Select GenVisStauTaus with |eta| < 2.4 and pt > 20
        cut_filtered_events.GenVisStauTaus = cut_filtered_events.GenVisStauTaus[(cut_filtered_events.GenVisStauTaus.pt > 20) & (abs(cut_filtered_events.GenVisStauTaus.eta) < 2.4)]

        # Select jets with |eta| < 2.4 and pt > 20
        # Use these jet selections to trouble shoot looking for how each veto effects efficiency
        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]

        # add isTightLeptonVeto to jets if lepton veto needed
        #jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTightLeptonVeto)]
        #jets = jets[jets.disTauTag_score1 > 0.90]

        # Sort the selected jets by disTauTag_score1 (descending) and take the first jet per event
        #sorted_by_score = jets[ak.argsort(jets.disTauTag_score1, ascending=False)]
        #highest_score_jets = ak.singletons(ak.firsts(sorted_by_score))

        #sorted_by_pt = jets[ak.argsort(jets.pt, ascending=False)]
        #leading_pt_jets = ak.singletons(ak.firsts(sorted_by_pt))
        #jet_matched_gen_vis_taus = cut_filtered_events.GenVisStauTaus.nearest(jets, threshold=0.4)
        #jet_matched_gen_vis_taus = ak.drop_none(jet_matched_gen_vis_taus)

        jet_matched_gen_vis_taus = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus = ak.drop_none(jet_matched_gen_vis_taus)

        #jet_matched_gen_vis_taus_score = cut_filtered_events.GenVisStauTaus.nearest(highest_score_jets, threshold=0.4)
        #jet_matched_gen_vis_taus_score = ak.drop_none(jet_matched_gen_vis_taus_score)

        # Define jets before veto
        jets_before_veto = jet_matched_gen_vis_taus
        #jets_before_veto = jet_matched_gen_vis_taus_score

        ##################################################################################################################################
        # tau parents and children that come from cut_filtered_events.GenVisStauTaus so NOT matched to jets
        ##################################################################################################################################
        #tau_parents = cut_filtered_events.GenVisStauTaus.parent
        #tau_children = tau_parents.distinctChildren  # GenParticles

        ##################################################################################################################################
        # tau parents and children that come from cut_filtered_events.GenVisStauTaus MATCHED to jets
        ##################################################################################################################################
        tau_parents = jets_before_veto.parent
        tau_children = tau_parents.distinctChildren

        photons_from_tau = tau_children[abs(tau_children.pdgId) == 22]
        is_direct_tau_product = photons_from_tau.hasFlags("isDirectTauDecayProduct")

        # Filter: only photons that are *direct* products of the tau decay
        direct_photons_from_tau = photons_from_tau[is_direct_tau_product]
        tau_has_direct_photon = ak.any(is_direct_tau_product, axis=1)
        tau_children_with_photon = tau_children[tau_has_direct_photon]

        # Remove neutrinos (12, 14, 16)
        tau_children_with_photon_no_nu = tau_children_with_photon[
            (abs(tau_children_with_photon.pdgId) != 12)
            & (abs(tau_children_with_photon.pdgId) != 14)
            & (abs(tau_children_with_photon.pdgId) != 16)
        ]
     
        all_tau_children_pdgids = ak.to_numpy(ak.ravel(abs(tau_children_with_photon_no_nu.pdgId.compute())))
        plt.hist(all_tau_children_pdgids, bins=range(0, 330), histtype='step')
        plt.xlabel("PDG ID of GenVisStauTau children (with direct tau->γ)")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.title(f"{sample_name} Matched Photon Veto Direct Tau Decay Children (γ included)")
        plt.savefig(f"{sample_name}_matchedPhotonVetoTauChildren_DirectPhoton.pdf")
        plt.close()

        #pt_mask = (jets_before_veto.pt > 30) & (jets_before_veto.pt < 100)
        #jets_before_veto = jets_before_veto[pt_mask]

        #pt_mask = (jets_before_veto.pt > 300) & (jets_before_veto.pt < 400)
        #jets_before_veto = jet_matched_gen_vis_taus[pt_mask]
        '''
        plt.figure(figsize=(8, 6))

        for label, veto_obj in [
            ("Photon", cut_filtered_events.Photon),
            ("Electron", cut_filtered_events.Electron),
            ("Muon", cut_filtered_events.Muon),
            ("DisMuon", cut_filtered_events.DisMuon)
        ]:
            # Skip empty collections
            if ak.num(veto_obj, axis=0).compute().sum() == 0:
                continue

            # Compute ΔR matrix and materialize it
            drs = jets_before_veto.metric_table(veto_obj).compute()

            # Take the min ΔR for each jet
            min_dr = ak.min(drs, axis=1)

            # Drop None entries
            min_dr = min_dr[~ak.is_none(min_dr)]

            # Now flatten and convert to numpy
            min_dr_np = ak.to_numpy(ak.flatten(min_dr, axis=None))

            # Plot
            plt.hist(min_dr_np, bins=50, range=(0, 1.0), histtype='step', label=label)


        plt.axvline(0.4, color='red', linestyle='--', label="ΔR = 0.4 Veto Threshold")
        plt.xlabel("Minimum ΔR(jet, object)")
        plt.ylabel("Jet count")
        plt.legend()
        plt.title("Minimum ΔR Between Jets and Veto Objects (Before Veto)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{sample_name}_DeltaR_VetoSources.pdf")
        plt.close()
        '''
        
        # Define each veto separately
        #jets_no_photon   = jets_before_veto[delta_r_mask(jets_before_veto, cut_filtered_events.Photon,   0.4)]
        #jets_no_electron = jets_before_veto[delta_r_mask(jets_before_veto, cut_filtered_events.Electron, 0.4)]
        #jets_no_muon     = jets_before_veto[delta_r_mask(jets_before_veto, cut_filtered_events.Muon,     0.4)]
        #jets_no_dismuon  = jets_before_veto[delta_r_mask(jets_before_veto, cut_filtered_events.DisMuon,  0.4)]
        '''
        taus_after_veto = jets_no_muon.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        tau_parents = taus_after_veto.parent
        tau_children = tau_parents.distinctChildren

        # Exclude neutrinos (pdgId = 12, 14, 16) → only visible decay products
        visible_tau_children = tau_children[
            ~(abs(tau_children.pdgId) == 12)
            & ~(abs(tau_children.pdgId) == 14)
            & ~(abs(tau_children.pdgId) == 16)
        ]

        # Flatten and compute if needed (for Dask arrays)
        pdg_ids = ak.to_numpy(ak.flatten(abs(visible_tau_children.pdgId).compute(), axis=None))
        pts = ak.to_numpy(ak.flatten(visible_tau_children.pt.compute(), axis=None))

        # Plot PDG ID histogram
        plt.hist(pdg_ids, bins=range(0, 400), histtype='step')
        plt.xlabel("PDG ID of Gen Tau Children (Excluding Neutrinos)")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Visible Tau Decay Products")
        plt.grid(True)
        plt.savefig(f"{sample_name}_VisibleTauDecayPDGID.pdf")
        plt.close()
        '''
        '''
        plt.hist(ak.to_numpy(ak.flatten(jets_before_veto.disTauTag_score1.compute())), bins=10, range=(0.9, 1), histtype='step', label='Before Veto')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_photon.disTauTag_score1.compute())), bins=10, range=(0.9, 1), histtype='step', label='No Photon')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_electron.disTauTag_score1.compute())), bins=10, range=(0.9, 1), histtype='step', label='No Electron')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_muon.disTauTag_score1.compute())), bins=10, range=(0.9, 1), histtype='step', label='No Muon')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_dismuon.disTauTag_score1.compute())), bins=10, range=(0.9, 1), histtype='step', label='No DisMuon')

        plt.xlabel("Jet disTauTag Score")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet disTauTag Score Before/After Vetoes")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{sample_name}_JetScore_VetoComparison.pdf")
        plt.close()
        
        plt.hist(ak.to_numpy(ak.flatten(jets_before_veto.pt.compute())), bins=60, range=(0,750), histtype='step', label='Before Veto')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_photon.pt.compute())), bins=60, range=(0,750), histtype='step', label='No Photon')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_electron.pt.compute())), bins=60, range=(0,750), histtype='step', label='No Electron')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_muon.pt.compute())), bins=60, range=(0,750), histtype='step', label='No Muon')
        plt.hist(ak.to_numpy(ak.flatten(jets_no_dismuon.pt.compute())), bins=60, range=(0,750), histtype='step', label='No DisMuon')

        plt.xlabel("Jet $p_T$ [GeV]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet $p_T$ Before/After Vetoes")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{sample_name}_JetPt_VetoComparison.pdf")
        plt.close()
        
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
    '''




