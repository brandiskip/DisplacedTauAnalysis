import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from hist import Hist, axis
from coffea import processor
import coffea.nanoevents.methods.vector as vector
from coffea.nanoevents import PFNanoAODSchema

class SignalJetProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            "muEF": Hist(
                axis.StrCategory([], name="cat", label="Jet Category", growth=True),
                axis.Regular(50, 0, 0.8, name="val", label="muEF")
            ),
            "total_gvt_events": 0,
            "matched_highest": 0,
            "matched_second": 0
        }

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")

        # Jet dxy variables
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.fill_none(abs(ak.firsts(events.Jet.constituents.pf[charged_sel][ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis=2)), -999)
        events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")
        
        dxy_err = ak.fill_none(abs(ak.firsts(events.Jet.constituents.pf[charged_sel][ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0Err, axis=2)), -999)
        events['Jet'] = ak.with_field(events.Jet, dxy_err, where="dxy_err")
        
        # Staus and Taus
        gpart = events.GenPart
        events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] 

        events['staus_taus'] = events.staus.distinctChildren[ 
            (abs(events.staus.distinctChildren.pdgId) == 15) & 
            (events.staus.distinctChildren.hasFlags("isLastCopy")) & 
            (events.staus.distinctChildren.hasFlags("fromHardProcess")) 
        ]
        
        vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.distinctParent.vx
        vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.distinctParent.vy
        genvistau_Lxy = np.sqrt(vx**2 + vy**2)
        
        events['GenVisStauTaus'] = events.GenVisTau[
            (abs(events.GenVisTau.parent.pdgId) == 15) & 
            (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & 
            (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & 
            (events.GenVisTau.parent.hasFlags("fromHardProcess")) & 
            (genvistau_Lxy < 100.0) & 
            (events.GenVisTau.pt > 20) & 
            (abs(events.GenVisTau.eta) < 2.4)
        ]

        # GenMuon logic
        events['GenMuon'] = gpart[(abs(gpart.pdgId) == 13) & (gpart.hasFlags("isLastCopy"))] 
        events['GenMuon'] = events.GenMuon[
            (events.GenMuon.pt > 20) & 
            (abs(events.GenMuon.eta) < 2.4) & 
            (events.GenMuon.distinctParent.distinctParent.pdgId == 1000015)
        ]

        # Event level masks
        mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenMuon) == 1)
        events = events[mask]
        
        if len(events) == 0:
            return self.output

        # Sort staus_taus
        events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis=2)
        staus_taus = events['staus_taus']

        # Tau decay masks
        mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        mask_tauh = ~mask_taul

        one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
        one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

        filtered_events = events[one_tauh_evt & one_taul_evt]
        
        tau_selections = (filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus_taus.eta) < 2.4)
        num_taus = ak.sum(tau_selections, axis=-1)
        num_tau_mask = num_taus > 1
        cut_filtered_events = filtered_events[num_tau_mask]

        # Jet selections
        jets = cut_filtered_events.Jet[
            (abs(cut_filtered_events.Jet.eta) < 2.4) & 
            (cut_filtered_events.Jet.pt > 20) & 
            (cut_filtered_events.Jet.neHEF < 0.99) & 
            (cut_filtered_events.Jet.neEmEF < 0.9) & 
            ((cut_filtered_events.Jet.chMultiplicity + cut_filtered_events.Jet.neMultiplicity) > 1) & 
            (cut_filtered_events.Jet.chMultiplicity > 0) & 
            (cut_filtered_events.Jet.muEF < 0.8) & 
            (cut_filtered_events.Jet.chEmEF < 0.8)
        ]
        
        has_2_jets = ak.num(jets) == 2
        jets_2j = jets[has_2_jets]
        cut_filtered_events_2j = cut_filtered_events[has_2_jets]
        
        if len(jets_2j) == 0:
            return self.output

        # Sorting by disTauTag_score1
        sorted_by_score_2j = jets_2j[ak.argsort(jets_2j.disTauTag_score1, ascending=False)]
        highest_score_jets = sorted_by_score_2j[:, 0]
        second_highest_score_jets = sorted_by_score_2j[:, 1]
        
        # We know there is exactly 1 GenVisStauTau due to the mask earlier
        single_gvt = cut_filtered_events_2j.GenVisStauTaus[:, 0]

        # Calculate Delta R
        dr_highest = highest_score_jets.delta_r(single_gvt)
        dr_second = second_highest_score_jets.delta_r(single_gvt)

        # Matched masks (dR < 0.4)
        is_highest_matched = dr_highest < 0.4
        is_second_matched = dr_second < 0.4

        # Track stats for the percentage calculation
        self.output["total_gvt_events"] += len(single_gvt)
        self.output["matched_highest"] += ak.sum(is_highest_matched)
        self.output["matched_second"] += ak.sum(is_second_matched)

        # Plot condition: highest is NOT matched AND second IS matched
        evt_keep = (~is_highest_matched) & is_second_matched

        highest_unmatched_jets = highest_score_jets[evt_keep]
        second_matched_jets = second_highest_score_jets[evt_keep]

        # Fill Histograms
        self.output["muEF"].fill(cat="Highest Score Jet (Unmatched)", val=highest_unmatched_jets.muEF)
        self.output["muEF"].fill(cat="2nd Highest Score Jet (Matched)", val=second_matched_jets.muEF)

        return self.output

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # 1. Load the preprocessed signal pickle
    signal_pkl = "samples/Signal/Stau_300_100mm_preprocessed.pkl"
    print(f"Loading preprocessed data from {signal_pkl}...")
    with open(signal_pkl, "rb") as f:
        runnable = pickle.load(f)

    # 2. Run the Processor
    print("Starting Processor...")
    executor = processor.FuturesExecutor(workers=8)
    runner = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
    )

    out = runner(
        runnable,
        treename="Events",
        processor_instance=SignalJetProcessor(),
    )

    # 3. Calculate and Print Percentages
    total_events = out["total_gvt_events"]
    matched_high = out["matched_highest"]
    matched_sec = out["matched_second"]

    if total_events > 0:
        pct_highest = (matched_high / total_events) * 100
        pct_second = (matched_sec / total_events) * 100
        print("\n=== Matching Statistics ===")
        print(f"Total Events with exactly 1 GenVisTau and 2 clean jets: {total_events}")
        print(f"GenVisTau matched to Highest Score Jet: {matched_high} ({pct_highest:.2f}%)")
        print(f"GenVisTau matched to 2nd Highest Score Jet: {matched_sec} ({pct_second:.2f}%)")
        print("===========================\n")

    # 4. Make the Plot
    OUTPUT_DIR = "signal_jet_comparisons"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    h_muEF = out["muEF"]
    
    if np.sum(h_muEF.values()) > 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Plot raw event counts as an overlay
        for label in h_muEF.axes["cat"]:
            h_slice = h_muEF[{"cat": label}]
            h_slice.plot1d(ax=ax, label=label)
            
        ax.legend(title="Jet Category")
        ax.set_ylabel("Events")
        ax.set_xlabel("muEF")
        
        # Clean and clear title for analysis presentation
        ax.set_title("Jet muEF Comparison (Stau 300 GeV, 100 mm) v18")
        
        outpath = os.path.join(OUTPUT_DIR, "Stau_300_100mm_muEF_Matched_vs_Unmatched.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f"Saved plot to: {outpath}")
    else:
        print("No events passed the specific matched/unmatched condition. Plot skipped.")