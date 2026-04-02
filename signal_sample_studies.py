import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from hist import Hist, axis
from coffea import processor
import coffea.nanoevents.methods.vector as vector
from coffea.nanoevents import PFNanoAODSchema

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_Lxy(tau_array):
    """Calculates the transverse decay length (Lxy) for an array of taus."""
    vx = tau_array.parent.vx - tau_array.parent.distinctParent.vx
    vy = tau_array.parent.vy - tau_array.parent.distinctParent.vy
    return np.sqrt(vx**2 + vy**2)

def save_categorical_overlay(h_obj, xlabel, title, outpath):
    """Plots a 1D histogram with a categorical axis as overlaid lines."""
    if np.sum(h_obj.values()) <= 0:
        print(f"Skipping {outpath}: No events passed conditions.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for label in h_obj.axes[0]:
        h_slice = h_obj[{h_obj.axes[0].name: label}]
        h_slice.plot1d(ax=ax, label=label)
        
    ax.legend(title="Category")
    ax.set_ylabel("Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved plot to: {outpath}")

def plot_efficiency_overlay(out, dataset_name, num_key, den_key, xlabel, title, outpath, ylim=(0.0, 1.05)):
    """
    Helper function to extract histograms from the processor output,
    loop over the jet variants, plot the efficiencies, and save the figure.
    """
    plt.figure(figsize=(8, 6))
    h_den = out[den_key][{"dataset": dataset_name}]
    
    if np.sum(h_den.values()) <= 0:
        print(f"Skipping {outpath}: Denominator histogram is empty.")
        plt.close()
        return

    for label in out[num_key].axes["variant"]:
        h_num = out[num_key][{"dataset": dataset_name, "variant": label}]
        plot_efficiency(h_num, h_den, label=label)

    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved plot to: {outpath}")

def plot_efficiency(h_num, h_den, label):
    """
    Calculates and plots the efficiency (numerator / denominator)
    with simple binomial error bars for 1D histograms.
    """
    num = h_num.values()
    den = h_den.values()
    
    edges = h_num.axes[0].edges
    centers = (edges[:-1] + edges[1:]) / 2
    
    eff = np.zeros_like(num, dtype=float)
    valid = den > 0
    eff[valid] = num[valid] / den[valid]
    
    err = np.zeros_like(num, dtype=float)
    err[valid] = np.sqrt(eff[valid] * (1.0 - eff[valid]) / den[valid])
    
    plt.errorbar(centers, eff, yerr=err, fmt='o', markersize=4, label=label)

# ==============================================================================
# Processor Class
# ==============================================================================
class SignalJetProcessor(processor.ProcessorABC):
    def __init__(self):
        
        cat_axis = axis.StrCategory([], name="cat", label="Jet Category", growth=True)
        muEF_axis = axis.Regular(50, 0, 0.8, name="val", label="muEF")

        dataset_axis = axis.StrCategory([], name="dataset", label="Dataset", growth=True)
        variant_axis = axis.StrCategory([], name="variant", label="Jet Cut Variant", growth=True)
        
        eta_bins = np.arange(-2.1, 2.1 + 1e-6, 0.1)
        eta_axis = axis.Variable(eta_bins, name="eta", label=r"$\eta$")
        
        Lxy_bins = np.arange(0, 100 + 1e-6, 5.0)
        Lxy_axis = axis.Variable(Lxy_bins, name="Lxy", label=r"$L_{xy}$ (cm)")
        
        pt_bins_low = np.arange(20, 101, 20)
        pt_bins_med = np.arange(100, 400, 30)
        pt_bins_high = np.arange(400, 600, 40)
        pt_bins_higher = np.arange(600, 1000, 50)
        pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))
        pt_axis = axis.Variable(pt_bins_eff, name="pt", label=r"$p_T$ (GeV)")

        d0_bins = np.arange(0, 20 + 1e-6, 0.5)
        d0_axis = axis.Variable(d0_bins, name="d0", label=r"$|d_{0}|$ (cm)")

        self.output = {
            "muEF": Hist(cat_axis, muEF_axis),
            "total_gvt_events": 0,
            "matched_highest": 0,
            "matched_second": 0,
            
            # Common Denominators
            "den_eta": Hist(dataset_axis, eta_axis),
            "den_Lxy": Hist(dataset_axis, Lxy_axis),
            "den_pt": Hist(dataset_axis, pt_axis),
            "den_d0": Hist(dataset_axis, d0_axis),  
            
            # Study 1: Highest Score Jets (4 plots)
            "num_eta_highest": Hist(dataset_axis, variant_axis, eta_axis),
            "num_Lxy_highest": Hist(dataset_axis, variant_axis, Lxy_axis),
            "num_pt_highest": Hist(dataset_axis, variant_axis, pt_axis),
            "num_d0_highest": Hist(dataset_axis, variant_axis, d0_axis),  

            # Study 2: Individual Cuts (3 plots)
            "num_eta_indiv": Hist(dataset_axis, variant_axis, eta_axis),
            "num_Lxy_indiv": Hist(dataset_axis, variant_axis, Lxy_axis),
            "num_pt_indiv": Hist(dataset_axis, variant_axis, pt_axis),
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
        
        genvistau_Lxy = get_Lxy(events.GenVisTau)
        
        events['GenVisStauTaus'] = events.GenVisTau[
            (abs(events.GenVisTau.parent.pdgId) == 15) & 
            (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & 
            (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & 
            (events.GenVisTau.parent.hasFlags("fromHardProcess")) & 
            (genvistau_Lxy < 100.0) & 
            (events.GenVisTau.pt > 20) & 
            (abs(events.GenVisTau.eta) < 2.4)
        ]

        d0 = abs((events.GenVisStauTaus.parent.vy - events.GenVtx.y) * np.cos(events.GenVisStauTaus.parent.phi) - \
               (events.GenVisStauTaus.parent.vx - events.GenVtx.x) * np.sin(events.GenVisStauTaus.parent.phi))
        events['GenVisStauTaus'] = ak.with_field(events.GenVisStauTaus, d0, where="d0")

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

        cut_filtered_events = events

        # -------------------------------------------------------------
        # Fill Common Denominator Histograms
        # -------------------------------------------------------------
        den_Lxy = get_Lxy(cut_filtered_events.GenVisStauTaus)

        self.output["den_eta"].fill(dataset=dataset, eta=ak.flatten(cut_filtered_events.GenVisStauTaus.eta, axis=None))
        self.output["den_pt"].fill(dataset=dataset, pt=ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None))
        self.output["den_Lxy"].fill(dataset=dataset, Lxy=ak.flatten(den_Lxy, axis=None))
        self.output["den_d0"].fill(dataset=dataset, d0=ak.flatten(cut_filtered_events.GenVisStauTaus.d0, axis=None))

        # =============================================================
        # STUDY 2: INDIVIDUAL CUTS (All Jets - 3 Plots)
        # =============================================================
        baseline_jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
        baseline_jets_neHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.neHEF < 0.99)]
        baseline_jets_neEmEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.neEmEF < 0.9)]
        baseline_jets_ch_ne_Mult = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & ((cut_filtered_events.Jet.chMultiplicity + cut_filtered_events.Jet.neMultiplicity) > 1)]
        baseline_jets_chHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.chHEF > 0.01)]
        baseline_jets_chMultiplicity = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.chMultiplicity > 0)]

        jets_variants_indiv = {
            "baseline (pt>20,|eta|<2.4)":            baseline_jets,
            "baseline + neHEF < 0.99":               baseline_jets_neHEF,
            "baseline + neEmEF < 0.90":              baseline_jets_neEmEF,
            "baseline + ch+ne mult > 1":             baseline_jets_ch_ne_Mult,
            "baseline + chHEF > 0.01":               baseline_jets_chHEF,
            "baseline + chMultiplicity > 0":         baseline_jets_chMultiplicity,
        }

        for label, jcol in jets_variants_indiv.items():
            matched_gen = jcol.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
            matched_gen = ak.drop_none(matched_gen)

            m_Lxy = get_Lxy(matched_gen)

            self.output["num_eta_indiv"].fill(dataset=dataset, variant=label, eta=ak.flatten(matched_gen.eta, axis=None))
            self.output["num_pt_indiv"].fill(dataset=dataset, variant=label, pt=ak.flatten(matched_gen.pt, axis=None))
            self.output["num_Lxy_indiv"].fill(dataset=dataset, variant=label, Lxy=ak.flatten(m_Lxy, axis=None))


        # =============================================================
        # STUDY 1: HIGHEST SCORE JETS (isTight variants - 4 Plots)
        # =============================================================
        jets = cut_filtered_events.Jet
        base_mask = (abs(jets.eta) < 2.4) & (jets.pt > 20)
        
        mask_isTight = base_mask & \
                       (jets.neHEF < 0.99) & \
                       (jets.neEmEF < 0.9) & \
                       ((jets.chMultiplicity + jets.neMultiplicity) > 1) & \
                       (jets.chMultiplicity > 0)
                       
        mask_isTightLV = mask_isTight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8)

        raw_jets_variants = {
            "Baseline (pt>20, |eta|<2.4)": jets[base_mask],
            "isTight (no chHEF)":           jets[mask_isTight],
            "isTight (with chHEF > 0.01)":  jets[mask_isTight & (jets.chHEF > 0.01)],
            "isTightLepVeto (no chHEF)":    jets[mask_isTightLV],
            "isTightLepVeto (with chHEF > 0.01)": jets[mask_isTightLV & (jets.chHEF > 0.01)],
        }

        # Restrict each variant to ONLY the Leading Score Jet
        jets_variants_highest = {}
        for label, jcol in raw_jets_variants.items():
            sorted_by_score = jcol[ak.argsort(jcol.disTauTag_score1, ascending=False)]
            jets_variants_highest[label] = ak.singletons(ak.firsts(sorted_by_score))

        for label, jcol in jets_variants_highest.items():
            matched_gen = jcol.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
            matched_gen = ak.drop_none(matched_gen)

            m_Lxy = get_Lxy(matched_gen)

            self.output["num_eta_highest"].fill(dataset=dataset, variant=label, eta=ak.flatten(matched_gen.eta, axis=None))
            self.output["num_pt_highest"].fill(dataset=dataset, variant=label, pt=ak.flatten(matched_gen.pt, axis=None))
            self.output["num_Lxy_highest"].fill(dataset=dataset, variant=label, Lxy=ak.flatten(m_Lxy, axis=None))
            self.output["num_d0_highest"].fill(dataset=dataset, variant=label, d0=ak.flatten(matched_gen.d0, axis=None))


        # =============================================================
        # muEF Matched/Unmatched Logic
        # =============================================================
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

        sorted_by_score_2j = jets_2j[ak.argsort(jets_2j.disTauTag_score1, ascending=False)]
        highest_score_jets = sorted_by_score_2j[:, 0]
        second_highest_score_jets = sorted_by_score_2j[:, 1]
        
        single_gvt = cut_filtered_events_2j.GenVisStauTaus[:, 0]

        dr_highest = highest_score_jets.delta_r(single_gvt)
        dr_second = second_highest_score_jets.delta_r(single_gvt)

        is_highest_matched = dr_highest < 0.4
        is_second_matched = dr_second < 0.4

        self.output["total_gvt_events"] += len(single_gvt)
        self.output["matched_highest"] += ak.sum(is_highest_matched)
        self.output["matched_second"] += ak.sum(is_second_matched)

        evt_keep = (~is_highest_matched) & is_second_matched

        highest_unmatched_jets = highest_score_jets[evt_keep]
        second_matched_jets = second_highest_score_jets[evt_keep]

        self.output["muEF"].fill(cat="Highest Score Jet (Unmatched)", val=highest_unmatched_jets.muEF)
        self.output["muEF"].fill(cat="2nd Highest Score Jet (Matched)", val=second_matched_jets.muEF)

        return self.output

    def postprocess(self, accumulator):
        return accumulator

# ==============================================================================
# Execution Block
# ==============================================================================
if __name__ == '__main__':
    signal_pkl = "samples/Signal/Stau_300_100mm_preprocessed.pkl"
    print(f"Loading preprocessed data from {signal_pkl}...")
    with open(signal_pkl, "rb") as f:
        runnable = pickle.load(f)

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

    # -------------------------------------------------------------
    # Plot Configuration & Loop
    # -------------------------------------------------------------
    OUTPUT_DIR = "signal_jet_comparisons"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sample_name = "Stau_300_100mm"
    
    try:
        d_name = list(out["den_eta"].axes["dataset"])[0]
    except IndexError:
        d_name = "Unknown"

    # Define the 7 plots explicitly
    plots_to_make = [
        {"num_key": "num_eta_highest", "den_key": "den_eta", "xlabel": r"GenVisTau $\eta$", "title": "Matching Efficiency (Highest Score Jet) vs $\eta$", "filename": "eff_vs_eta_highest_score"},
        {"num_key": "num_Lxy_highest", "den_key": "den_Lxy", "xlabel": r"GenVisTau $L_{xy}$ [cm]", "title": "Matching Efficiency (Highest Score Jet) vs $L_{xy}$", "filename": "eff_vs_Lxy_highest_score"},
        {"num_key": "num_pt_highest",  "den_key": "den_pt",  "xlabel": r"GenVisTau $p_T$ [GeV]", "title": "Matching Efficiency (Highest Score Jet) vs $p_T$", "filename": "eff_vs_pt_highest_score"},
        {"num_key": "num_d0_highest",  "den_key": "den_d0",  "xlabel": r"GenVisTau $|d_{0}|$ [cm]", "title": "Matching Efficiency (Highest Score Jet) vs $|d_{0}|$", "filename": "eff_vs_d0_highest_score"},
        
        {"num_key": "num_eta_indiv",   "den_key": "den_eta", "xlabel": r"GenVisTau $\eta$", "title": "Matching Efficiency (Individual Cuts) vs $\eta$", "filename": "eff_vs_eta_individual_cuts", "ylim": (0.4, 1.05)},
        {"num_key": "num_Lxy_indiv",   "den_key": "den_Lxy", "xlabel": r"GenVisTau $L_{xy}$ [cm]", "title": "Matching Efficiency (Individual Cuts) vs $L_{xy}$", "filename": "eff_vs_Lxy_individual_cuts", "ylim": (0.4, 1.05)},
        {"num_key": "num_pt_indiv",    "den_key": "den_pt",  "xlabel": r"GenVisTau $p_T$ [GeV]", "title": "Matching Efficiency (Individual Cuts) vs $p_T$", "filename": "eff_vs_pt_individual_cuts", "ylim": (0.4, 1.05)}]

    for p in plots_to_make:
        title = f"{p['title']} ({sample_name.replace('_', ' ')}) v18"
        outpath = os.path.join(OUTPUT_DIR, f"{p['filename']}_{sample_name}_v18.pdf")

        current_ylim = p.get("ylim", (0.0, 1.05))
        
        plot_efficiency_overlay(
            out=out,
            dataset_name=d_name,
            num_key=p["num_key"],
            den_key=p["den_key"],
            xlabel=p["xlabel"],
            title=title,
            outpath=outpath,
            ylim=current_ylim # Pass the dynamic parameter
        )

    # Generate the muEF Plot
    muEF_title = f"Jet muEF Comparison ({sample_name.replace('_', ' ')}) v18"
    muEF_outpath = os.path.join(OUTPUT_DIR, f"Stau_300_100mm_muEF_Matched_vs_Unmatched_v18.pdf")
    save_categorical_overlay(out["muEF"], "muEF", muEF_title, muEF_outpath)

    print("Done!")