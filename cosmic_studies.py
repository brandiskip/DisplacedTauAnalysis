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

def save_plot(h, var_name, sample_name, PREFIX, OUTPUT_DIR):
    """Plots a histogram and saves it with proper cosmic naming."""
    fig, ax = plt.subplots()
    # The variable is now passed in as an argument
    h.plot1d(ax=ax, label=f"Gen Muons from {sample_name}")
    ax.set_title(f"Gen Muon Distribution: {var_name}")
    ax.legend()
    
    # Create and save the filename
    filename = f"{PREFIX}{var_name}_hist.pdf"
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved plot for {var_name} to: {outpath}")

filenames = {
    'Cosmic'  : 'root://cmseos.fnal.gov///store/group/lpcdisptau//displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v14/LooseMuCosmic_2024/*.root',
}

PFNanoAODSchema.mixins["DisMuon"] = "Muon"
samples = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()

OUTPUT_DIR = "cosmic_muon_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREFIX_GEN = "cosmics_genmuon_"
PREFIX_RECO = "cosmics_recomuon_"
PLOTS_TO_SAVE = {}

if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        gpart = events.GenPart
        events['GenMuon'] = gpart[(abs(gpart.pdgId) == 13) & (gpart.hasFlags("isLastCopy"))] 

        # build 4-vector for muons and store momentum
        GenMuon_vec = ak.zip(
            {
                "pt":  events.GenMuon.pt,
                "eta": events.GenMuon.eta,
                "phi": events.GenMuon.phi,
                "mass": events.GenMuon.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=coffea.nanoevents.methods.vector.behavior,
        )
        events["GenMuon"] = ak.with_field(events.GenMuon, GenMuon_vec.p, where="p")

        GenMuon_d0 = abs((events.GenMuon.vy - events.GenVtx.y) * np.cos(events.GenMuon.phi) - \
              (events.GenMuon.vx - events.GenVtx.x) * np.sin(events.GenMuon.phi))
        events['GenMuon'] = ak.with_field(events.GenMuon, GenMuon_d0, where="d0")

        RecoMuon_vec = ak.zip(
            {
                "pt":  events.Muon.pt,
                "eta": events.Muon.eta,
                "phi": events.Muon.phi,
                "mass": events.Muon.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=coffea.nanoevents.methods.vector.behavior,
        )
        events["Muon"] = ak.with_field(events.Muon, RecoMuon_vec.p, where="p")

        '''
        RecoMuon_d0 = abs((events.Muon.vy - events.GenVtx.y) * np.cos(events.Muon.phi) - \
            (events.Muon.vx - events.GenVtx.x) * np.sin(events.Muon.phi))
        events['Muon'] = ak.with_field(events.Muon, RecoMuon_d0, where="d0")
        '''

        # --- Gen Muon Histogram Definitions ---
        pt_axis_g = axis.Variable(np.arange(0, 500, 20), name="pt", label="Gen Muon $p_T$ [GeV]")
        hist_pt_g = Hist(pt_axis_g)
        eta_axis_g = axis.Variable(np.arange(-2.5, 2.5, 0.1), name="eta", label="Gen Muon $\eta$")
        hist_eta_g = Hist(eta_axis_g)
        phi_axis_g = axis.Variable(np.arange(-np.pi, np.pi, 0.1), name="phi", label="Gen Muon $\phi$ [rad]")
        hist_phi_g = Hist(phi_axis_g)
        d0_axis_g = axis.Variable(np.arange(0, 100, 2), name="d0", label="Gen Muon $|d_0|$ [cm]")
        hist_d0_g = Hist(d0_axis_g)

        # --- Reco Muon Histogram Definitions ---
        count_bins = np.arange(0, 10, 1) # Muon count up to 9
        count_axis = axis.Variable(count_bins, name="n_reco_muon", label="Number of Reconstructed Muons")
        hist_count_r = Hist(count_axis)

        pt_axis_r = pt_axis_g # Reuse the same bins for comparison
        hist_pt_r = Hist(pt_axis_r)
        eta_axis_r = eta_axis_g
        hist_eta_r = Hist(eta_axis_r)
        phi_axis_r = phi_axis_g
        hist_phi_r = Hist(phi_axis_r)
        #d0_axis_r = d0_axis_g
        #hist_d0_r = Hist(d0_axis_r)

        # ====================================================================
        # SINGLE COMPUTATION AND FILLING
        # ====================================================================

        # Define the combined Gen data structure (for re-filling)
        gen_data_to_compute = ak.zip({
            "pt": ak.flatten(events.GenMuon.pt, axis=None),
            "eta": ak.flatten(events.GenMuon.eta, axis=None),
            "phi": ak.flatten(events.GenMuon.phi, axis=None),
            "d0": ak.flatten(events.GenMuon.d0, axis=None),
        })

        # Define the combined Reco data structure
        reco_data_to_compute = ak.zip({
            "count": ak.num(events.Muon, axis=1), # Count is calculated at the event level (axis=1)
            "pt": ak.flatten(events.Muon.pt, axis=None),
            "eta": ak.flatten(events.Muon.eta, axis=None),
            "phi": ak.flatten(events.Muon.phi, axis=None),
            #"d0": ak.flatten(events.Muon.d0, axis=None),
        })

        # Combine both and compute EVERYTHING at once
        all_data_to_compute = ak.zip({
            "gen": gen_data_to_compute,
            "reco": reco_data_to_compute,
        })
        computed_data = all_data_to_compute.compute()
        
        # Fill Gen Muon Histograms (Re-filling ensures the run is complete)
        hist_pt_g.fill(computed_data["gen"]["pt"])
        hist_eta_g.fill(computed_data["gen"]["eta"])
        hist_phi_g.fill(computed_data["gen"]["phi"])
        hist_d0_g.fill(computed_data["gen"]["d0"])

        # Fill Reco Muon Histograms
        hist_count_r.fill(computed_data["reco"]["count"]) # Count is already flat
        hist_pt_r.fill(computed_data["reco"]["pt"])
        hist_eta_r.fill(computed_data["reco"]["eta"])
        hist_phi_r.fill(computed_data["reco"]["phi"])
        #hist_d0_r.fill(computed_data["reco"]["d0"])

        # ====================================================================
        # PLOTTING AND SAVING
        # ====================================================================

        print("--- Saving Gen Muon Histograms ---")
        save_plot(hist_pt_g, "GenMuon_pt", sample_name, PREFIX_GEN, OUTPUT_DIR)
        save_plot(hist_eta_g, "GenMuon_eta", sample_name, PREFIX_GEN, OUTPUT_DIR)
        save_plot(hist_phi_g, "GenMuon_phi", sample_name, PREFIX_GEN, OUTPUT_DIR)
        save_plot(hist_d0_g, "GenMuon_d0", sample_name, PREFIX_GEN, OUTPUT_DIR)

        print("--- Saving Reco Muon Histograms ---")
        save_plot(hist_count_r, "RecoMuon_count", sample_name, PREFIX_RECO, OUTPUT_DIR)
        save_plot(hist_pt_r, "RecoMuon_pt", sample_name, PREFIX_RECO, OUTPUT_DIR)
        save_plot(hist_eta_r, "RecoMuon_eta", sample_name, PREFIX_RECO, OUTPUT_DIR)
        save_plot(hist_phi_r, "RecoMuon_phi", sample_name, PREFIX_RECO, OUTPUT_DIR)
        #save_plot(hist_d0_r, "RecoMuon_d0", sample_name, PREFIX_RECO, OUTPUT_DIR)
        print("-----------------------------------")