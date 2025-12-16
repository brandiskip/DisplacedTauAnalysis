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

def save_plot(h, var_name, sample_name, PREFIX, OUTPUT_DIR, is_reco_muon=False):
    """Plots a histogram and saves it with proper cosmic naming."""
    fig, ax = plt.subplots()
    
    # Determine the particle type label
    if var_name.startswith("GenMuon"):
        label = f"Gen Muons from {sample_name}"
        particle_type = "Gen"
    elif var_name.startswith("DisMuon"):
        label = f"Displaced Muons from {sample_name}"
        particle_type = "Displaced"
    else:
        label = f"Reco Muons from {sample_name}"
        particle_type = "Reco"
        
    h.plot1d(ax=ax, label=label)

    # Ensure title reflects the correct particle type and variable
    title_var = var_name.split('_')[-1].upper()
    
    if var_name == "RecoMuon_count":
        ax.set_title(f"Reconstructed Muon Multiplicity")
        ax.set_xlabel(ax.get_xlabel().replace("count", "Count"))
    else:
        ax.set_title(f"{particle_type} Muon Distribution: {title_var}")
    
    ax.legend()
    
    # Create and save the filename
    filename = f"{PREFIX}{var_name}.pdf"
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved plot for {var_name} to: {outpath}")

def save_overlay_plot(h_lead, h_subl, var_name, sample_name, PREFIX, OUTPUT_DIR):
    """Plots leading and subleading histograms on the same canvas."""
    fig, ax = plt.subplots()
    
    # Plotting both histograms with distinct labels
    h_lead.plot1d(ax=ax, label="Leading Muon") 
    h_subl.plot1d(ax=ax, label="Subleading Muon")
    
    # Set Title and Labels
    title_var = var_name.split('_')[-1] # Extracts 'isGlobal' or 'isStandalone'
    ax.set_title(f"Leading vs Subleading Muon: {title_var}")
    ax.set_xlabel(h_lead.axes[0].label) # Use the categorical axis label
    ax.legend()
    
    # Create and save the filename
    filename = f"{PREFIX}{var_name}.pdf"
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved overlay plot for {var_name} to: {outpath}")

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
PREFIX_DISMUON = "cosmics_dismuon_"
PLOTS_TO_SAVE = {}

if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        '''
        mask = (ak.num(events.Muon) >= 2)
        events = events[mask]
        '''
        '''
        mask = (ak.num(events.DisMuon) >= 2)
        events = events[mask]
        '''

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
        
        px = RecoMuon_vec.px
        py = RecoMuon_vec.py
        pz = RecoMuon_vec.pz

        '''
        sorted_muons = events.Muon[ak.argsort(events.Muon.p, ascending=False)]
        leading_muon_reco = sorted_muons[:, 0]
        subleading_muon_reco = sorted_muons[:, 1]

        mask_same_charge = (leading_muon_reco.charge * subleading_muon_reco.charge) > 0
        events = events[mask_same_charge]

        flag_axis = hist.axis.StrCategory(["True", "False"], name="flag", label="Muon Reconstruction Flag Status")

        # isGlobal Muon Hists (Leading and Subleading)
        hist_isGlobal_lead = Hist(flag_axis, label="isGlobal")
        hist_isGlobal_subl = Hist(flag_axis, label="isGlobal")

        # isStandalone Muon Hists (Leading and Subleading)
        hist_isStandalone_lead = Hist(flag_axis, label="isStandalone")
        hist_isStandalone_subl = Hist(flag_axis, label="isStandalone")

        # isGlobal Muon (Reco)
        isGlobal_lead_data = ak.to_numpy(leading_muon_reco.isGlobal.compute())
        isGlobal_subl_data = ak.to_numpy(subleading_muon_reco.isGlobal.compute())

        isStandalone_lead_data = ak.to_numpy(leading_muon_reco.isStandalone.compute())
        isStandalone_subl_data = ak.to_numpy(subleading_muon_reco.isStandalone.compute())

        hist_isGlobal_lead.fill(np.where(isGlobal_lead_data, "True", "False"))
        hist_isGlobal_subl.fill(np.where(isGlobal_subl_data, "True", "False"))

        hist_isStandalone_lead.fill(np.where(isStandalone_lead_data, "True", "False"))
        hist_isStandalone_subl.fill(np.where(isStandalone_subl_data, "True", "False"))

        print("--- Saving Reco Muon Overlay Plots ---")
        save_overlay_plot(
            hist_isGlobal_lead, hist_isGlobal_subl,
            "RecoMuon_isGlobal_overlay", sample_name, PREFIX_RECO, OUTPUT_DIR
        )
        save_overlay_plot(
            hist_isStandalone_lead, hist_isStandalone_subl,
            "RecoMuon_isStandalone_overlay", sample_name, PREFIX_RECO, OUTPUT_DIR
        )
        print("-----------------------------------")
        '''

        '''
        dot_product = leading_muon_reco.px * subleading_muon_reco.px + \
                      leading_muon_reco.py * subleading_muon_reco.py + \
                      leading_muon_reco.pz * subleading_muon_reco.pz

        den = leading_muon_reco.p * subleading_muon_reco.p
        cosA = ak.where(den != 0, dot_product / den, -1000.0)
        valid_mask = (cosA >= -1.0) & (cosA <= 1.0)
        angle_radians = ak.where(valid_mask, np.arccos(cosA), -1000.0)
        alpha_degrees = ak.where(valid_mask, np.degrees(angle_radians), -1000.0)

        alpha_bins = np.arange(0, 185, 5) # 5 degree bins from 0 to 180
        alpha_axis = axis.Variable(alpha_bins, name="alpha_degrees", label="Angle $\\alpha$ between Leading Muons [degrees]")
        hist_alpha_r = Hist(alpha_axis)
        flat_reco_alpha = alpha_degrees.compute()
        hist_alpha_r.fill(flat_reco_alpha)
        save_plot(hist_alpha_r, "RecoMuon_alpha", sample_name, PREFIX_RECO, OUTPUT_DIR, is_reco_muon=True)
        '''
        DisMuon_vec = ak.zip(
            {
                "pt":  events.DisMuon.pt,
                "eta": events.DisMuon.eta,
                "phi": events.DisMuon.phi,
                "mass": events.DisMuon.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=coffea.nanoevents.methods.vector.behavior,
        )
        events["DisMuon"] = ak.with_field(events.DisMuon, DisMuon_vec.p, where="p")
        
        px = DisMuon_vec.px
        py = DisMuon_vec.py
        pz = DisMuon_vec.pz

        '''
        hist_count_d = Hist(axis.Variable(np.arange(0, 10, 1), name="n_dis_muon", label="Number of Displaced Muons"))
        dis_count_data = ak.num(events.DisMuon, axis=1).compute()
        hist_count_d.fill(dis_count_data)
        save_plot(hist_count_d, "DisMuon_count", sample_name, PREFIX_DISMUON, OUTPUT_DIR)
        '''
        '''
        sorted_dis_muons = events.DisMuon[ak.argsort(events.DisMuon.p, ascending=False)]
        leading_dis_muon_reco = sorted_dis_muons[:, 0]
        subleading_dis_muon_reco = sorted_dis_muons[:, 1]

        mask_same_charge_dis = (leading_dis_muon_reco.charge * subleading_dis_muon_reco.charge) > 0
        events = events[mask_same_charge_dis]
        '''
        '''
        # --- DisMuon Reconstruction Flag Histograms ---
        flag_axis = hist.axis.StrCategory(["True", "False"], name="flag", label="Muon Reconstruction Flag Status")

        # isGlobal Muon Hists (Leading and Subleading)
        hist_isGlobal_dlead = Hist(flag_axis, label="isGlobal")
        hist_isGlobal_dsubl = Hist(flag_axis, label="isGlobal")

        # isStandalone Muon Hists (Leading and Subleading)
        hist_isStandalone_dlead = Hist(flag_axis, label="isStandalone")
        hist_isStandalone_dsubl = Hist(flag_axis, label="isStandalone")

        # isGlobal Muon (DisMuon)
        # NOTE: We use the already sorted leading/subleading DisMuons
        isGlobal_dlead_data = ak.to_numpy(leading_dis_muon_reco.isGlobal.compute())
        isGlobal_dsubl_data = ak.to_numpy(subleading_dis_muon_reco.isGlobal.compute())

        isStandalone_dlead_data = ak.to_numpy(leading_dis_muon_reco.isStandalone.compute())
        isStandalone_dsubl_data = ak.to_numpy(subleading_dis_muon_reco.isStandalone.compute())

        hist_isGlobal_dlead.fill(np.where(isGlobal_dlead_data, "True", "False"))
        hist_isGlobal_dsubl.fill(np.where(isGlobal_dsubl_data, "True", "False"))

        hist_isStandalone_dlead.fill(np.where(isStandalone_dlead_data, "True", "False"))
        hist_isStandalone_dsubl.fill(np.where(isStandalone_dsubl_data, "True", "False"))

        print("--- Saving DisMuon Overlay Plots ---")
        save_overlay_plot(
            hist_isGlobal_dlead, hist_isGlobal_dsubl,
            "DisMuon_isGlobal_overlay", sample_name, PREFIX_DISMUON, OUTPUT_DIR
        )
        save_overlay_plot(
            hist_isStandalone_dlead, hist_isStandalone_dsubl,
            "DisMuon_isStandalone_overlay", sample_name, PREFIX_DISMUON, OUTPUT_DIR
        )
        print("-----------------------------------")
        '''
        
        '''
        dot_product_dis = leading_dis_muon_reco.px * subleading_dis_muon_reco.px + \
                          leading_dis_muon_reco.py * subleading_dis_muon_reco.py + \
                          leading_dis_muon_reco.pz * subleading_dis_muon_reco.pz

        den_dis = leading_dis_muon_reco.p * subleading_dis_muon_reco.p

        cosA_dis = ak.where(den_dis != 0, dot_product_dis / den_dis, -1000.0)
        valid_mask_dis = (cosA_dis >= -1.0) & (cosA_dis <= 1.0)
        angle_radians_dis = ak.where(valid_mask_dis, np.arccos(cosA_dis), -1000.0)
        alpha_degrees_dis = ak.where(valid_mask_dis, np.degrees(angle_radians_dis), -1000.0)

        alpha_bins_dis = np.arange(0, 185, 5) # 5 degree bins from 0 to 180
        alpha_axis_dis = axis.Variable(alpha_bins_dis, name="alpha_degrees", label="Angle $\\alpha$ between Leading DisMuons [degrees]")
        hist_alpha_r_dis = Hist(alpha_axis_dis)
        flat_dis_alpha = alpha_degrees_dis.compute()
        hist_alpha_r_dis.fill(flat_dis_alpha)
        save_plot(hist_alpha_r_dis, "DisMuon_alpha", sample_name, PREFIX_DISMUON, OUTPUT_DIR)
        '''
        '''
        # ====================================================================
        # HISTOGRAM DEFINITION
        # ====================================================================
        pt_bins = np.arange(0, 500, 20)
        eta_bins = np.arange(-2.5, 2.5, 0.1)
        phi_bins = np.arange(-np.pi, np.pi, 0.1)
        d0_bins = np.arange(0, 100, 2)
        '''
        eta_bins = np.arange(-1, 1, 0.1)
        
        '''
        # Gen Muon Hists
        hist_pt_g = Hist(axis.Variable(pt_bins, name="pt", label="Gen Muon $p_T$ [GeV]"))
        hist_eta_g = Hist(axis.Variable(eta_bins, name="eta", label="Gen Muon $\eta$"))
        hist_phi_g = Hist(axis.Variable(phi_bins, name="phi", label="Gen Muon $\phi$ [rad]"))
        hist_d0_g = Hist(axis.Variable(d0_bins, name="d0", label="Gen Muon $|d_0|$ [cm]"))
        '''
        # Reco Muon Hists
        #hist_count_r = Hist(axis.Variable(np.arange(0, 10, 1), name="n_reco_muon", label="Number of Reconstructed Muons"))
        #hist_pt_r = Hist(axis.Variable(pt_bins, name="pt", label="Reco Muon $p_T$ [GeV]"))
        hist_eta_r = Hist(axis.Variable(eta_bins, name="eta", label="Reco Muon $\eta$"))
        #hist_phi_r = Hist(axis.Variable(phi_bins, name="phi", label="Reco Muon $\phi$ [rad]"))
        #hist_d0_r = Hist(axis.Variable(d0_bins, name="d0", label="Reco Muon $|d_0|$ [cm]"))
        
        '''
        # DisMuon Hists ---
        hist_pt_d = Hist(axis.Variable(pt_bins, name="pt", label="Displaced Muon $p_T$ [GeV]"))
        hist_eta_d = Hist(axis.Variable(eta_bins, name="eta", label="Displaced Muon $\eta$"))
        hist_phi_d = Hist(axis.Variable(phi_bins, name="phi", label="Displaced Muon $\phi$ [rad]"))
        hist_d0_d = Hist(axis.Variable(d0_bins, name="d0", label="Displaced Muon $|d_0|$ [cm]"))

        disMuon_data_to_compute = ak.zip({
            "dis_pt": ak.flatten(events.DisMuon.pt, axis=None),
            "dis_eta": ak.flatten(events.DisMuon.eta, axis=None),
            "dis_phi": ak.flatten(events.DisMuon.phi, axis=None),
            "dis_d0": ak.flatten(events.DisMuon.dxy, axis=None),
        })
        # Compute ONCE for all DisMuon variables
        disMuon_data = disMuon_data_to_compute.compute()

        hist_pt_d.fill(disMuon_data["dis_pt"])
        hist_eta_d.fill(disMuon_data["dis_eta"])
        hist_phi_d.fill(disMuon_data["dis_phi"])
        hist_d0_d.fill(disMuon_data["dis_d0"])

        print("--- Saving Displaced Muon Histograms ---")
        save_plot(hist_pt_d, "DisMuon_pt", sample_name, PREFIX_DISMUON, OUTPUT_DIR)
        save_plot(hist_eta_d, "DisMuon_eta", sample_name, PREFIX_DISMUON, OUTPUT_DIR)
        save_plot(hist_phi_d, "DisMuon_phi", sample_name, PREFIX_DISMUON, OUTPUT_DIR)
        save_plot(hist_d0_d, "DisMuon_d0", sample_name, PREFIX_DISMUON, OUTPUT_DIR)
        print("-----------------------------------")
        '''
        
        '''
        # ====================================================================
        # SINGLE COMPUTATION AND FILLING
        # ====================================================================

        # --- Gen Muon (Particle-Level) ---
        gen_data_to_compute = ak.zip({
            "pt": ak.flatten(events.GenMuon.pt, axis=None),
            "eta": ak.flatten(events.GenMuon.eta, axis=None),
            "phi": ak.flatten(events.GenMuon.phi, axis=None),
            "d0": ak.flatten(events.GenMuon.d0, axis=None),
        })
        # Compute ONCE for all Gen variables
        gen_data = gen_data_to_compute.compute()

        # Fill Gen Histograms
        hist_pt_g.fill(gen_data["pt"])
        hist_eta_g.fill(gen_data["eta"])
        hist_phi_g.fill(gen_data["phi"])
        hist_d0_g.fill(gen_data["d0"])

        # --- Reco Muon Count (Event-Level) ---
        reco_count_data = ak.num(events.Muon, axis=1).compute()
        hist_count_r.fill(reco_count_data)
        '''
        # --- Reco Muon (Particle-Level) ---
        reco_data_to_compute = ak.zip({
            #"pt": ak.flatten(events.Muon.pt, axis=None),
            "eta": ak.flatten(events.Muon.eta, axis=None),
            #"phi": ak.flatten(events.Muon.phi, axis=None),
            #"d0": ak.flatten(events.Muon.dxy, axis=None), # Use events.Muon.d0 as defined earlier
        })
        # Compute ONCE for all Reco variables
        reco_data = reco_data_to_compute.compute()

        # Fill Reco Histograms
        #hist_pt_r.fill(reco_data["pt"])
        hist_eta_r.fill(reco_data["eta"])
        #hist_phi_r.fill(reco_data["phi"])
        #hist_d0_r.fill(reco_data["d0"])

        # ====================================================================
        # PLOTTING AND SAVING
        # ====================================================================
        '''
        print("--- Saving Gen Muon Histograms ---")
        save_plot(hist_pt_g, "GenMuon_pt", sample_name, PREFIX_GEN, OUTPUT_DIR)
        save_plot(hist_eta_g, "GenMuon_eta", sample_name, PREFIX_GEN, OUTPUT_DIR)
        save_plot(hist_phi_g, "GenMuon_phi", sample_name, PREFIX_GEN, OUTPUT_DIR)
        save_plot(hist_d0_g, "GenMuon_d0", sample_name, PREFIX_GEN, OUTPUT_DIR)
        '''
        print("--- Saving Reco Muon Histograms ---")
        #save_plot(hist_count_r, "RecoMuon_count", sample_name, PREFIX_RECO, OUTPUT_DIR, is_reco_muon=True)
        #save_plot(hist_pt_r, "RecoMuon_pt", sample_name, PREFIX_RECO, OUTPUT_DIR, is_reco_muon=True)
        save_plot(hist_eta_r, "RecoMuon_eta_zoom", sample_name, PREFIX_RECO, OUTPUT_DIR, is_reco_muon=True)
        #save_plot(hist_phi_r, "RecoMuon_phi", sample_name, PREFIX_RECO, OUTPUT_DIR, is_reco_muon=True)
        #save_plot(hist_d0_r, "RecoMuon_d0", sample_name, PREFIX_RECO, OUTPUT_DIR, is_reco_muon=True)
        print("-----------------------------------")