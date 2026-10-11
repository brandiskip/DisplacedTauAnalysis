import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

def plot_score_chHEF_overlay(h_obj, title, outpath):
    """
    Overlays jet disTauTag_score1 with and without the chHEF > 0.01 cut.
    Lower panel: (chHEF > 0.01) / (no chHEF cut) = fraction of jets kept in each score bin.
    """
    if np.sum(h_obj.values()) <= 0:
        print(f"Skipping {outpath}: No jets in histogram.")
        return

    h_all = h_obj[{"cat": "no chHEF cut"}]
    h_cut = h_obj[{"cat": "chHEF > 0.01"}]
    edges = h_all.axes[0].edges

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    h_all.plot1d(ax=ax, label="no chHEF cut")
    h_cut.plot1d(ax=ax, label="chHEF > 0.01")
    ax.set_yscale("log")
    ax.set_ylabel("Jets")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend()

    num, den = h_cut.values(), h_all.values()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = num / den  # NaN in empty bins, so they aren't drawn
        err = np.sqrt(frac * (1.0 - frac) / den)
    rax.errorbar(h_all.axes[0].centers, frac, yerr=err, fmt="o", markersize=3, color="black")
    rax.axhline(1.0, color="gray", ls="--", lw=1)
    rax.set_xlim(edges[0], edges[-1])
    rax.set_xlabel("Jet disTauTag_score1")
    rax.set_ylabel("Fraction kept")
    rax.grid(True, ls="--", alpha=0.5)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved plot to: {outpath}")
    print(f"  jets in range: {den.sum():.0f} (no cut) -> {num.sum():.0f} (chHEF > 0.01); "
          f"below {edges[0]}: {h_all.values(flow=True)[0]:.0f} -> {h_cut.values(flow=True)[0]:.0f}")

def plot_score_vs_chHEF(h_obj, title, outpath):
    """
    2D jet disTauTag_score1 vs. chHEF with a log color scale.
    The chHEF axis is linear below 0.01 and log above; the dashed line marks the chHEF > 0.01 cut.
    """
    if np.sum(h_obj.values()) <= 0:
        print(f"Skipping {outpath}: No jets in histogram.")
        return

    x_edges = h_obj.axes["chHEF"].edges
    y_edges = h_obj.axes["score"].edges

    fig, ax = plt.subplots(figsize=(9, 7))
    mesh = ax.pcolormesh(x_edges, y_edges, h_obj.values().T, cmap="viridis", norm=LogNorm())
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Jets")

    ax.set_xscale("symlog", linthresh=0.01)
    ax.axvline(0.01, color="red", ls="--", lw=1.5)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xlabel("Jet chHEF (linear below 0.01, log above; dashed line = cut)")
    ax.set_ylabel("Jet disTauTag_score1")
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
# Muon-jet vs. tau_h-jet study: settings and plotting helper
# ==============================================================================
HIGH_SCORE = 0.9                          # muon jets above this score get their own category
PF_TYPES = [1, 2, 11, 13, 22, 130, 211]   # |pdgId| of PF candidates, in x-axis order
PF_TYPE_LABELS = ["HF had", r"HF e/$\gamma$", "e", r"$\mu$", r"$\gamma$", "h0", r"h$^\pm$"]

MU_STUDY_CATS = [
    "GenMuon-matched",
    f"GenMuon-matched, score > {HIGH_SCORE}",
    r"1-prong $\tau_h$-matched",
    r"3-prong $\tau_h$-matched",
]

# name: (nbins, lo, hi, x label). Values above hi go in the last bin.
MU_STUDY_VARS = {
    # Jet level: one entry per jet
    "score":             (50, 0.0, 1.0, "Jet disTauTag_score1"),
    "score_zoom":        (50, 0.9, 1.0, "Jet disTauTag_score1"),
    "jet_pt":            (60, 0, 600, r"Jet $p_T$ [GeV]"),
    "jet_eta":           (48, -2.4, 2.4, r"Jet $\eta$"),
    "jet_muEF":          (50, 0, 1, "Jet muEF"),
    "jet_chHEF":         (50, 0, 1, "Jet chHEF"),
    "jet_neHEF":         (50, 0, 1, "Jet neHEF"),
    "jet_chEmEF":        (50, 0, 1, "Jet chEmEF"),
    "jet_neEmEF":        (50, 0, 1, "Jet neEmEF"),
    "jet_nConstituents": (60, -0.5, 59.5, "Jet nConstituents"),
    "jet_dxy":           (50, 0, 20, "Jet dxy [cm] (leading charged PF cand)"),
    "nPF":               (60, -0.5, 59.5, "Number of PF candidates in jet"),
    "nPFCharged":        (40, -0.5, 39.5, "Number of charged PF candidates in jet"),
    "nPFMuon":           (5, -0.5, 4.5, "Number of PF muons in jet"),
    "lead_pf_ptfrac":    (50, 0, 1, r"Leading PF candidate $p_T$ / jet $p_T$"),
    "lead_pf_type":      (len(PF_TYPES), -0.5, len(PF_TYPES) - 0.5, "Leading PF candidate type"),
    # PF-candidate level: one entry per jet constituent
    "pf_type":           (len(PF_TYPES), -0.5, len(PF_TYPES) - 0.5, "PF candidate type"),
    "pf_ptfrac":         (50, 0, 1, r"PF candidate $p_T$ / jet $p_T$"),
    "pf_dR":             (40, 0, 0.4, r"$\Delta R$(PF candidate, jet axis)"),
    "pf_absd0":          (50, 0, 20, r"PF candidate $|d_0|$ [cm] (charged)"),
    "pf_d0sig":          (50, 0, 200, r"PF candidate $|d_0|/\sigma_{d_0}$ (charged)"),
    "pf_pt":             (50, 0, 200, r"PF candidate $p_T$ [GeV]"),
    "pf_eta":            (50, -2.5, 2.5, r"PF candidate $\eta$"),
    "pf_charge":         (3, -1.5, 1.5, "PF candidate charge"),
    "pf_puppiWeight":    (50, 0, 1, "PF candidate puppiWeight"),
    "pf_puppiWeightNoLep": (50, 0, 1, "PF candidate puppiWeightNoLep"),
    "pf_pvAssocQuality": (8, -0.5, 7.5, "PF candidate pvAssocQuality"),
    "pf_d0":             (80, -20, 20, r"PF candidate $d_0$ [cm] (charged)"),
    "pf_dz":             (80, -40, 40, r"PF candidate $d_z$ [cm] (charged)"),
    "pf_d0Err":          (50, 0, 0.5, r"PF candidate $\sigma_{d_0}$ [cm] (charged)"),
    "pf_dzErr":          (50, 0, 0.5, r"PF candidate $\sigma_{d_z}$ [cm] (charged)"),
    "pf_numberOfPixelHits": (10, -0.5, 9.5, "PF candidate numberOfPixelHits (charged)"),
    "pf_numberOfHits":   (40, -0.5, 39.5, "PF candidate numberOfHits (charged)"),
    "pf_lostInnerHits":  (5, -1.5, 3.5, "PF candidate lostInnerHits (charged)"),
    "pf_trkChi2":        (50, 0, 20, r"PF candidate track $\chi^2$ (charged)"),
    "pf_vtxChi2":        (50, 0, 20, r"PF candidate vertex $\chi^2$ (charged)"),
}

# PF candidate branches filled directly as "pf_<field>". True = charged candidates only.
# Fields missing from the ntuple are skipped.
PF_RAW_FIELDS = {
    "pt": False, "eta": False, "charge": False,
    "puppiWeight": False, "puppiWeightNoLep": False, "pvAssocQuality": False,
    "d0": True, "dz": True, "d0Err": True, "dzErr": True,
    "numberOfPixelHits": True, "numberOfHits": True, "lostInnerHits": True,
    "trkChi2": True, "vtxChi2": True,
}

def pf_type_index(pdgId):
    """Maps PF candidate pdgId to its position in PF_TYPES (-1 if not listed)."""
    abs_id = abs(pdgId)
    idx = ak.full_like(abs_id, -1)
    for i, t in enumerate(PF_TYPES):
        idx = ak.where(abs_id == t, i, idx)
    return idx

def plot_mu_study_overlay(h_obj, categories, xlabel, title, outpath, logy=False, ticklabels=None):
    """
    Overlays one variable for each jet category. Each category is normalized to its
    total number of entries (including under/overflow), so shapes can be compared directly.
    """
    present = list(h_obj.axes["cat"])
    fig, ax = plt.subplots(figsize=(8, 7))
    drew = False
    for cat in categories:
        if cat not in present:
            continue
        h = h_obj[{"cat": cat}]
        total = h.values(flow=True).sum()
        if total <= 0:
            continue
        ax.stairs(h.values() / total, h.axes[0].edges, label=f"{cat} (N={total:.0f})", lw=2)
        drew = True

    if not drew:
        print(f"Skipping {outpath}: No entries.")
        plt.close(fig)
        return

    if ticklabels is not None:
        ax.set_xticks(range(len(ticklabels)))
        ax.set_xticklabels(ticklabels)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of entries")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=9)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved plot to: {outpath}")

# ==============================================================================
# Processor Class
# ==============================================================================
class SignalJetProcessor(processor.ProcessorABC):
        def __init__(self, do_mu_study=False):
        
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

        # chHEF binning for the 2D plots: 20 linear bins below the 0.01 cut, 40 log bins above
        chHEF_bins = np.concatenate([np.linspace(0, 0.01, 20, endpoint=False), np.geomspace(0.01, 1, 41)])
        chHEF_axis = axis.Variable(chHEF_bins, name="chHEF", label="chHEF")

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

            "grid_den_gvt": 0,
            "grid_num_score": 0,
            "grid_num_pt": 0,
            "grid_num_dxy": 0,
            "grid_num_all": 0,

            # Jet score with vs. without chHEF > 0.01
            "score_chHEF_full": Hist(cat_axis, axis.Regular(50, 0.0, 1.0, name="score", label="disTauTag_score1")),
            "score_chHEF_zoom": Hist(cat_axis, axis.Regular(50, 0.9, 1.0, name="score", label="disTauTag_score1")),

            # Jet score vs. chHEF (2D)
            "score_vs_chHEF_full": Hist(chHEF_axis, axis.Regular(50, 0.0, 1.0, name="score", label="disTauTag_score1")),
            "score_vs_chHEF_zoom": Hist(chHEF_axis, axis.Regular(50, 0.9, 1.0, name="score", label="disTauTag_score1")),
        }


        # Muon-jet vs. tau_h-jet study histograms
        self.do_mu_study = do_mu_study
        for name, (nbins, lo, hi, xlabel) in MU_STUDY_VARS.items():
            self.output[f"mustudy_{name}"] = Hist(cat_axis, axis.Regular(nbins, lo, hi, name="val", label=xlabel))
        self.output["mustudy_n_overlap"] = 0
        self.output["mustudy_pf_fields"] = set()

    def _mu_fill(self, cat, name, values):
        """Flattens values and fills mustudy_<name>; values above the range go in the last bin."""
        values = ak.to_numpy(ak.drop_none(ak.flatten(values, axis=None)))
        hi = MU_STUDY_VARS[name][2]
        values = np.minimum(values, np.nextafter(hi, -np.inf))
        self.output[f"mustudy_{name}"].fill(cat=cat, val=values)

    def _fill_mu_study(self, cat, jets):
        """Fills all muon-jet study histograms for one jet category (jets: [event, jet])."""
        if ak.sum(ak.num(jets)) == 0:
            return

        pf = jets.constituents.pf
        charged = pf.charge != 0
        self.output["mustudy_pf_fields"] |= set(pf.fields)

        # Jet level
        self._mu_fill(cat, "score", jets.disTauTag_score1)
        self._mu_fill(cat, "score_zoom", jets.disTauTag_score1)
        for f in ["pt", "eta", "muEF", "chHEF", "neHEF", "chEmEF", "neEmEF", "nConstituents"]:
            self._mu_fill(cat, f"jet_{f}", jets[f])
        self._mu_fill(cat, "jet_dxy", jets.dxy[jets.dxy >= 0])  # dxy = -999 for jets with no charged PF cand

        self._mu_fill(cat, "nPF", ak.num(pf, axis=2))
        self._mu_fill(cat, "nPFCharged", ak.sum(charged, axis=2))
        self._mu_fill(cat, "nPFMuon", ak.sum(abs(pf.pdgId) == 13, axis=2))

        lead_pf = ak.firsts(pf[ak.argsort(pf.pt, axis=2, ascending=False)], axis=2)
        self._mu_fill(cat, "lead_pf_ptfrac", lead_pf.pt / jets.pt)
        self._mu_fill(cat, "lead_pf_type", pf_type_index(lead_pf.pdgId))

        # PF-candidate level (jet-level quantities broadcast over each jet's constituents)
        self._mu_fill(cat, "pf_type", pf_type_index(pf.pdgId))
        self._mu_fill(cat, "pf_ptfrac", pf.pt / jets.pt)
        dphi = (pf.phi - jets.phi + np.pi) % (2 * np.pi) - np.pi
        self._mu_fill(cat, "pf_dR", np.sqrt((pf.eta - jets.eta) ** 2 + dphi ** 2))
        self._mu_fill(cat, "pf_absd0", abs(pf.d0[charged]))
        pf_sig = pf[charged & (pf.d0Err > 0)]
        self._mu_fill(cat, "pf_d0sig", abs(pf_sig.d0) / pf_sig.d0Err)

        for field, charged_only in PF_RAW_FIELDS.items():
            if field not in pf.fields:
                continue
            values = pf[field][charged] if charged_only else pf[field]
            self._mu_fill(cat, f"pf_{field}", values)

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
            (abs(events.GenMuon.distinctParent.distinctParent.pdgId) == 1000015)
        ]

        events['GenElectron'] = events.GenPart[(abs(events.GenPart.pdgId) == 11) & (events.GenPart.hasFlags("isLastCopy"))]
        events['GenElectron'] = events.GenElectron[(events.GenElectron.pt > 20) & \
                                                    (abs(events.GenElectron.eta) < 2.4) & \
                                                    (abs(events.GenElectron.distinctParent.distinctParent.pdgId) == 1000015)]
        
        mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenMuon) == 1) & (ak.num(events.GenElectron) == 0)

        # Event level masks
        #mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenMuon) == 1)
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
        # Jet Score: with vs. without chHEF > 0.01 (baseline jets)
        # =============================================================
        score_variants = {
            "no chHEF cut": baseline_jets,
            "chHEF > 0.01": baseline_jets_chHEF,
        }
        for label, jcol in score_variants.items():
            # A score of exactly 1.0 would go to the overflow bin; keep it in the last visible bin
            score = np.minimum(ak.flatten(jcol.disTauTag_score1, axis=None), 1 - 1e-6)
            self.output["score_chHEF_full"].fill(cat=label, score=score)
            self.output["score_chHEF_zoom"].fill(cat=label, score=score) 

        # Jet score vs. chHEF (2D): baseline jets with no chHEF cut, since chHEF is on the axis.
        # Values of exactly 1.0 are kept in the last bin, as for the 1D score plots.
        chHEF_vals = np.minimum(ak.flatten(baseline_jets.chHEF, axis=None), 1 - 1e-6)
        score_vals = np.minimum(ak.flatten(baseline_jets.disTauTag_score1, axis=None), 1 - 1e-6)
        self.output["score_vs_chHEF_full"].fill(chHEF=chHEF_vals, score=score_vals)
        self.output["score_vs_chHEF_zoom"].fill(chHEF=chHEF_vals, score=score_vals) 

        # =============================================================
        # Muon-jet vs. tau_h-jet study (baseline jets, no jet ID, so
        # muon-rich jets are not removed by muEF / lepton-veto cuts)
        # =============================================================
        if self.do_mu_study:
            mjets = baseline_jets
            _, dr_mu = mjets.nearest(cut_filtered_events.GenMuon, return_metric=True)
            near_gvt, dr_gvt = mjets.nearest(cut_filtered_events.GenVisStauTaus, return_metric=True)
            mu_match = ak.fill_none(dr_mu < 0.4, False)
            gvt_match = ak.fill_none(dr_gvt < 0.4, False)
            dm = ak.fill_none(near_gvt.status, -1)  # GenVisTau.status = hadronic decay mode

            # Jets matched to both the GenMuon and the GenVisStauTau are counted and left out
            self.output["mustudy_n_overlap"] += ak.sum(mu_match & gvt_match)
            mu_only = mu_match & ~gvt_match
            tau_only = gvt_match & ~mu_match

            mu_study_sel = {
                MU_STUDY_CATS[0]: mu_only,
                MU_STUDY_CATS[1]: mu_only & (mjets.disTauTag_score1 > HIGH_SCORE),
                MU_STUDY_CATS[2]: tau_only & ((dm == 0) | (dm == 1) | (dm == 2)),
                MU_STUDY_CATS[3]: tau_only & ((dm == 10) | (dm == 11)),
            }
            for cat, sel in mu_study_sel.items():
                self._fill_mu_study(cat, mjets[sel])          

        # =============================================================
        # Grid Efficiency (Score vs pT)
        # =============================================================
        
        grid_jets = cut_filtered_events.Jet[
            (abs(cut_filtered_events.Jet.eta) < 2.4) & 
            (cut_filtered_events.Jet.pt > 20) & 
            (cut_filtered_events.Jet.neHEF < 0.99) & 
            (cut_filtered_events.Jet.neEmEF < 0.9) & 
            ((cut_filtered_events.Jet.chMultiplicity + cut_filtered_events.Jet.neMultiplicity) > 1) & 
            (cut_filtered_events.Jet.chMultiplicity > 0) & 
            (cut_filtered_events.Jet.muEF < 0.5) & 
            (cut_filtered_events.Jet.chEmEF < 0.8)
        ]
        '''
        grid_jets = cut_filtered_events.Jet[
            (abs(cut_filtered_events.Jet.eta) < 2.4) & 
            (cut_filtered_events.Jet.pt > 20)]
        '''
        '''
        grid_jets = cut_filtered_events.Jet[
            (abs(cut_filtered_events.Jet.eta) < 2.4) &
            (cut_filtered_events.Jet.pt > 20) &
            (cut_filtered_events.Jet.neHEF < 0.99) &
            (cut_filtered_events.Jet.neEmEF < 0.9) &
            ((cut_filtered_events.Jet.chMultiplicity + cut_filtered_events.Jet.neMultiplicity) > 1) &
            (cut_filtered_events.Jet.chMultiplicity > 0) &
            (cut_filtered_events.Jet.muEF < 0.8) &
            (cut_filtered_events.Jet.chEmEF < 0.8) &
            (cut_filtered_events.Jet.chHEF > 0.01)
        ]
        '''
        # Denominator: Total GenVisStauTaus in the event
        self.output["grid_den_gvt"] += ak.sum(ak.num(cut_filtered_events.GenVisStauTaus))

        # Numerator 0: All jets, no leading chosen
        matched_all = cut_filtered_events.GenVisStauTaus.nearest(grid_jets, threshold=0.4)
        matched_all = ak.drop_none(matched_all)
        self.output["grid_num_all"] += ak.sum(ak.num(matched_all))

        # Numerator 1: Leading Score Jet Matched
        jets_by_score = grid_jets[ak.argsort(grid_jets.disTauTag_score1, ascending=False)]
        lead_score_jet = ak.singletons(ak.firsts(jets_by_score))
        
        matched_score = lead_score_jet.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        matched_score = ak.drop_none(matched_score)
        self.output["grid_num_score"] += ak.sum(ak.num(matched_score))

        # Numerator 2: Leading pT Jet Matched
        jets_by_pt = grid_jets[ak.argsort(grid_jets.pt, ascending=False)]
        lead_pt_jet = ak.singletons(ak.firsts(jets_by_pt))
        
        matched_pt = lead_pt_jet.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        matched_pt = ak.drop_none(matched_pt)
        self.output["grid_num_pt"] += ak.sum(ak.num(matched_pt))

        # Numerator 3: Leading dxy Jet Matched
        jets_by_dxy = grid_jets[ak.argsort(grid_jets.dxy, ascending=False)]
        lead_dxy_jet = ak.singletons(ak.firsts(jets_by_dxy))

        matched_dxy = lead_dxy_jet.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        matched_dxy = ak.drop_none(matched_dxy)
        self.output["grid_num_dxy"] += ak.sum(ak.num(matched_dxy))

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

        self.output["muEF"].fill(
            cat="Highest Score Jet (Unmatched)", 
            val=ak.drop_none(ak.flatten(highest_unmatched_jets.muEF, axis=None))
        )
        self.output["muEF"].fill(
            cat="2nd Highest Score Jet (Matched)", 
            val=ak.drop_none(ak.flatten(second_matched_jets.muEF, axis=None))
        )

        return self.output

    def postprocess(self, accumulator):
        return accumulator

# ==============================================================================
# Execution Block
# ==============================================================================
if __name__ == '__main__':
    signal_pkl = "samples/Signal_Samples/Stau_300_100mm_preprocessed.pkl"
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
        processor_instance=SignalJetProcessor(do_mu_study=True),
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
        {"num_key": "num_d0_highest",  "den_key": "den_d0",  "xlabel": r"GenVisTau $|d_{xy}|$ [cm]", "title": "Matching Efficiency (Highest Score Jet) vs $|d_{xy}|$", "filename": "eff_vs_dxy_highest_score"},
        
        {"num_key": "num_eta_indiv",   "den_key": "den_eta", "xlabel": r"GenVisTau $\eta$", "title": "Matching Efficiency (Individual Cuts) vs $\eta$", "filename": "eff_vs_eta_individual_cuts", "ylim": (0.4, 1.05)},
        {"num_key": "num_Lxy_indiv",   "den_key": "den_Lxy", "xlabel": r"GenVisTau $L_{xy}$ [cm]", "title": "Matching Efficiency (Individual Cuts) vs $L_{xy}$", "filename": "eff_vs_Lxy_individual_cuts", "ylim": (0.4, 1.05)},
        {"num_key": "num_pt_indiv",    "den_key": "den_pt",  "xlabel": r"GenVisTau $p_T$ [GeV]", "title": "Matching Efficiency (Individual Cuts) vs $p_T$", "filename": "eff_vs_pt_individual_cuts", "ylim": (0.4, 1.05)}]

    for p in plots_to_make:
        title = f"{p['title']} ({sample_name.replace('_', ' ')}) v19"
        outpath = os.path.join(OUTPUT_DIR, f"{p['filename']}_{sample_name}_v19.pdf")

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
    muEF_title = f"Jet muEF Comparison ({sample_name.replace('_', ' ')})"
    muEF_outpath = os.path.join(OUTPUT_DIR, f"Stau_300_100mm_muEF_Matched_vs_Unmatched_v19.pdf")
    save_categorical_overlay(out["muEF"], "muEF", muEF_title, muEF_outpath)

    # Generate the jet score plots (with vs. without chHEF > 0.01)
    for key, tag in [("score_chHEF_full", "0to1"), ("score_chHEF_zoom", "0p9to1")]:
        plot_score_chHEF_overlay(
            out[key],
            title=f"Jet Score, Baseline Jets ({sample_name.replace('_', ' ')}) v19",
            outpath=os.path.join(OUTPUT_DIR, f"{sample_name}_jet_score_chHEF_{tag}_v19.pdf"),
        )  

    # Generate the 2D jet score vs. chHEF plots
    for key, tag in [("score_vs_chHEF_full", "0to1"), ("score_vs_chHEF_zoom", "0p9to1")]:
        plot_score_vs_chHEF(
            out[key],
            title=f"Jet Score vs. chHEF, Baseline Jets ({sample_name.replace('_', ' ')}) v19",
            outpath=os.path.join(OUTPUT_DIR, f"{sample_name}_jet_score_vs_chHEF_{tag}_v19.pdf"),
        )  


    # Generate the muon-jet vs. tau_h-jet study plots
    MU_DIR = os.path.join(OUTPUT_DIR, "muon_vs_tauh_jets")
    os.makedirs(MU_DIR, exist_ok=True)
    for name, (nbins, lo, hi, xlabel) in MU_STUDY_VARS.items():
        # The high-score muon category is redundant on the score plots themselves
        cats = [c for c in MU_STUDY_CATS if c != MU_STUDY_CATS[1]] if name.startswith("score") else MU_STUDY_CATS
        plot_mu_study_overlay(
            out[f"mustudy_{name}"],
            categories=cats,
            xlabel=xlabel,
            title=f"Muon vs. $\\tau_h$ Jets ({sample_name.replace('_', ' ')}) v19",
            outpath=os.path.join(MU_DIR, f"{sample_name}_{name}_v19.pdf"),
            logy=name.startswith("score"),
            ticklabels=PF_TYPE_LABELS if name.endswith("_type") else None,
        )

    print(f"\nMuon-jet study summary ({sample_name}):")
    h_score = out["mustudy_score"]
    for cat in MU_STUDY_CATS:
        if cat not in list(h_score.axes["cat"]):
            continue
        vals = h_score[{"cat": cat}].values()
        n_tot = vals.sum()
        n_high = vals[h_score.axes["val"].centers > HIGH_SCORE].sum()
        if n_tot > 0:
            print(f"  {cat}: {n_tot:.0f} jets, fraction with score > {HIGH_SCORE}: {n_high / n_tot:.3f}")
    print(f"  Jets matched to both GenMuon and GenVisStauTau (left out): {out['mustudy_n_overlap']}")
    print(f"  PF candidate fields in the ntuple that are not plotted: "
          f"{sorted(out['mustudy_pf_fields'] - set(PF_RAW_FIELDS))}")

    # ==============================================================================
    # ADDITION: Generate muEF plots for 100 GeV and 500 GeV (v19)
    # ==============================================================================
    PKL_DIR = "samples/Signal_Samples/"
    additional_samples = ["Stau_100_100mm", "Stau_500_100mm"]

    for samp in additional_samples:
        pkl_path = os.path.join(PKL_DIR, f"{samp}_preprocessed.pkl")
        print(f"\nProcessing {samp} for muEF plot...")
        
        try:
            with open(pkl_path, "rb") as f:
                runnable_samp = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find {pkl_path}. Check the spelling or path.")
            continue

        # Run the processor on this specific sample
        out_samp = runner(
            runnable_samp,
            treename="Events",
            processor_instance=SignalJetProcessor(),
        )

        # Generate ONLY the muEF plot with the v19 title and filename
        muEF_title_samp = f"Jet muEF Comparison ({samp.replace('_', ' ')}) v19"
        muEF_outpath_samp = os.path.join(OUTPUT_DIR, f"{samp}_muEF_Matched_vs_Unmatched_v19.pdf")
        
        save_categorical_overlay(out_samp["muEF"], "muEF", muEF_title_samp, muEF_outpath_samp)

    # ==============================================================================
    # ADDITION: Generate 2D Grid Efficiency Plots (Mass vs cTau)
    # ==============================================================================
    import json
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe

    masses = [100, 200, 300, 500]
    lifetimes = [1, 10, 100, 1000]

    grid_data_score = []
    grid_data_pt = []
    grid_data_dxy = []
    grid_data_all = []

    print("\nStarting Grid Efficiency Processing...")
    for m in masses:
        for ct in lifetimes:
            samp_name = f"Stau_{m}_{ct}mm"
            pkl_path = os.path.join(PKL_DIR, f"{samp_name}_preprocessed.pkl")
            
            if not os.path.exists(pkl_path):
                print(f"Skipping {samp_name}: File not found.")
                continue
                
            with open(pkl_path, "rb") as f:
                runnable_grid = pickle.load(f)
                
            out_grid = runner(
                runnable_grid,
                treename="Events",
                processor_instance=SignalJetProcessor(),
            )
            
            den = out_grid["grid_den_gvt"]
            num_score = out_grid["grid_num_score"]
            num_pt = out_grid["grid_num_pt"]
            num_dxy = out_grid["grid_num_dxy"]
            num_all = out_grid["grid_num_all"]
            
            eff_score = num_score / den if den > 0 else 0.0
            eff_pt = num_pt / den if den > 0 else 0.0
            eff_dxy   = num_dxy   / den if den > 0 else 0.0
            eff_all = num_all / den if den > 0 else 0.0
            
            grid_data_score.append({"mass": m, "lifetime": ct, "efficiency": eff_score})
            grid_data_pt.append({"mass": m, "lifetime": ct, "efficiency": eff_pt})
            grid_data_dxy.append( {"mass": m, "lifetime": ct, "efficiency": eff_dxy})
            grid_data_all.append({"mass": m, "lifetime": ct, "efficiency": eff_all})

    # Save to JSON
    with open(os.path.join(OUTPUT_DIR, "grid_eff_score.json"), "w") as f:
        json.dump(grid_data_score, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "grid_eff_pt.json"), "w") as f:
        json.dump(grid_data_pt, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "grid_eff_dxy.json"), "w") as f:    # <-- ADD THIS
        json.dump(grid_data_dxy, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "grid_eff_all.json"), "w") as f:
        json.dump(grid_data_all, f, indent=4)

    # Plotting Function
    def plot_efficiency_grid(data_list, title, out_filename):
        if not data_list:
            return
            
        Z = np.zeros((len(lifetimes), len(masses)))
        mass_idx = {m: i for i, m in enumerate(masses)}
        lifetime_idx = {lt: i for i, lt in enumerate(lifetimes)}

        for entry in data_list:
            Z[lifetime_idx[entry["lifetime"]], mass_idx[entry["mass"]]] = entry["efficiency"]

        # Reverse y-axis to match standard layout
        Z = Z[::-1]
        lts = lifetimes[::-1]

        fig, ax = plt.subplots(figsize=(650/72, 566/72), constrained_layout=True)
        cmap = cm.get_cmap("plasma")
        norm = mcolors.Normalize(vmin=0, vmax=1)
        im = ax.imshow(Z, cmap=cmap, norm=norm)
        ax.set_aspect("auto")

        ax.set_xticks(range(len(masses)))
        ax.set_xticklabels([str(m) for m in masses])
        ax.set_yticks(range(len(lts)))
        ax.set_yticklabels([f"{lt} mm" for lt in lts])
        ax.tick_params(axis="both", labelsize=18)
        
        ax.set_xlabel("Mass [GeV]", fontsize=20)
        ax.set_ylabel(r"$c\tau$ [mm]", fontsize=20)
        plt.title(title, fontsize=15, pad=15)

        for i in range(len(lts)):
            for j in range(len(masses)):
                eff_val = Z[i, j]
                if eff_val > 0:
                    text_color = "white" if eff_val < 0.5 else "black"
                    ax.text(j, i, f"{eff_val:.3f}", ha="center", va="center",
                            color=text_color, fontsize=26, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Efficiency", fontsize=18)
        cbar.ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("Mass [GeV]", fontsize=22)
        ax.set_ylabel(r"$c\tau$ [mm]", fontsize=22)
        plt.title(title, fontsize=22, pad=15)
        outpath = os.path.join(OUTPUT_DIR, out_filename)
        plt.savefig(outpath)
        plt.close()
        print(f"Saved grid plot to {outpath}")
    '''
    # Generate Heatmaps
    plot_efficiency_grid(
        grid_data_score, 
        "(Leading Score Jet Matched) / (GenVisTaus)\n[isTightLV (muEF < 0.5), Require GenMuon]", 
        "grid_efficiency_leading_score_v19.pdf"
    )
    
    plot_efficiency_grid(
        grid_data_pt, 
        "(Leading $p_T$ Jet Matched) / (GenVisTaus)\n[isTightLV (muEF < 0.5), Require GenMuon]", 
        "grid_efficiency_leading_pt_v19.pdf"
    )

    plot_efficiency_grid(
        grid_data_dxy,
        r"(Leading $|d_{xy}|$ Jet Matched) / (GenVisTaus)" + "\n[isTightLV (muEF < 0.5), Require GenMuon]",
        "grid_efficiency_leading_dxy_v19.pdf"
    )
    '''
    plot_efficiency_grid(
        grid_data_score, 
        "(Leading Score Matched Jets) / (GenVisTaus)]", 
        "grid_efficiency_leading_score_v19_jetId_muEF05_noChHEF.pdf"
    )
    
    plot_efficiency_grid(
        grid_data_pt, 
        "(Leading pT Matched Jets) / (GenVisTaus)]", 
        "grid_efficiency_leading_pt_v19_jetId_muEF05_noChHEF.pdf"
    )

    plot_efficiency_grid(
        grid_data_dxy,
        r"(Leading |d_{xy}| Matched Jets) / (GenVisTaus)]",
        "grid_efficiency_leading_dxy_v19_jetId_muEF05_noChHEF.pdf"
    )

    plot_efficiency_grid(
        grid_data_all,
        r"(All Matched Jets) / (GenVisTaus)]",
        "grid_efficiency_all_jets_v19_jetId_muEF05_noChHEF.pdf"
    )
    print("\nDone! All muEF plots generated.")