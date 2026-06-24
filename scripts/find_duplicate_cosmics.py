#!/usr/bin/env python3
"""
Scan a muon dump (Sara's log format, now with DT + CSC segments) for DUPLICATE
pairs -- two muons in the same event with the same pt, eta, phi -- and compute
the back-to-back angle two independent ways:

   cos_seg     : angle between the two legs' AVERAGE segment positions (DT + CSC)
   cos_OTouter : angle between the two legs' OT-outer points (standalone track end)
   cos_mom     : angle between the two momentum vectors (confirmation; ~+1 for dups)

Outputs:
   duplicate_pairs.csv      one row per duplicate pair (opens in Excel)
   cos_seg_hist.png         histogram of cos_seg   (average method, incl. CSC)
   cos_OTouter_hist.png     histogram of cos_OTouter (OT-outer-only method)

Run:
   python3 find_duplicate_cosmics.py
   python3 find_duplicate_cosmics.py somefile.txt
"""

import sys, re, math, csv

# ---------------------------------------------------------------- settings
DEFAULT_FILE = "log_AllEvts_addCSC_brandi.txt"
DPT, DETA, DPHI = 0.5, 0.02, 0.02     # "same pt/eta/phi" tolerances
NUM = re.compile(r"^-?\d")            # token that starts a number
# ------------------------------------------------------------------------

# ---------- vector helpers (all the math lives here) --------------------
def wrap(d):
    while d >  math.pi: d -= 2*math.pi
    while d < -math.pi: d += 2*math.pi
    return d
def dot(a, b):  return sum(x*y for x, y in zip(a, b))
def norm(a):    return math.sqrt(dot(a, a))
def cosang(a, b):
    if a is None or b is None or norm(a) == 0 or norm(b) == 0: return None
    return dot(a, b) / (norm(a) * norm(b))
def avg(seglist):
    n = len(seglist)
    return (sum(s[0] for s in seglist)/n,
            sum(s[1] for s in seglist)/n,
            sum(s[2] for s in seglist)/n)
def mom3(m):
    return (m["pt"]*math.cos(m["phi"]),
            m["pt"]*math.sin(m["phi"]),
            m["pt"]*math.sinh(m["eta"]))
def impact_xy(P, Q):
    ux, uy = Q[0]-P[0], Q[1]-P[1]
    cz = P[0]*uy - P[1]*ux
    un = math.hypot(ux, uy)
    return abs(cz)/un if un else 0.0
def last3(ln):                         # last three numbers on a line
    nums = [float(x) for x in ln.replace("/", " ").split() if NUM.match(x)]
    return tuple(nums[-3:]) if len(nums) >= 3 else None

# ---------- parse the log file -----------------------------------------
def parse(path):
    events, cur, mu = [], None, None
    rec_re = re.compile(r"Run (\d+), Event (\d+), LumiSection (\d+)")
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            s = ln.strip()
            if "Begin processing" in ln:
                m = rec_re.search(ln)
                cur = {"run": m.group(1) if m else "?",
                       "ev":  m.group(2) if m else "?",
                       "lumi":m.group(3) if m else "?", "mu": []}
                events.append(cur); mu = None
            elif s == "MUON":
                if cur is None:
                    cur = {"run":"?","ev":"?","lumi":"?","mu":[]}; events.append(cur)
                mu = {"seg": [], "nDT": 0, "nCSC": 0, "OTo": None, "OTi": None}
                cur["mu"].append(mu)
            elif s.startswith("pt ="):
                f = ln.replace("=", " ").split()
                mu["pt"], mu["eta"], mu["phi"] = float(f[1]), float(f[3]), float(f[5])
            elif s.startswith("STA ="):
                f = ln.replace("=", " ").split()
                mu["STA"], mu["TRK"], mu["GLB"] = int(f[1]), int(f[3]), int(f[5])
            elif s.startswith("OT outer"):
                mu["OTo"] = last3(ln)
            elif s.startswith("OT inner"):
                mu["OTi"] = last3(ln)
            elif "gpos" in ln:                       # DT or CSC segment
                p = last3(ln)
                if p:
                    mu["seg"].append(p)
                    if "CSC" in ln: mu["nCSC"] += 1
                    else:           mu["nDT"]  += 1
    return events

def mtype(m): return "STA%dTRK%dGLB%d" % (m.get("STA",0), m.get("TRK",0), m.get("GLB",0))

# ---------- find duplicates and build the table ------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    events = parse(path)

    rows = []
    for e in events:
        mus = [m for m in e["mu"] if "pt" in m]
        for i in range(len(mus)):
            for j in range(i+1, len(mus)):
                a, b = mus[i], mus[j]
                if not (abs(a["pt"]-b["pt"]) < DPT and
                        abs(a["eta"]-b["eta"]) < DETA and
                        abs(wrap(a["phi"]-b["phi"])) < DPHI):
                    continue
                ra = avg(a["seg"]) if a["seg"] else None
                rb = avg(b["seg"]) if b["seg"] else None
                cos_seg = cosang(ra, rb)
                cos_oto = cosang(a["OTo"], b["OTo"])
                dxy  = impact_xy(ra, rb) if (ra and rb) else None
                hemi = ("opposite" if ra[1]*rb[1] < 0 else "same") if (ra and rb) else None
                rows.append(dict(
                    run=e["run"], lumi=e["lumi"], event=e["ev"],
                    A_type=mtype(a), A_pt=a["pt"], A_eta=a["eta"], A_phi=a["phi"],
                    A_nDT=a["nDT"], A_nCSC=a["nCSC"],
                    A_segx=ra[0] if ra else "", A_segy=ra[1] if ra else "", A_segz=ra[2] if ra else "",
                    A_OTox=a["OTo"][0] if a["OTo"] else "", A_OToy=a["OTo"][1] if a["OTo"] else "", A_OToz=a["OTo"][2] if a["OTo"] else "",
                    B_type=mtype(b), B_pt=b["pt"], B_eta=b["eta"], B_phi=b["phi"],
                    B_nDT=b["nDT"], B_nCSC=b["nCSC"],
                    B_segx=rb[0] if rb else "", B_segy=rb[1] if rb else "", B_segz=rb[2] if rb else "",
                    B_OTox=b["OTo"][0] if b["OTo"] else "", B_OToy=b["OTo"][1] if b["OTo"] else "", B_OToz=b["OTo"][2] if b["OTo"] else "",
                    cos_mom=cosang(mom3(a), mom3(b)),
                    cos_seg=cos_seg, cos_OTouter=cos_oto,
                    dxy_proxy_cm=dxy, hemisphere=hemi))

    cols = ["run","lumi","event",
            "A_type","A_pt","A_eta","A_phi","A_nDT","A_nCSC","A_segx","A_segy","A_segz","A_OTox","A_OToy","A_OToz",
            "B_type","B_pt","B_eta","B_phi","B_nDT","B_nCSC","B_segx","B_segy","B_segz","B_OTox","B_OToy","B_OToz",
            "cos_mom","cos_seg","cos_OTouter","dxy_proxy_cm","hemisphere"]
    def r(v): return round(v, 4) if isinstance(v, float) else v
    with open("duplicate_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for row in rows: w.writerow({k: r(row[k]) for k in cols})

    seg_vals = [x["cos_seg"]     for x in rows if x["cos_seg"]     is not None]
    oto_vals = [x["cos_OTouter"] for x in rows if x["cos_OTouter"] is not None]
    print("file scanned         : %s" % path)
    print("events parsed        : %d" % len(events))
    print("duplicate pairs      : %d" % len(rows))
    print("  cos_seg computable : %d  (both legs have >=1 segment, DT or CSC)" % len(seg_vals))
    print("  cos_OTouter comp.  : %d  (both legs have an OT outer point)" % len(oto_vals))
    print("  -> wrote duplicate_pairs.csv")

    # ---- the two plots ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        step = 0.01                                                       
        bins = [-1.0 + step*i for i in range(int(round(1.0/step)) + 1)]

        def hist(vals, fname, title):
            plt.figure(figsize=(7,4.5))
            plt.hist(vals, bins=bins, edgecolor="black")
            plt.xlim(-1.0, 0.0)
            plt.axvline(-0.9, color="green", ls="--", label="cut cos α = -0.9")
            plt.xlabel(r"$\cos\alpha$"); plt.ylabel("duplicate pairs")
            plt.title(title); plt.legend()
            plt.tight_layout(); plt.savefig(fname, dpi=130); plt.close()
            print("  -> wrote %s" % fname)

        hist(seg_vals, "cos_seg_hist.png",
             "cos α from AVERAGE segment position (DT + CSC), %d pairs" % len(seg_vals))
        hist(oto_vals, "cos_OTouter_hist.png",
             "cos α from OT-outer only, %d pairs" % len(oto_vals))
    except Exception as ex:
        print("  (plots skipped: %s)" % ex)

if __name__ == "__main__":
    main()