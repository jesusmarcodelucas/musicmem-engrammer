#!/usr/bin/env python3
"""Regenerate the three preprint figures from the FACTORED-encoding result JSONs.
Outputs: capacity_dale.png, ablation_dale.png, wiring_dale.png (in results/)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R = Path("results"); OUT = R

def fig_capacity():
    sw = json.load(open(R / "sweep_factored.json"))
    Ls = [64,128,256,512,1024]
    arm=[np.mean([sw[f"L{L}_s{s}"]["AR"] for s in range(3)]) for L in Ls]
    ars=[np.std ([sw[f"L{L}_s{s}"]["AR"] for s in range(3)]) for L in Ls]
    tfm=[np.mean([sw[f"L{L}_s{s}"]["TF"] for s in range(3)]) for L in Ls]
    fig,ax=plt.subplots(figsize=(7,4.3))
    ax.plot(Ls,tfm,"s--",color="#c82828",label="teacher-forced",ms=6)
    ax.errorbar(Ls,arm,yerr=ars,fmt="o-",color="#1e50b4",label="autoregressive",ms=7,capsize=3)
    ax.axhline(1.0,ls=":",color="#999",lw=1)
    ax.set_xscale("log",base=2); ax.set_xticks(Ls); ax.set_xticklabels(Ls)
    ax.set_xlabel("latent size L"); ax.set_ylabel("pitch accuracy"); ax.set_ylim(0,1.05)
    ax.set_title("Capacity scaling, factored encoding (3 seeds): perfect recall at L=512")
    ax.legend(loc="lower right",fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/"capacity_dale.png",dpi=140); print("wrote capacity_dale.png")

def fig_ablation():
    rows=json.load(open(R/"ablationF256_3seed.json"))
    order=["full","dale=False","ltd=False","kwta=False","multisyn=False","consolidation=False",
           "div_inh=False","sparse_wiring=False","classifier=False","+encoder_learn"]
    labels={"full":"full","dale=False":"no E/I","ltd=False":"no sign/LTD","kwta=False":"no k-WTA",
            "multisyn=False":"no multisyn","consolidation=False":"no consol.","div_inh=False":"no div.inh.",
            "sparse_wiring=False":"dense","classifier=False":"no recog.","+encoder_learn":"+enc.learn"}
    d={r["config"]:r for r in rows}
    x=np.arange(len(order)); w=0.40
    ar=[d[c]["AR_mean"] for c in order]; arsd=[d[c]["AR_sd"] for c in order]
    tf=[d[c]["TF_mean"] for c in order]; tfsd=[d[c]["TF_sd"] for c in order]
    fig,ax=plt.subplots(figsize=(10.5,4.4))
    b=ax.bar(x-w/2,ar,w,yerr=arsd,capsize=2,color="#1e50b4",alpha=0.9,label="autoregressive"); b[0].set_color("#777")
    ax.bar(x+w/2,tf,w,yerr=tfsd,capsize=2,facecolor="none",edgecolor="#c82828",lw=1.3,ls="--",hatch="///",label="teacher-forced")
    ax.axhline(d["full"]["AR_mean"],ls=":",color="#777",lw=1)
    ax.set_xticks(x); ax.set_xticklabels([labels[c] for c in order],rotation=30,ha="right")
    ax.set_ylabel("pitch accuracy"); ax.set_ylim(0,1.08)
    ax.set_title("Leave-one-out ablation, factored encoding (L=256, 3 seeds)")
    ax.legend(loc="upper right",fontsize=9); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/"ablation_dale.png",dpi=140); print("wrote ablation_dale.png")

def fig_wiring():
    g=json.load(open(R/"wiringF256.json"))
    ds=[1.0,0.25,0.15,0.10,0.05]
    m=[np.mean([g[f"d{d}_s{s}"]["AR"] for s in range(3)]) for d in ds]
    sd=[np.std([g[f"d{d}_s{s}"]["AR"] for s in range(3)]) for d in ds]
    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.axvspan(5,15,color="#c8e0c8",alpha=0.4,label="cortical / MB range")
    ax.errorbar([d*100 for d in ds],m,yerr=sd,fmt="o-",color="#1e50b4",ms=7,capsize=3)
    ax.set_xlabel("input wiring density (%)"); ax.set_ylabel("AR pitch accuracy"); ax.set_ylim(0.6,1.05)
    ax.set_title("Recall vs wiring density, factored (L=256, 3 seeds)")
    ax.legend(loc="lower right",fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/"wiring_dale.png",dpi=140); print("wrote wiring_dale.png")

if __name__=="__main__":
    fig_capacity(); fig_ablation(); fig_wiring(); print("done")
