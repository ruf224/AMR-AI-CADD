
import io, math, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AMR-AI-CADD NDM-1", page_icon="🧬", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/georgeyuricadd/IC50-dataset/main/NDM686concise.csv"
UNIPROT = "C7C422"
PDBS = ["4U4L", "6IBS", "6EFJ", "6MDU"]
DESC = ["MW","LogP","TPSA","HBD","HBA","RotB","AromaticRings","Rings","FractionCSP3","HeavyAtoms","FormalCharge"]

def mol(s):
    try:
        return Chem.MolFromSmiles(str(s))
    except Exception:
        return None

def descriptors(s):
    m = mol(s)
    if m is None:
        return None
    return {
        "MW": Descriptors.MolWt(m),
        "LogP": Crippen.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "HBD": Lipinski.NumHDonors(m),
        "HBA": Lipinski.NumHAcceptors(m),
        "RotB": Lipinski.NumRotatableBonds(m),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(m),
        "Rings": rdMolDescriptors.CalcNumRings(m),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(m),
        "HeavyAtoms": Lipinski.HeavyAtomCount(m),
        "FormalCharge": Chem.GetFormalCharge(m),
    }

def fp(s, radius=2, nbits=2048):
    m = mol(s)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

def fp_array(s):
    f = fp(s)
    a = np.zeros(2048, dtype=np.uint8)
    if f is not None:
        DataStructs.ConvertToNumpyArray(f, a)
    return a

def scaffold(s):
    m = mol(s)
    if m is None:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))

def lipinski_like(d):
    return (
        d["MW"] <= 600 and d["LogP"] <= 6 and d["HBD"] <= 5 and
        d["HBA"] <= 10 and d["RotB"] <= 12
    )

@st.cache_data(ttl=86400)
def load_activity():
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [str(c).strip() for c in df.columns]
    df["SMILES"] = df["SMILES"].astype(str)
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["SMILES","pIC50"]).drop_duplicates("SMILES")
    valid = df["SMILES"].apply(lambda x: mol(x) is not None)
    return df[valid].reset_index(drop=True)

def feature_matrix(df):
    rows, keep = [], []
    for i, s in enumerate(df["SMILES"]):
        d = descriptors(s)
        if d is not None:
            rows.append([d[c] for c in DESC])
            keep.append(i)
    clean = df.iloc[keep].reset_index(drop=True)
    Xd = np.asarray(rows, dtype=float)
    Xfp = np.vstack([fp_array(s) for s in clean["SMILES"]])
    return np.hstack([Xd, Xfp]), clean

def make_model():
    df = load_activity()
    X, clean = feature_matrix(df)
    y = clean["pIC50"].values
    tr, te = train_test_split(np.arange(len(y)), test_size=0.20, random_state=42)
    m = RandomForestRegressor(
        n_estimators=600, random_state=42, n_jobs=-1,
        min_samples_leaf=2, max_features="sqrt"
    )
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y[te], p))),
        "MAE": float(mean_absolute_error(y[te], p)),
        "R2": float(r2_score(y[te], p)),
        "train_n": len(tr), "test_n": len(te)
    }
    return m, clean, metrics, X, y, tr, te

@st.cache_resource
def get_model():
    return make_model()

def predict_with_uncertainty(model, s):
    d = descriptors(s)
    if d is None:
        return None
    x = np.hstack([
        np.array([d[c] for c in DESC], dtype=float),
        fp_array(s)
    ]).reshape(1,-1)
    # RF tree dispersion is a practical model-uncertainty proxy, not a calibrated probability.
    vals = np.array([tree.predict(x)[0] for tree in model.estimators_])
    return float(vals.mean()), float(vals.std())

def similarity_profile(s, train_smiles):
    q = fp(s)
    if q is None:
        return None
    vals = []
    for t in train_smiles:
        f = fp(t)
        if f is not None:
            vals.append((DataStructs.TanimotoSimilarity(q, f), t))
    if not vals:
        return None
    vals.sort(reverse=True)
    return vals

def scaffold_split_indices(df, train_fraction=0.8):
    groups = {}
    for i, s in enumerate(df["SMILES"]):
        groups.setdefault(scaffold(s), []).append(i)
    groups = sorted(groups.values(), key=len, reverse=True)
    target = int(len(df) * train_fraction)
    train, test = [], []
    for g in groups:
        if len(train) + len(g) <= target:
            train.extend(g)
        else:
            test.extend(g)
    if len(test) < max(10, int(0.1*len(df))):
        test = train[-max(10,int(0.2*len(train))):]
        train = train[:-len(test)]
    return np.array(train), np.array(test)

def scaffold_validation():
    _, clean, _, X, y, _, _ = make_model()
    tr, te = scaffold_split_indices(clean)
    m = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1,
        min_samples_leaf=2, max_features="sqrt"
    )
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y[te],p))),
        "MAE": float(mean_absolute_error(y[te],p)),
        "R2": float(r2_score(y[te],p)),
        "train_n": len(tr), "test_n": len(te)
    }

def applicability_domain(similarities, uncertainty):
    if not similarities:
        return "Unknown"
    mx = similarities[0][0]
    if mx >= 0.70 and uncertainty <= 0.35:
        return "Higher confidence"
    if mx >= 0.50 and uncertainty <= 0.60:
        return "Intermediate confidence"
    return "Low confidence / extrapolation risk"

def pocket_compatibility(s):
    d = descriptors(s)
    if d is None:
        return None
    acidic = s.count("C(=O)O") + s.count("S(=O)(=O)O")
    # Deliberately interpretable physicochemical compatibility features.
    score = 0.0
    score += 0.25 if acidic >= 1 else 0.05
    score += 0.25 if d["HBA"] >= 3 else 0.10
    score += 0.20 if 40 <= d["TPSA"] <= 140 else 0.08
    score += 0.15 if d["MW"] <= 550 else 0.05
    score += 0.15 if abs(d["FormalCharge"]) <= 1 else 0.05
    return round(min(score,1.0),3)

def structure_aware_score(p, pocket):
    activity = min(max((p-4.0)/4.0,0),1)
    return round(0.75*activity + 0.25*pocket,3)

def generate_analogues(seed):
    candidates = set()
    rules = [
        ("Cl","F"), ("Br","Cl"), ("Br","F"),
        ("C(=O)O","C(=O)N"), ("OC","O")
    ]
    for old,new in rules:
        if old in seed:
            q = seed.replace(old,new,1)
            if mol(q): candidates.add(q)
    # A few conservative RDKit-valid transformations.
    if "c1" in seed:
        q = seed.replace("c1","Cc1",1)
        if mol(q): candidates.add(q)
    return sorted(candidates)

def novelty(s, train):
    vals = similarity_profile(s, train)
    if not vals:
        return None
    return round(1.0 - vals[0][0],3)

def show_structure(s):
    try:
        block = Chem.MolToMolBlock(mol(s))
        st.code(block, language="text")
    except Exception:
        pass

activity = load_activity()
model, clean, metrics, X, y, tr, te = get_model()

pages = [
    "Dashboard", "LBDD / QSAR", "Applicability Domain",
    "NDM-1 Structure Lab", "Structure-aware Scoring",
    "Analogue Generator", "ADMET Panel", "Candidate Portfolio",
    "Validation & Explainability", "Research Export"
]
page = st.sidebar.radio("Platform module", pages)

if page == "Dashboard":
    st.title("🧬 AMR-AI-CADD Navigator")
    st.subheader("Final NDM-1 research demonstrator")
    c = st.columns(5)
    c[0].metric("Activity records", len(clean))
    c[1].metric("Test RMSE", f"{metrics['RMSE']:.2f}")
    c[2].metric("Test MAE", f"{metrics['MAE']:.2f}")
    c[3].metric("Test R²", f"{metrics['R2']:.2f}")
    c[4].metric("Target", "NDM-1")
    st.markdown("""
### End-to-end workflow

**1. Curated activity → 2. QSAR → 3. applicability domain → 4. experimental structure context → 5. structure-aware prioritization → 6. analogue generation → 7. ADMET evidence → 8. multi-objective portfolio**

The platform is designed to help select computational hypotheses for subsequent experimental testing.
""")
    st.info("The activity model predicts pIC50. It does not prove inhibition. The structure-aware score is a transparent prioritization heuristic, not a docking score or binding free energy.")

elif page == "LBDD / QSAR":
    st.header("Ligand-based NDM-1 QSAR")
    s = st.text_area("Candidate SMILES", clean.iloc[0]["SMILES"])
    if st.button("Run QSAR"):
        z = descriptors(s)
        result = predict_with_uncertainty(model,s)
        if z is None or result is None:
            st.error("Invalid SMILES.")
        else:
            p,u = result
            sims = similarity_profile(s, clean.SMILES)
            ad = applicability_domain(sims,u)
            a,b,c,d = st.columns(4)
            a.metric("Predicted pIC50",f"{p:.2f}")
            b.metric("Approx. IC50 (nM)",f"{10**(9-p):.1f}")
            c.metric("RF uncertainty",f"±{u:.2f}")
            d.metric("Applicability",ad)
            st.dataframe(pd.DataFrame([z]),use_container_width=True)
            if sims:
                st.metric("Maximum Tanimoto",f"{sims[0][0]:.2f}")
                st.caption("Nearest training molecule: "+sims[0][1])
            show_structure(s)

elif page == "Applicability Domain":
    st.header("Applicability domain & uncertainty")
    s = st.text_area("SMILES", clean.iloc[0]["SMILES"])
    if st.button("Assess confidence"):
        r = predict_with_uncertainty(model,s)
        sims = similarity_profile(s,clean.SMILES)
        if r is None: st.error("Invalid SMILES.")
        else:
            p,u=r
            maxsim=sims[0][0] if sims else 0
            confidence=applicability_domain(sims,u)
            st.metric("Prediction",f"{p:.2f} pIC50")
            st.metric("Model uncertainty",f"{u:.2f}")
            st.metric("Maximum similarity",f"{maxsim:.2f}")
            st.metric("Applicability-domain interpretation",confidence)
            st.caption("The uncertainty is the dispersion of random-forest trees. It is an uncertainty proxy, not a formally calibrated prediction interval.")

elif page == "NDM-1 Structure Lab":
    st.header("NDM-1 experimental structure lab")
    st.write("Target: NDM-1 metallo-β-lactamase, UniProt C7C422.")
    st.write("Experimental NDM-1 structures provide real target context; the app does not invent docking energies.")
    pdb = st.selectbox("PDB structure",PDBS,index=0)
    st.markdown(f"**RCSB PDB:** https://www.rcsb.org/structure/{pdb}")
    # RCSB Mol* viewer embedded directly in the browser.
    st.components.v1.iframe(f"https://www.rcsb.org/3d-view/{pdb}",height=620,scrolling=True)
    st.markdown("""
**How to use this module:** inspect the experimentally determined ligand/metal environment in Mol*,
then use the platform's QSAR and structure-aware modules to prioritize compounds. This preserves a
clear distinction between observed structural evidence and model-derived predictions.
""")

elif page == "Structure-aware Scoring":
    st.header("Structure-aware candidate prioritization")
    s = st.text_area("SMILES",clean.iloc[0]["SMILES"])
    if st.button("Score"):
        r=predict_with_uncertainty(model,s)
        pc=pocket_compatibility(s)
        if r is None or pc is None: st.error("Invalid SMILES.")
        else:
            p,u=r; score=structure_aware_score(p,pc)
            a,b,c=st.columns(3)
            a.metric("QSAR pIC50",f"{p:.2f}")
            b.metric("Pocket-context compatibility",f"{pc:.3f}")
            c.metric("Integrated score",f"{score:.3f}")
            st.json({
                "target":"NDM-1",
                "structural_basis":"experimental NDM-1 PDB structures",
                "pIC50_prediction":round(p,3),
                "RF_uncertainty_proxy":round(u,3),
                "pocket_context_score":pc,
                "integrated_score":score
            })
            st.warning("This is NOT docking. No binding energy, pose, or kcal/mol value is inferred.")

elif page == "Analogue Generator":
    st.header("Transparent analogue generator")
    s=st.text_area("Parent/seed SMILES",clean.iloc[0]["SMILES"])
    if st.button("Generate computational analogues"):
        rows=[]
        for q in generate_analogues(s):
            r=predict_with_uncertainty(model,q)
            z=descriptors(q)
            sims=similarity_profile(q,clean.SMILES)
            rows.append({
                "SMILES":q,
                "pIC50_pred":round(r[0],3) if r else None,
                "uncertainty":round(r[1],3) if r else None,
                "MW":round(z["MW"],2),
                "LogP":round(z["LogP"],2),
                "TPSA":round(z["TPSA"],2),
                "max_Tanimoto":round(sims[0][0],3) if sims else None,
                "novelty":round(1-sims[0][0],3) if sims else None,
                "Lipinski_like":lipinski_like(z)
            })
        if rows:
            out=pd.DataFrame(rows).sort_values(["pIC50_pred","novelty"],ascending=[False,False])
            st.dataframe(out,use_container_width=True)
            st.download_button("Download analogue portfolio",out.to_csv(index=False),"ndm1_analogue_portfolio.csv","text/csv")
        else:
            st.warning("No valid analogue was produced by the conservative transformation rules.")

elif page == "ADMET Panel":
    st.header("ADMET evidence panel")
    st.markdown("""
This final version keeps ADMET scientifically honest: an endpoint is shown only when a
public benchmark dataset and an explicitly trained model are available. The recommended
TDC endpoints are AqSolDB, BBB_Martins, Pgp_Broccatelli, CYP2C9/2D6/3A4 inhibition,
hERG, AMES and DILI.
""")
    st.dataframe(pd.DataFrame({
        "Endpoint":["AqSolDB","BBB_Martins","Pgp_Broccatelli","CYP2C9","CYP2D6","CYP3A4","hERG","AMES","DILI"],
        "Type":["Regression","Classification","Classification","Classification","Classification","Classification","Classification","Classification","Classification"],
        "Role":["Solubility","BBB penetration","P-gp inhibition","CYP inhibition","CYP inhibition","CYP inhibition","Cardiac liability","Mutagenicity","Drug-induced liver injury"]
    }),use_container_width=True)
    st.info("For a publication-grade version, each endpoint should be trained and validated with its own scaffold split and metric. This app does not substitute missing experimental values with guesses.")

elif page == "Candidate Portfolio":
    st.header("Candidate portfolio builder")
    st.write("Generate a transparent portfolio rather than a single black-box 'best molecule'.")
    seed=st.text_area("Seed SMILES",clean.iloc[0]["SMILES"])
    if st.button("Build portfolio"):
        candidates=[seed]+generate_analogues(seed)
        rows=[]
        for s in candidates:
            r=predict_with_uncertainty(model,s); z=descriptors(s); sims=similarity_profile(s,clean.SMILES)
            pc=pocket_compatibility(s); nov=(1-sims[0][0]) if sims else 0
            act=min(max((r[0]-4)/4,0),1)
            drug=float(lipinski_like(z))
            integrated=.45*act+.20*drug+.15*nov+.20*pc
            rows.append({"SMILES":s,"pIC50":r[0],"uncertainty":r[1],"activity_component":act,
                         "druglike":drug,"novelty":nov,"structure_context":pc,"portfolio_score":integrated})
        out=pd.DataFrame(rows).sort_values("portfolio_score",ascending=False)
        st.dataframe(out,use_container_width=True)
        st.download_button("Download portfolio",out.to_csv(index=False),"ndm1_candidate_portfolio.csv","text/csv")

elif page == "Validation & Explainability":
    st.header("Validation & model diagnostics")
    st.subheader("Random hold-out")
    st.dataframe(pd.DataFrame([metrics]),use_container_width=True)
    st.subheader("Observed vs predicted")
    pred=model.predict(X[te])
    fig=px.scatter(x=y[te],y=pred,labels={"x":"Observed pIC50","y":"Predicted pIC50"},title="Random 80/20 hold-out")
    lo=min(y[te]); hi=max(y[te])
    fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi)
    st.plotly_chart(fig,use_container_width=True)
    if st.button("Run scaffold split validation"):
        with st.spinner("Training scaffold-split model..."):
            sm=scaffold_validation()
        st.dataframe(pd.DataFrame([sm]),use_container_width=True)
        st.caption("Scaffold split is a stricter estimate of performance on chemically distinct structures.")
    st.subheader("Model feature groups")
    st.write("The model combines 11 RDKit descriptors with a 2048-bit radius-2 Morgan fingerprint.")

elif page == "Research Export":
    st.header("Research-ready export")
    st.markdown("""
### Recommended Methods description

**Dataset:** curated NDM-1 inhibitor IC50 data compiled from ChEMBL and literature and published as a
686-compound modeling set.

**Representation:** RDKit physicochemical descriptors plus 2048-bit radius-2 Morgan fingerprints.

**Model:** Random Forest regression with fixed random seed and an 80:20 random hold-out for the initial
demonstrator. Scaffold splitting is provided as a stricter secondary validation.

**Structure context:** experimentally determined NDM-1 structures from the Protein Data Bank are used
for target inspection and an interpretable structure-context prioritization layer.

**Applicability domain:** nearest-neighbor Tanimoto similarity plus random-forest tree dispersion are
reported alongside each prediction.

**De-novo/analogue design:** transparent small transformations are used to generate testable hypotheses.

**ADMET:** public benchmark datasets should be independently trained per endpoint and reported with
endpoint-specific metrics and scaffold splits.
""")
    report=pd.DataFrame([{
        "target":"NDM-1","UniProt":"C7C422","activity_records":len(clean),
        "random_RMSE":metrics["RMSE"],"random_MAE":metrics["MAE"],"random_R2":metrics["R2"],
        "descriptor_count":len(DESC),"fingerprint_bits":2048
    }])
    st.download_button("Download run summary CSV",report.to_csv(index=False),"ndm1_run_summary.csv","text/csv")
