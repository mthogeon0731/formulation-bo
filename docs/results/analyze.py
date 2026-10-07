"""실측 캠페인(35회) 수치 계산. README의 모든 숫자가 여기서 나온다.

저장소 루트에서 실행:
    python docs/results/analyze.py

입력 : docs/results/data/tim_vlab_results.csv (35행, tc = k_app, d_cv = mediator)
출력 : docs/results/data/summary_metrics.json
       docs/results/data/cv_out_of_fold.csv
       docs/results/data/prospective_predictions.csv
       docs/results/data/hypervolume_by_iteration.csv

모델은 formulation_bo의 GP를 그대로 쓴다. 수치를 바꾸는 상수는 두지 않는다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from scipy.stats import t as student_t
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from formulation_bo import gem_tc, kd_viscosity  # noqa: E402
from formulation_bo.models import MediatorModel, ResidualObjectiveModel  # noqa: E402

DATA = HERE / "data"
CSV = DATA / "tim_vlab_results.csv"

K_FOLDS = 5
N_SPLIT_SEEDS = 20
M_MC = 200     # FormulationOptimizer의 M 기본값
HV_PAD = 0.08  # 캠페인 중 Pareto 차트가 쓴 정규화 여백


def k_model(seed: int, use_dcv: bool) -> ResidualObjectiveModel:
    """k_app 모델: GEM prior 잔차."""
    return ResidualObjectiveModel(use_dcv, gem_tc, "linear", seed)


def eta_model(seed: int, use_dcv: bool) -> ResidualObjectiveModel:
    """ln η 모델: Krieger-Dougherty prior 로그 잔차."""
    return ResidualObjectiveModel(use_dcv, kd_viscosity, "log", seed)


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["passed"] = df["passed"].astype(str).str.lower().eq("true")
    return df


# ---------------------------------------------------------------- 1. 측정 후 교차검증
def cross_validation(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """동일 25점(초기 15 + 실험군 10)에서 φ만 모델 vs φ + 실측 D_CV 모델.

    행 순서는 arm 알파벳순 → iteration (experimental 1–10, initial 1–15).
    KFold 분할이 순서에 의존하므로 고정한다.
    """
    d = df[df["arm"].isin(["experimental", "initial"]) & df["passed"]]
    d = d.sort_values(["arm", "iteration"]).reset_index(drop=True)
    phi, tc, dcv, eta = (d[c].to_numpy(float) for c in ("phi", "tc", "d_cv", "eta"))
    log_eta = np.log(eta)

    rows = []
    oof_without = np.zeros((N_SPLIT_SEEDS, len(d)))
    oof_with = np.zeros((N_SPLIT_SEEDS, len(d)))
    for seed in range(N_SPLIT_SEEDS):
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
        le_wo = np.zeros(len(d))
        le_w = np.zeros(len(d))
        for tr, te in kf.split(phi):
            oof_with[seed, te], _ = k_model(seed, True).fit(phi[tr], tc[tr], dcv[tr]).predict(phi[te], dcv[te])
            oof_without[seed, te], _ = k_model(seed, False).fit(phi[tr], tc[tr]).predict(phi[te])
            le_w[te], _ = eta_model(seed, True).fit(phi[tr], eta[tr], dcv[tr]).predict(phi[te], dcv[te])
            le_wo[te], _ = eta_model(seed, False).fit(phi[tr], eta[tr]).predict(phi[te])

        def rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        def r2(pred, y):
            return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))

        rows.append(dict(
            seed=seed,
            k_rmse_without=rmse(oof_without[seed], tc), k_rmse_with=rmse(oof_with[seed], tc),
            k_r2_without=r2(oof_without[seed], tc), k_r2_with=r2(oof_with[seed], tc),
            lneta_rmse_without=rmse(le_wo, log_eta), lneta_rmse_with=rmse(le_w, log_eta),
        ))
    s = pd.DataFrame(rows)
    s["k_red"] = (s.k_rmse_without - s.k_rmse_with) / s.k_rmse_without * 100
    s["lneta_red"] = (s.lneta_rmse_without - s.lneta_rmse_with) / s.lneta_rmse_without * 100

    mean_wo, mean_w = oof_without.mean(0), oof_with.mean(0)
    resid_wo = tc - mean_wo
    r_resid_dcv = pearsonr(resid_wo, dcv)

    # 부분상관: φ 1차 효과를 뺀 잔차끼리
    def lin_resid(y):
        A = np.column_stack([np.ones_like(phi), phi])
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]

    def partial_corr(y):
        """φ를 통제한 부분상관. 자유도 = n − 3 (통제 변수 1개)."""
        r = float(np.corrcoef(lin_resid(dcv), lin_resid(y))[0, 1])
        df_ = len(phi) - 3
        t = r * np.sqrt(df_ / (1 - r ** 2))
        return r, float(2 * student_t.sf(abs(t), df_))

    pc_k = partial_corr(tc)
    pc_eta = partial_corr(log_eta)

    def r2_all(pred):
        return float(1 - np.sum((tc - pred) ** 2) / np.sum((tc - tc.mean()) ** 2))

    out = dict(
        n=int(len(d)),
        k_rmse_without=float(s.k_rmse_without.mean()), k_rmse_with=float(s.k_rmse_with.mean()),
        k_reduction_pct_mean=float(s.k_red.mean()), k_reduction_pct_sd=float(s.k_red.std(ddof=1)),
        k_reduction_pct_min=float(s.k_red.min()), k_reduction_pct_max=float(s.k_red.max()),
        k_reduction_positive=int((s.k_red > 0).sum()),
        lneta_rmse_without=float(s.lneta_rmse_without.mean()), lneta_rmse_with=float(s.lneta_rmse_with.mean()),
        lneta_reduction_pct_mean=float(s.lneta_red.mean()), lneta_reduction_pct_sd=float(s.lneta_red.std(ddof=1)),
        lneta_reduction_pct_min=float(s.lneta_red.min()), lneta_reduction_pct_max=float(s.lneta_red.max()),
        lneta_reduction_positive=int((s.lneta_red > 0).sum()),
        k_r2_without_split_mean=float(s.k_r2_without.mean()), k_r2_with_split_mean=float(s.k_r2_with.mean()),
        k_r2_without_of_mean_pred=r2_all(mean_wo), k_r2_with_of_mean_pred=r2_all(mean_w),
        r_phi_only_residual_vs_dcv=float(r_resid_dcv[0]), p_phi_only_residual_vs_dcv=float(r_resid_dcv[1]),
        partial_r_k_dcv=float(pc_k[0]), partial_p_k_dcv=float(pc_k[1]),
        partial_r_lneta_dcv=float(pc_eta[0]), partial_p_lneta_dcv=float(pc_eta[1]),
        r_phi_dcv=float(pearsonr(phi, dcv)[0]),
    )
    oof = d[["arm", "iteration", "phi", "d_cv", "tc"]].copy()
    oof["pred_phi_only"] = mean_wo
    oof["pred_phi_dcv"] = mean_w
    return out, oof


# ---------------------------------------------------------------- 2. 추천 시점 사전 예측
def prospective(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """실험군 t회차 φ를 '초기 15 + 실험군 1…t−1'만으로 예측 (미래 자료 미사용)."""
    init = df[df.arm == "initial"]
    exp = df[df.arm == "experimental"].sort_values("iteration")
    rows = []
    for t in range(1, 11):
        tr = pd.concat([init, exp[exp.iteration < t]])
        row = exp[exp.iteration == t].iloc[0]
        x = np.array([row.phi])
        phi, tc, dcv = (tr[c].to_numpy(float) for c in ("phi", "tc", "d_cv"))

        mu0, s0 = k_model(seed, False).fit(phi, tc).predict(x)

        # ask()와 같은 경로: D_CV 예측분포를 뽑아 k 모델로 전파
        rng = np.random.default_rng(seed)
        f1 = MediatorModel(seed).fit(phi, dcv)
        f2 = k_model(seed, True).fit(phi, tc, dcv)
        mu1, s1 = f1.predict(x)
        samples = np.empty(M_MC)
        for m in range(M_MC):
            d_s = rng.normal(mu1, np.maximum(s1, 1e-9))
            mu2, s2 = f2.predict(x, d_s)
            samples[m] = rng.normal(mu2, np.maximum(s2, 1e-9))[0]

        mu3, _ = f2.predict(x, np.array([row.d_cv]))
        rows.append(dict(
            iteration=t, phi=row.phi, d_cv=row.d_cv, tc=row.tc,
            pred_phi_only=float(mu0[0]), sd_phi_only=float(s0[0]),
            pred_phi_dhat=float(samples.mean()), sd_phi_dhat=float(samples.std()),
            pred_phi_dmeas=float(mu3[0]),
        ))
    return pd.DataFrame(rows)


def prospective_summary(p: pd.DataFrame) -> dict:
    e0 = (p.tc - p.pred_phi_only).abs()
    e1 = (p.tc - p.pred_phi_dhat).abs()
    e2 = (p.tc - p.pred_phi_dmeas).abs()
    return dict(
        mae_phi_only=float(e0.mean()), mae_phi_dhat=float(e1.mean()), mae_phi_dmeas=float(e2.mean()),
        reduction_pct=float((e0.mean() - e1.mean()) / e0.mean() * 100),
        cover2s_phi_only=int((e0 <= 2 * p.sd_phi_only).sum()),
        cover2s_phi_dhat=int((e1 <= 2 * p.sd_phi_dhat).sum()),
        mean_sd_phi_only=float(p.sd_phi_only.mean()), mean_sd_phi_dhat=float(p.sd_phi_dhat.mean()),
        n_dhat_better=int((e1 < e0).sum()),
        wilcoxon_p=float(wilcoxon(e0, e1).pvalue),
    )


# ---------------------------------------------------------------- 3. 파레토 초체적
def domain_of(v, pad=HV_PAD):
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    p = (hi - lo) * pad
    return lo - p, hi + p


def pareto_mask(x, y):
    """x 최소화(η), y 최대화(k)."""
    n = len(x)
    return np.array([
        not any(j != i and x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i])
                for j in range(n))
        for i in range(n)
    ])


def hypervolume(x, y, xr, yr):
    """정규화 [0,1]² '좋음' 공간에서의 2D 초체적. 기준점 = 최저 k · 최고 η 모서리."""
    m = pareto_mask(x, y)
    gx = 1 - (x[m] - xr[0]) / (xr[1] - xr[0])
    gy = (y[m] - yr[0]) / (yr[1] - yr[0])
    order = np.argsort(gx)
    hv, prev = 0.0, 0.0
    for a, b in zip(gx[order], gy[order]):
        hv += (a - prev) * b
        prev = a
    return float(hv)


def hv_analysis(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    p = df[df.passed]
    init = p[p.arm == "initial"]

    def arm_hv(arm, upto=10, drop=None, pad=HV_PAD, log_eta=False):
        xs_all = np.log(p.eta.to_numpy()) if log_eta else p.eta.to_numpy()
        xr, yr = domain_of(xs_all, pad), domain_of(p.tc.to_numpy(), pad)
        a = p[(p.arm == arm) & (p.iteration <= upto)]
        if drop is not None:
            a = a[a.iteration != drop]
        s = pd.concat([init, a])
        x = np.log(s.eta.to_numpy()) if log_eta else s.eta.to_numpy()
        return hypervolume(x, s.tc.to_numpy(), xr, yr)

    curve = pd.DataFrame(dict(
        iteration=range(0, 11),
        hv_experimental=[arm_hv("experimental", t) for t in range(0, 11)],
        hv_control=[arm_hv("control", t) for t in range(0, 11)],
        hv_experimental_excl_iter8=[arm_hv("experimental", t, drop=8) for t in range(0, 11)],
    ))
    loo_exp = {int(i): arm_hv("experimental", drop=int(i)) for i in p[p.arm == "experimental"].iteration}
    loo_ctl = {int(i): arm_hv("control", drop=int(i)) for i in p[p.arm == "control"].iteration}
    pads = {f"{pad:.2f}": dict(experimental=arm_hv("experimental", pad=pad),
                               control=arm_hv("control", pad=pad),
                               experimental_excl_iter8=arm_hv("experimental", drop=8, pad=pad))
            for pad in (0.0, 0.04, 0.08, 0.15, 0.25)}
    out = dict(
        hv_initial_only=arm_hv("experimental", upto=0),
        hv_experimental=arm_hv("experimental"), hv_control=arm_hv("control"),
        hv_experimental_excl_iter8=arm_hv("experimental", drop=8),
        hv_experimental_upto7=arm_hv("experimental", upto=7), hv_control_upto7=arm_hv("control", upto=7),
        hv_log_eta=dict(experimental=arm_hv("experimental", log_eta=True),
                        experimental_excl_iter8=arm_hv("experimental", drop=8, log_eta=True),
                        control=arm_hv("control", log_eta=True)),
        hv_leave_one_out_experimental=loo_exp, hv_leave_one_out_control=loo_ctl,
        hv_pad_sensitivity=pads,
    )
    return out, curve


# ---------------------------------------------------------------- 4. 기타 관찰값
def misc(df: pd.DataFrame) -> dict:
    p = df[df.passed]
    w = p[(p.phi >= 0.46) & (p.phi <= 0.49) & p.d_cv.notna()]
    lo, hi = w[w.d_cv <= 0.50], w[w.d_cv >= 0.575]
    e8 = p[(p.arm == "experimental") & (p.iteration == 8)].iloc[0]
    i15 = p[(p.arm == "initial") & (p.iteration == 15)].iloc[0]
    kd_ratio = float(kd_viscosity(e8.phi)[0] / kd_viscosity(i15.phi)[0])

    summary = {}
    for arm in ("initial", "experimental", "control"):
        a = df[df.arm == arm]
        ap = a[a.passed]
        summary[arm] = dict(
            n=int(len(a)), n_pass=int(len(ap)), n_fail=int((~a.passed).sum()),
            phi=[float(a.phi.min()), float(a.phi.max())],
            tc=[float(ap.tc.min()), float(ap.tc.max())],
            eta=[float(ap.eta.min()), float(ap.eta.max())],
            d_cv=[float(ap.d_cv.min()), float(ap.d_cv.max())] if ap.d_cv.notna().any() else None,
        )
    return dict(
        same_phi_window=dict(
            k_low_dcv=float(lo.tc.mean()), k_high_dcv=float(hi.tc.mean()),
            eta_low_dcv=float(lo.eta.mean()), eta_high_dcv=float(hi.eta.mean()),
            phi_low_dcv=float(lo.phi.mean()), phi_high_dcv=float(hi.phi.mean()),
            k_change_pct=float((hi.tc.mean() - lo.tc.mean()) / lo.tc.mean() * 100),
            eta_change_pct=float((hi.eta.mean() - lo.eta.mean()) / lo.eta.mean() * 100),
        ),
        iter8_vs_initial_best=dict(
            eta_change_pct=float((e8.eta - i15.eta) / i15.eta * 100),
            kd_expected_change_pct=(kd_ratio - 1) * 100,
        ),
        gem_at_0p54=float(gem_tc(0.54)[0]),
        thickness_mm=[float(p.thickness_mm.min()), float(p.thickness_mm.max())],
        arm_summary=summary,
    )


def main() -> None:
    df = load()
    cv, oof = cross_validation(df)
    pro = {s: prospective(df, s) for s in range(10)}
    pro_sum = {s: prospective_summary(v) for s, v in pro.items()}
    hv, curve = hv_analysis(df)

    def rng_of(key):
        vals = [pro_sum[s][key] for s in pro_sum]
        return [min(vals), max(vals)]

    metrics = dict(
        cross_validation=cv,
        prospective_seed0=pro_sum[0],
        prospective_seed0_9_range={k: rng_of(k) for k in pro_sum[0]},
        hypervolume=hv,
        misc=misc(df),
    )
    (DATA / "summary_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    oof.to_csv(DATA / "cv_out_of_fold.csv", index=False, float_format="%.4f")
    pro[0].to_csv(DATA / "prospective_predictions.csv", index=False, float_format="%.4f")
    curve.to_csv(DATA / "hypervolume_by_iteration.csv", index=False, float_format="%.4f")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
