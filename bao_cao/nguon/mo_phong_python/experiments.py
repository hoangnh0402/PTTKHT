"""Các kịch bản kiểm chứng (Chương 5). Xuất results.json và hình vào thư mục out/."""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import aid_sim as A

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 100, "savefig.dpi": 220})
C_AID, C_SAP, C_ALT = "#1f4e79", "#c0504d", "#7f7f7f"
R = {}


def hours(n):
    return np.arange(n) * A.TS / 60.0


def band(ax):
    ax.axhspan(70, 180, color="#d9ead3", alpha=0.6, lw=0)
    ax.axhline(54, color="k", lw=0.6, ls=":")
    ax.axhline(250, color="k", lw=0.6, ls=":")


# ---------------- 1. Quần thể ảo: AID so với SAP ----------------
rng = np.random.default_rng(2026)
pats = [dict(si_factor=float(rng.uniform(0.8, 1.2)), basal_true=float(rng.uniform(0.64, 0.96)),
             Gb=float(rng.uniform(110, 130)), tau_m=float(rng.uniform(45, 65))) for _ in range(10)]
EXERCISE = [(1440 + 17 * 60, 45, 1.8)]  # ngày 2: tập thể dục 17:00–17:45
CARB_ERR = 0.20
DAY_SI = [list(np.random.default_rng(500 + i).uniform(0.75, 1.25, 3)) for i in range(10)]
pop = {"SAP": [], "AID": []}
traces = {"SAP": [], "AID": []}
auto_time, alarms_hi, mard_all, viol_total, rej_total = [], 0, [], 0, {}
plgs_cycles = plgs_viol = 0
for i, pp in enumerate(pats):
    plan = A.standard_meals(3, np.random.default_rng(100 + i), err=CARB_ERR, forget_lunch_day=2)
    for mode in ("SAP", "AID"):
        pt = A.Patient(**pp)
        pt.exercise = EXERCISE
        pt.day_si = DAY_SI[i]
        r = A.run(pt, mode=mode, days=3, seed=i, meal_plan=plan)
        m = A.metrics(r["G"])
        m["TDD"] = float(r["U"].sum() / 3)
        m12 = A.metrics(r["G"][:576])
        m["TIR_d12"], m["CV_d12"], m["TBR70_d12"] = m12["TIR"], m12["CV"], m12["TBR70"]
        m3 = A.metrics(r["G"][576:])
        m["TIR_d3"], m["CV_d3"], m["Gmax_d3"] = m3["TIR"], m3["CV"], m3["Gmax"]
        pop[mode].append(m)
        traces[mode].append(r)
        if mode == "AID":
            modes = np.array(r["mode"])
            auto_time.append(float(np.mean(np.isin(modes, ["AUTO", "SUSPEND"])) * 100))
            y, g = r["Y"], r["G"]
            ok = ~np.isnan(y)
            mard_all.append(float(np.mean(np.abs(y[ok] - g[ok]) / g[ok]) * 100))
            viol_total += r["viol"]
            cond = (r["Y"] < 70) | (r["Ghat"] < 70)
            plgs_cycles += int(np.sum(cond))
            plgs_viol += int(np.sum(cond & (r["rate"] > 0)))
            for k, v in r["ssu"].reasons.items():
                rej_total[k] = rej_total.get(k, 0) + v
R["patients"] = pats
R["pop"] = pop
R["pop_mean"] = {mode: {k: float(np.mean([m[k] for m in pop[mode]])) for k in pop[mode][0]} for mode in pop}
R["pop_sd"] = {mode: {k: float(np.std([m[k] for m in pop[mode]])) for k in pop[mode][0]} for mode in pop}
R["auto_time"] = float(np.mean(auto_time))
R["mard"] = float(np.mean(mard_all))
R["viol_nominal"] = viol_total
R["plgs_cycles"] = plgs_cycles
R["plgs_viol"] = plgs_viol
R["rej_nominal_aid"] = rej_total
R["rej_nominal_sap"] = {}
for r in traces["SAP"]:
    for k, v in r["ssu"].reasons.items():
        R["rej_nominal_sap"][k] = R["rej_nominal_sap"].get(k, 0) + v
print("pop", json.dumps(R["pop_mean"], indent=0)[:600])

# Hình 5.2 – đáp ứng 24 h (ngày 3, quên khai báo bữa trưa) của một bệnh nhân ảo
pid = 0
DAY = 2
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
n = 288
sl = slice(DAY * n, (DAY + 1) * n)
t = hours(n)
band(a1)
a1.plot(t, traces["SAP"][pid]["G"][sl], color=C_SAP, lw=1.4, label="SAP (basal cố định + bolus calculator)")
a1.plot(t, traces["AID"][pid]["G"][sl], color=C_AID, lw=1.6, label="AID (vòng kín lai)")
a1.plot(t, traces["AID"][pid]["Y"][sl], ".", color=C_AID, ms=2, alpha=0.5, label="Giá trị CGM (AID)")
plan = A.standard_meals(3, np.random.default_rng(100 + pid), err=CARB_ERR, forget_lunch_day=2)
for t0, cho, ann in plan:
    if DAY * 1440 <= t0 < (DAY + 1) * 1440:
        x = (t0 - DAY * 1440) / 60
        a1.annotate(f"{cho} g" + ("" if ann else "\n(không\nkhai báo)"), (x, 36), ha="center", fontsize=8)
        a1.axvline(x, color="k", lw=0.5, ls="--", alpha=0.5)
a1.set_ylabel("Glucose (mg/dL)")
a1.set_ylim(25, 430)
a1.legend(loc="upper left", fontsize=8.5, ncol=3)
a2.bar(t, traces["AID"][pid]["U"][sl], width=A.TS / 60, color=C_AID, label="Insulin bơm mỗi chu kỳ – AID")
a2.step(t, np.full(n, A.THETA["basal"] * A.TS / 60), color=C_SAP, lw=1, where="post", label="Basal theo lịch (SAP)")
a2.set_ylabel("Insulin (U/5 phút)")
a2.set_xlabel("Thời gian trong ngày thứ 3 (giờ)")
a2.set_ylim(0, 1.0)
a2.set_xticks(range(0, 25, 2))
a2.legend(loc="upper left", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_trace24h.png"))
plt.close(fig)
R["trace_day3"] = dict(SAP=A.metrics(traces["SAP"][pid]["G"][sl]), AID=A.metrics(traces["AID"][pid]["G"][sl]))

# Hình 5.3 – AGP quần thể
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, mode, col in ((axs[0], "SAP", C_SAP), (axs[1], "AID", C_AID)):
    M = np.array([r["G"].reshape(3, 288) for r in traces[mode]]).reshape(-1, 288)
    tt = hours(288)
    band(ax)
    p5, p25, p50, p75, p95 = np.percentile(M, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(tt, p5, p95, color=col, alpha=0.15, lw=0, label="5–95 %")
    ax.fill_between(tt, p25, p75, color=col, alpha=0.35, lw=0, label="25–75 %")
    ax.plot(tt, p50, color=col, lw=1.6, label="Trung vị")
    ax.set_title(mode + " – 10 bệnh nhân ảo × 3 ngày", fontsize=11)
    ax.set_xlabel("Giờ trong ngày")
    ax.set_xticks(range(0, 25, 4))
    ax.set_ylim(40, 350)
    ax.legend(fontsize=8, loc="upper left")
axs[0].set_ylabel("Glucose (mg/dL)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_agp.png"))
plt.close(fig)

# Hình 5.4 – thanh TIR
fig, ax = plt.subplots(figsize=(6.5, 3.6))
cats = [("TBR54", "< 54", "#8b0000"), ("TBR70_only", "54–69", "#e06666"), ("TIR", "70–180", "#6aa84f"),
        ("TAR180_only", "181–250", "#f6b26b"), ("TAR250", "> 250", "#e69138")]
for j, mode in enumerate(("SAP", "AID")):
    m = R["pop_mean"][mode]
    vals = dict(TBR54=m["TBR54"], TBR70_only=m["TBR70"] - m["TBR54"], TIR=m["TIR"],
                TAR180_only=m["TAR180"] - m["TAR250"], TAR250=m["TAR250"])
    left = 0
    for key, lab, col in cats:
        v = vals[key]
        ax.barh(j, v, left=left, color=col, edgecolor="white", label=lab if j == 0 else None)
        if v > 4:
            ax.text(left + v / 2, j, f"{v:.1f} %", ha="center", va="center", fontsize=9, color="k")
        left += v
ax.set_yticks([0, 1])
ax.set_yticklabels(["SAP", "AID"])
ax.set_xlim(0, 100)
ax.set_xlabel("Tỉ lệ thời gian (%) theo dải glucose (mg/dL)")
ax.legend(ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tir.png"))
plt.close(fig)

# phiên bản 1 của quy tắc dữ liệu không hợp lệ (trước khi làm mịn yêu cầu)
A.FIX_INVALID_RULE = False
c1 = v1 = 0
v1_examples = []
for i, pp in enumerate(pats):
    plan = A.standard_meals(3, np.random.default_rng(100 + i), err=CARB_ERR, forget_lunch_day=2)
    pt = A.Patient(**pp)
    pt.exercise = EXERCISE
    pt.day_si = DAY_SI[i]
    r = A.run(pt, mode="AID", days=3, seed=i, meal_plan=plan)
    cond = (r["Y"] < 70) | (r["Ghat"] < 70)
    c1 += int(np.sum(cond))
    v1 += int(np.sum(cond & (r["rate"] > 0)))
    for k in np.where(cond & (r["rate"] > 0))[0]:
        v1_examples.append(dict(patient=i + 1, t=int(k * A.TS), y_prev=float(r["Y"][k - 1]), y=float(r["Y"][k]),
                                rate=float(r["rate"][k])))
R["plgs_v1"] = dict(cycles=c1, viol=v1, examples=v1_examples)
A.FIX_INVALID_RULE = True

# ---------------- 2. TC-SIM-01: khai báo CHO thừa (REQ-S-01) ----------------
nominal = dict(si_factor=1.0, basal_true=0.8, Gb=120.0, tau_m=50.0)
plan1 = [(7 * 60 + 30, 50, 50), (12 * 60 + 30, 45, 90), (19 * 60, 70, 70)]
s1 = {}
for key, kw in (("AID", dict(mode="AID", plgs=True)), ("SAP", dict(mode="SAP"))):
    r = A.run(A.Patient(**nominal), days=1, seed=7, meal_plan=plan1, **kw)
    g = r["G"]
    win = slice(int(12.5 * 12), int(19 * 12))
    susp = sum(1 for m in r["mode"][win] if m == "SUSPEND") * A.TS
    s1[key] = dict(bolus=float(r["U"][int(12.5 * 12)]), Gmin=float(g[win].min()), t70=float(np.sum(g[win] < 70) * A.TS),
                   t54=float(np.sum(g[win] < 54) * A.TS), suspend_min=float(susp), r=r)
R["s1"] = {k: {kk: vv for kk, vv in v.items() if kk != "r"} for k, v in s1.items()}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
t = hours(288)
band(a1)
for key, col, ls in (("AID", C_AID, "-"), ("SAP", C_SAP, "-.")):
    a1.plot(t, s1[key]["r"]["G"], color=col, ls=ls, lw=1.5, label=key)
a1.axvline(12.5, color="k", lw=0.6, ls="--")
a1.annotate("Ăn 45 g nhưng\nkhai báo 90 g", (12.5, 255), xytext=(13.2, 255), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.6))
a1.set_ylabel("Glucose (mg/dL)")
a1.set_ylim(20, 300)
a1.legend(fontsize=9, loc="upper left")
mm = np.array([1 if m == "SUSPEND" else 0 for m in s1["AID"]["r"]["mode"]])
a2.fill_between(t, 0, mm, step="post", color="#f4cccc", label="Đang tạm ngưng (PLGS)")
a2.bar(t, s1["AID"]["r"]["U"], width=A.TS / 60, color=C_AID, label="Insulin bơm – AID")
a2.set_ylim(0, 1.0)
a2.set_ylabel("U/5 phút")
a2.set_xlabel("Thời gian trong ngày (giờ)")
a2.set_xticks(range(0, 25, 2))
a2.legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_s1.png"))
plt.close(fig)

# ---------------- 3. TC-SIM-02: mất CGM 45 phút (REQ-S-04) ----------------
plan2 = [(7 * 60 + 30, 50, 50), (12 * 60 + 30, 70, 70), (19 * 60, 80, 80)]
loss = (14 * 60, 14 * 60 + 45)
r2 = A.run(A.Patient(**nominal), mode="AID", days=1, seed=3, meal_plan=plan2, dropouts=[loss])
modes = r2["mode"]
idx_fb = next(i for i, m in enumerate(modes) if m == "FALLBACK")
idx_back = next(i for i in range(idx_fb, len(modes)) if modes[i] == "AUTO")
al = [a for a in r2["alarms"] if a[1] == "CGM_LOST"]
R["s2"] = dict(loss_start=loss[0], loss_end=loss[1], fallback_at=idx_fb * A.TS, auto_again=idx_back * A.TS,
               alarm=[(a[0], a[1], a[2]) for a in al],
               basal_during=float(np.mean(r2["U"][idx_fb:idx_back]) * 60 / A.TS),
               Gmin=float(r2["G"].min()), Gmax=float(r2["G"].max()))
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1, 0.5]})
t = hours(288)
band(a1)
a1.plot(t, r2["G"], color="k", lw=1.2, label="Glucose máu (mô hình)")
a1.plot(t, r2["Y"], ".", color=C_AID, ms=3, label="Giá trị CGM nhận được")
a1.axvspan(loss[0] / 60, loss[1] / 60, color="#fff2cc", alpha=0.9, label="Mất tín hiệu CGM")
a1.set_ylim(40, 260)
a1.set_ylabel("mg/dL")
a1.legend(fontsize=8.5, loc="upper left", ncol=3)
a2.bar(t, r2["U"], width=A.TS / 60, color=C_AID)
a2.set_ylabel("U/5 phút")
a2.set_ylim(0, 1.0)
code = {"AUTO": 2, "SUSPEND": 1, "FALLBACK": 0}
a3.step(t, [code.get(m, 2) for m in modes], where="post", color="k")
a3.set_yticks([0, 1, 2])
a3.set_yticklabels(["Dự phòng", "Tạm ngưng", "Tự động"], fontsize=8)
a3.set_xlabel("Thời gian trong ngày (giờ)")
a3.set_xlim(10, 18)
a3.set_xticks(range(10, 19, 1))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_s2.png"))
plt.close(fig)

# ---------------- 4. TC-FI-01: tiêm lỗi lệnh liều vào bộ giám sát (REQ-S-02/03/05) ----------------
frng = np.random.default_rng(99)
L = A.LIMITS
N = 10000
approved = rejected = unsafe_approved = 0
reasons = {}
for k in range(N):
    ssu = A.SafetySupervisor(L)
    t0 = 2000
    for j in range(frng.integers(0, 6)):  # lịch sử liều ngẫu nhiên trong 4 giờ
        ssu.log.append((t0 - int(frng.uniform(5, 235)), round(float(frng.uniform(0, 6)) / 0.05) * 0.05))
    if frng.random() < 0.5:
        rate = float(frng.uniform(0, 3.0))
        bolus = 0.0
    else:
        rate = float(frng.uniform(0, 15.0))
        bolus = float(frng.choice([0.0, frng.uniform(0, 30.0)]))
    if frng.random() < 0.3:
        bolus = round(bolus / 0.05) * 0.05
    amount = rate * A.TS / 60 + bolus
    if frng.random() < 0.8:
        amount = math.floor(amount / 0.05) * 0.05
    ok, why = ssu.check(t0, amount, rate, bolus, A.THETA["basal"])
    if ok:
        approved += 1
        iob = ssu.net_iob(t0, A.THETA["basal"])
        extra = bolus + max(0.0, rate - A.THETA["basal"]) * A.TS / 60
        safe = (rate <= L["bmax"] + 1e-9 and bolus <= L["Bmax"] + 1e-9 and
                (extra == 0 or iob + extra <= L["IOBmax"] + 1e-9))
        unsafe_approved += 0 if safe else 1
    else:
        rejected += 1
        reasons[why] = reasons.get(why, 0) + 1
R["fi"] = dict(N=N, approved=approved, rejected=rejected, unsafe_approved=unsafe_approved, reasons=reasons)

# ---------------- 5. Phát hiện tắc nghẽn và độ trễ báo động (REQ-S-06, REQ-A-01) ----------------
# Mô hình áp lực: mỗi 0,05 U bị đẩy vào đường truyền tắc làm áp lực tăng 12 kPa; ngưỡng 90 kPa.
def occl_detect_minutes(rate_uph):
    steps_needed = math.ceil(90 / 12)
    units = steps_needed * 0.05
    return units / rate_uph * 60


R["occl"] = {str(r): occl_detect_minutes(r) for r in (0.5, 0.8, 1.5, 3.0)}
R["occl_bolus_s"] = 8 * 0.05 / (1.5 / 60)  # bolus bơm 1,5 U/phút
# Độ trễ báo động: tác vụ quản lý báo động chu kỳ 250 ms, thời gian xử lý 2–20 ms,
# trễ khởi động còi/rung 50–300 ms, tải BLE đột biến chặn tối đa 1,5 s (xác suất 5 %).
arng = np.random.default_rng(5)
lat = []
for _ in range(10000):
    phase = arng.uniform(0, 0.25)
    proc = arng.uniform(0.002, 0.020)
    out = arng.uniform(0.05, 0.30)
    burst = arng.uniform(0, 1.5) if arng.random() < 0.05 else 0.0
    lat.append(phase + proc + out + burst)
lat = np.array(lat)
R["alarm_latency"] = dict(mean=float(lat.mean()), p99=float(np.percentile(lat, 99)), max=float(lat.max()))
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(lat, bins=60, color=C_AID)
ax.axvline(10, color=C_SAP, ls="--", lw=1.2, label="Giới hạn yêu cầu 10 s")
ax.axvline(lat.max(), color="k", ls=":", lw=1, label=f"Lớn nhất {lat.max():.2f} s")
ax.set_xscale("log")
ax.set_xlabel("Độ trễ từ khi phát hiện tới khi phát báo động (s, thang log)")
ax.set_ylabel("Số lần")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_latency.png"))
plt.close(fig)

# ---------------- 6. Độ chính xác bơm (REQ-P-01) ----------------
prng = np.random.default_rng(11)
drv = A.PumpDriver(prng)
errs = []
for _ in range(2000):
    cmd_req = round(float(prng.uniform(0.05, 10)) / 0.05) * 0.05
    drv.carry = 0.0
    cmd, act = drv.deliver(cmd_req)
    errs.append((act - cmd) / cmd * 100)
errs = np.array(errs)
R["pump_acc"] = dict(mean=float(errs.mean()), sd=float(errs.std()), maxabs=float(np.abs(errs).max()))

# Bolus calculator ví dụ (1.6)
R["bolus_example"] = dict(Bmeal=60 / 12, Bcorr=(200 - 110) / 45, B=min(max(0, 60 / 12 + (200 - 110) / 45 - 1.5), 10))

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k not in ("pop", "patients")}, ensure_ascii=False, indent=1))
