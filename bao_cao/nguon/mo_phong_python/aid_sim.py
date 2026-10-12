"""Mô phỏng Hệ thống Bơm Insulin Tự động (AID) – kiểm chứng mô hình SysML.

Các khối mô phỏng tương ứng với các block trong BDD:
  Patient (CoTheNguoiBenh), CGMSensor (CamBienCGM), Controller (BoDieuKhienAID),
  SafetySupervisor (BoGiamSatAnToan), PumpDriver (DriverBom), AlarmManager (QuanLyBaoDong).
"""
import json
import math
import sys
import numpy as np

TS = 5  # chu kỳ điều khiển (phút)
DT = 1  # bước tích phân (phút)

# ---------------- Thông số điều trị (theta) và giới hạn an toàn ----------------
THETA = dict(basal=0.8, ICR=12.0, ISF=45.0, GT=110.0, DIA=240.0, TP=75.0)
FIX_INVALID_RULE = True
LIMITS = dict(bmax=3.0, Bmax=10.0, IOBmax=8.0, TDDmax=80.0, step=0.05)


# ---------------- Đường cong insulin còn hoạt tính (OpenAPS exponential) ----------------
def iob_curve(t, td=240.0, tp=75.0):
    """Tỉ lệ insulin còn hoạt tính sau t phút (1 → 0)."""
    t = np.asarray(t, dtype=float)
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    S = 1 / (1 - a + (1 + a) * math.exp(-td / tau))
    v = 1 - S * (1 - a) * ((t ** 2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1)
    return np.where(t <= 0, 1.0, np.where(t >= td, 0.0, v))


# ---------------- Quá trình vật lý: mô hình Bergman mở rộng (1.1)–(1.5) ----------------
class Patient:
    VG = 120.0      # dL
    VI = 8400.0     # mL
    n = 0.16        # 1/min
    p2 = 0.022      # 1/min
    SG = 0.0025     # 1/min
    tauI = 45.0     # min
    tauS = 8.0      # min (trễ dịch kẽ)
    f = 0.9         # sinh khả dụng CHO

    def __init__(self, si_factor=1.0, basal_true=0.8, Gb=120.0, tau_m=50.0, seed=0):
        self.SI = 1.0e-3 * si_factor
        self.Gb = Gb
        self.tau_m = tau_m
        self.Ib = (basal_true / 60.0) * 1e6 / (self.VI * self.n)
        # trạng thái cân bằng tại basal thực
        u = basal_true / 60.0
        self.Isc1 = u * self.tauI
        self.Isc2 = self.Isc1
        self.I = self.Ib
        self.X = 0.0
        self.G = Gb
        self.Gs = Gb
        self.meals = []  # (t_start, CHO_g)
        self.exercise = []  # (t_start, duration, factor)
        self.day_si = []  # hệ số độ nhạy insulin theo ngày (stress, bệnh, chu kỳ nội tiết)

    def add_meal(self, t, cho):
        self.meals.append((t, cho))

    def Ra(self, t):
        r = 0.0
        for t0, cho in self.meals:
            s = t - t0
            if 0 <= s < 480:
                r += self.f * cho * 1000.0 * s / self.tau_m ** 2 * math.exp(-s / self.tau_m)
        return r / self.VG  # mg/dL/min

    def dawn(self, t):
        h = (t / 60.0) % 24
        return 0.12 * math.exp(-((h - 6.0) / 1.5) ** 2)  # mg/dL/min, hiện tượng bình minh

    def si_mult(self, t):
        m = self.day_si[int(t // 1440) % len(self.day_si)] if self.day_si else 1.0
        for t0, dur, fac in self.exercise:
            if t0 <= t < t0 + dur + 60:
                m *= fac if t < t0 + dur else 1 + (fac - 1) * (1 - (t - t0 - dur) / 60)
        return m

    def step(self, t, u_per_min):
        SI = self.SI * self.si_mult(t)
        dG = -(self.SG + self.X) * self.G + self.SG * self.Gb + self.Ra(t) + self.dawn(t)
        if self.G > 162:  # thải glucose qua thận (Hovorka)
            dG -= 0.003 * (self.G - 162)
        dX = -self.p2 * self.X + self.p2 * SI * (self.I - self.Ib)
        dIsc1 = u_per_min - self.Isc1 / self.tauI
        dIsc2 = (self.Isc1 - self.Isc2) / self.tauI
        dI = self.Isc2 / self.tauI * 1e6 / self.VI - self.n * self.I
        dGs = (self.G - self.Gs) / self.tauS
        self.G = max(self.G + DT * dG, 10.0)
        self.X += DT * dX
        self.Isc1 += DT * dIsc1
        self.Isc2 += DT * dIsc2
        self.I += DT * dI
        self.Gs += DT * dGs


class CGMSensor:
    def __init__(self, rng):
        self.rng = rng
        self.e = 0.0
        self.dropouts = []  # (t_start, t_end)

    def read(self, t, Gs):
        self.e = 0.7 * self.e + self.rng.normal(0, 6.5)
        y = Gs * (1 + self.rng.normal(0, 0.045)) + self.e
        for a, b in self.dropouts:
            if a <= t < b:
                return None
        return float(np.clip(y, 40, 400))


# ---------------- Bộ giám sát an toàn độc lập (REQ-S-02, REQ-S-03, REQ-S-05) ----------------
class SafetySupervisor:
    def __init__(self, limits):
        self.L = limits
        self.log = []  # (t, amount) – nhật ký liều độc lập với bộ điều khiển
        self.rejects = 0
        self.checked = 0
        self.reasons = {}

    def net_iob(self, t, basal_sched):
        if not self.log:
            return 0.0
        ts = np.array([a for a, _ in self.log])
        us = np.array([b for _, b in self.log])
        recent = ts > t - 240
        delivered = float(np.sum(us[recent] * iob_curve(t - ts[recent])))
        # insulin basal theo lịch đã "được kỳ vọng" trong cùng cửa sổ
        k = np.arange(0, 240, TS)
        expected = float(np.sum(basal_sched * TS / 60.0 * iob_curve(k)))
        return delivered - expected

    def tdd(self, t):
        return sum(b for a, b in self.log if a > t - 1440)

    def check(self, t, amount, rate_uph, bolus, basal_sched):
        """Trả về (approved, reason). amount = tổng U sẽ bơm trong chu kỳ."""
        self.checked += 1
        L = self.L
        if amount < 0 or not math.isfinite(amount):
            return self._rej("invalid")
        if rate_uph > L["bmax"] + 1e-9:
            return self._rej("rate>bmax")
        if bolus > L["Bmax"] + 1e-9:
            return self._rej("bolus>Bmax")
        iob = self.net_iob(t, basal_sched)
        extra = bolus + max(0.0, rate_uph - basal_sched) * TS / 60.0  # insulin vượt basal theo lịch
        if extra > 0 and iob + extra > L["IOBmax"] + 1e-9:
            return self._rej("IOB>IOBmax")
        if self.tdd(t) + amount > L["TDDmax"] + 1e-9:
            return self._rej("TDD>TDDmax")
        if abs(round(amount / L["step"]) * L["step"] - amount) > 1e-6:
            return self._rej("step")
        return True, "ok"

    def _rej(self, why):
        self.rejects += 1
        self.reasons[why] = self.reasons.get(why, 0) + 1
        return False, why

    def record(self, t, amount):
        if amount > 0:
            self.log.append((t, amount))


# ---------------- Bộ điều khiển AID ----------------
class Controller:
    def __init__(self, theta, limits, mode="AID", plgs=True):
        self.th = theta
        self.L = limits
        self.mode_kind = mode  # "AID" hoặc "SAP" (basal cố định + bolus calculator)
        self.plgs = plgs
        self.hist = []  # (t, y) mẫu hợp lệ
        self.deliv = []  # (t, U) liều đã bơm
        self.carbs = []  # (t, CHO) đã khai báo
        self.last_valid_t = 0.0
        self.mode = "AUTO" if mode == "AID" else "MANUAL"
        self.valid_streak = 0
        self.suspended = False
        self.suspend_since = None
        self.rate = theta["basal"]
        self.last_smb = -1e9
        self.alarms = []  # (t, code, priority)
        self.mode_log = []
        self.last_ghat = None

    # --- ước lượng ---
    def net_iob(self, t, horizon=0.0):
        k = np.arange(0, 240, TS)
        expected = float(np.sum(self.th["basal"] * TS / 60.0 * iob_curve(k + horizon)))
        if not self.deliv:
            return -expected
        ts = np.array([a for a, _ in self.deliv])
        us = np.array([b for _, b in self.deliv])
        m = ts > t - 240
        return float(np.sum(us[m] * iob_curve(t + horizon - ts[m]))) - expected

    def cob(self, t, horizon=0.0):
        c = 0.0
        for t0, g in self.carbs:
            s = t + horizon - t0
            c += g * max(0.0, 1 - s / 180.0) if s >= 0 else g
        return c

    def trend(self):
        pts = self.hist[-3:]
        if len(pts) < 2:
            return 0.0
        tt = np.array([p[0] for p in pts])
        yy = np.array([p[1] for p in pts])
        return float(np.polyfit(tt, yy, 1)[0])

    def predict30(self, t, G):
        th = self.th
        slope = self.trend()
        d_iob = self.net_iob(t) - self.net_iob(t, 30)
        d_cob = self.cob(t) - self.cob(t, 30)
        # tác động insulin/CHO tương lai được trừ phần đã phản ánh trong xu hướng
        return G + 30 * 0.7 * slope - 0.5 * th["ISF"] * d_iob + 0.5 * th["ISF"] / th["ICR"] * d_cob

    def bolus_calc(self, t, G, cho):
        th = self.th
        b = cho / th["ICR"] + (G - th["GT"]) / th["ISF"] - max(self.net_iob(t), 0.0)
        return min(max(0.0, b), self.L["Bmax"])

    def quant(self, x):
        return math.floor(x / self.L["step"] + 1e-9) * self.L["step"]

    def alarm(self, t, code, pri):
        self.alarms.append((t, code, pri))

    # --- một chu kỳ điều khiển ---
    def cycle(self, t, y, meal_cho=None):
        th, L = self.th, self.L
        bolus = 0.0
        valid = y is not None and 40 <= y <= 400
        if valid and self.hist and t - self.hist[-1][0] <= 10:
            if abs(y - self.hist[-1][1]) / (t - self.hist[-1][0]) > 4:
                valid = False
        if valid:
            self.hist.append((t, y))
            self.last_valid_t = t
            self.valid_streak += 1
        else:
            self.valid_streak = 0

        if self.mode_kind == "SAP":
            self.rate = th["basal"]
            if meal_cho:
                if valid:
                    bolus = self.bolus_calc(t, y, meal_cho)
                else:  # CGM không hợp lệ: chỉ phần bữa ăn, không hiệu chỉnh
                    bolus = max(0.0, meal_cho / th["ICR"] - max(self.net_iob(t), 0.0))
                bolus = min(bolus, L["Bmax"], max(0.0, L["IOBmax"] - max(self.net_iob(t), 0.0)))
                self.carbs.append((t, meal_cho))
            self.mode_log.append((t, self.mode))
            return self.rate, self.quant(bolus), y

        # chế độ dự phòng khi mất CGM ≥ 20 phút (REQ-S-04)
        if not valid:
            if t - self.last_valid_t >= 20 and self.mode == "AUTO":
                self.mode = "FALLBACK"
                self.rate = th["basal"]
                self.suspended = False
                self.alarm(t, "CGM_LOST", "MEDIUM")
            elif self.mode == "AUTO":
                if FIX_INVALID_RULE:
                    # quy tắc làm mịn sau mô phỏng lần 1: không vượt basal theo lịch;
                    # giá trị nghi ngờ < 70 mg/dL thì ngừng bơm (an toàn theo hướng ít insulin)
                    self.rate = 0.0 if (y is not None and y < 70) else min(self.rate, th["basal"])
                else:
                    self.rate = self.rate  # phiên bản 1: giữ basal hiện hành
            if meal_cho:  # bolus bữa ăn vẫn được đề xuất theo BGM – ở đây hoãn
                self.carbs.append((t, meal_cho))
                bolus = min(meal_cho / th["ICR"] - max(self.net_iob(t), 0), L["Bmax"],
                            L["IOBmax"] - max(self.net_iob(t), 0.0))
                bolus = self.quant(max(bolus, 0.0))
            self.mode_log.append((t, self.mode))
            return self.rate, max(bolus, 0.0), y
        if self.mode == "FALLBACK" and self.valid_streak >= 3:
            self.mode = "AUTO"

        G = y
        if self.mode != "AUTO":
            self.rate = th["basal"]
            if meal_cho:
                self.carbs.append((t, meal_cho))
                bolus = self.bolus_calc(t, G, meal_cho)
                bolus = min(bolus, max(0.0, L["IOBmax"] - max(self.net_iob(t), 0.0)))
            self.mode_log.append((t, self.mode))
            return self.rate, self.quant(bolus), G

        meal_bolus = 0.0
        if meal_cho:
            meal_bolus = self.bolus_calc(t, G, meal_cho)  # tính trước khi ghi COB mới
            self.carbs.append((t, meal_cho))
        Ghat = self.predict30(t, G)
        self.last_ghat = Ghat
        iob = self.net_iob(t)
        # PLGS (REQ-S-01): ngừng basal; bolus bữa ăn vẫn được đề xuất (đã trừ IOB, hiệu chỉnh âm)
        if self.plgs and (G < 70 or Ghat < 70):
            if not self.suspended:
                self.suspended, self.suspend_since = True, t
                self.alarm(t, "LOW_PREDICTED", "MEDIUM")
            if G < 54:
                self.alarm(t, "URGENT_LOW", "HIGH")
            self.rate = 0.0
            self.mode_log.append((t, "SUSPEND"))
            return 0.0, self.quant(min(meal_bolus, max(0.0, L["IOBmax"] - max(iob, 0.0)))), G
        if self.suspended:
            long_susp = t - self.suspend_since >= 120 and G >= 70
            if (G >= 80 and Ghat >= 80) or long_susp or meal_cho:
                self.suspended = False
            else:
                self.mode_log.append((t, "SUSPEND"))
                return 0.0, 0.0, G

        # luật điều khiển dự báo dựa trên IOB/COB (điều chỉnh basal tạm thời)
        eventual = G - th["ISF"] * iob + th["ISF"] / th["ICR"] * self.cob(t)
        target_bg = min(eventual, Ghat + 40)
        insulin_req = (target_bg - th["GT"]) / th["ISF"]
        self.rate = float(np.clip(th["basal"] + insulin_req, 0.0, L["bmax"]))
        bolus = meal_bolus
        # bolus hiệu chỉnh tự động nhỏ: chỉ khi không còn COB, Ĝ > 180 và IOB thấp
        if (not meal_cho and self.cob(t) < 1 and Ghat > 180 and G > 180 and insulin_req > 0.5
                and iob < 3.0 and t - self.last_smb >= 30):
            bolus = min(0.3 * insulin_req, 1.0)
            self.last_smb = t
        # áp giới hạn IOBmax (REQ-S-03)
        bolus = min(bolus, max(0.0, L["IOBmax"] - max(iob, 0.0)))
        room = L["IOBmax"] - iob - bolus
        if self.rate > th["basal"]:
            self.rate = min(self.rate, th["basal"] + max(0.0, room) * 60.0 / TS)
        self.mode_log.append((t, "AUTO"))
        return self.rate, self.quant(bolus), G


class PumpDriver:
    def __init__(self, rng, step=0.05):
        self.rng = rng
        self.step = step
        self.carry = 0.0
        self.occluded_from = None

    def plan(self, amount):
        return math.floor((amount + self.carry) / self.step + 1e-9) * self.step

    def deliver(self, amount):
        """Bơm amount (U) theo bước 0,05 U; trả về lượng thực tế (sai số cơ khí ≤ ±2 %)."""
        total = amount + self.carry
        steps = math.floor(total / self.step + 1e-9)
        self.carry = total - steps * self.step
        cmd = steps * self.step
        actual = cmd * (1 + float(np.clip(self.rng.normal(0, 0.012), -0.04, 0.04)))
        return cmd, actual


def basal_amount(rate):
    return rate * TS / 60.0


def run(patient, mode="AID", plgs=True, days=3, seed=0, meal_plan=None, announce_scale=None,
        dropouts=(), fault_inject=False, record=True):
    rng = np.random.default_rng(seed)
    cgm = CGMSensor(rng)
    cgm.dropouts = list(dropouts)
    ctrl = Controller(THETA, LIMITS, mode=mode, plgs=plgs)
    ssu = SafetySupervisor(LIMITS)
    drv = PumpDriver(rng)
    T = days * 1440
    announced = {}
    for t0, cho, ann in meal_plan:
        patient.add_meal(t0, cho)
        announced[int(t0 // TS) * TS] = ann
    G_trace, Y_trace, U_trace, M_trace, GH_trace, R_trace = [], [], [], [], [], []
    u_per_min = THETA["basal"] / 60.0
    viol = 0
    for t in range(0, T, DT):
        if t % TS == 0:
            y = cgm.read(t, patient.Gs)
            meal = announced.get(t)
            ctrl.last_ghat = None
            rate, bolus, _ = ctrl.cycle(t, y, meal)
            GH_trace.append(ctrl.last_ghat if ctrl.last_ghat is not None else np.nan)
            R_trace.append(rate)
            amount_req = basal_amount(rate) + bolus
            cmd_plan = drv.plan(amount_req)
            ok, _ = ssu.check(t, cmd_plan, rate, bolus, THETA["basal"])
            if not ok:
                cmd, actual = 0.0, 0.0
                ctrl.alarm(t, "CONTROL_FAULT", "HIGH")
            else:
                cmd, actual = drv.deliver(amount_req)
            if rate > LIMITS["bmax"] + 1e-9 and ok:
                viol += 1
            ssu.record(t, cmd)
            if cmd > 0:
                ctrl.deliv.append((t, cmd))
            u_per_min = actual / TS
            Y_trace.append(y if y is not None else np.nan)
            U_trace.append(cmd)
            M_trace.append(ctrl.mode_log[-1][1] if ctrl.mode_log else "AUTO")
        patient.step(t, u_per_min)
        if t % TS == 0:
            G_trace.append(patient.G)
    return dict(G=np.array(G_trace), Y=np.array(Y_trace, dtype=float), U=np.array(U_trace),
                mode=M_trace, alarms=ctrl.alarms, ssu=ssu, viol=viol,
                Ghat=np.array(GH_trace, dtype=float), rate=np.array(R_trace))


def metrics(G):
    G = np.asarray(G)
    return dict(
        mean=float(G.mean()),
        TIR=float(np.mean((G >= 70) & (G <= 180)) * 100),
        TBR70=float(np.mean(G < 70) * 100),
        TBR54=float(np.mean(G < 54) * 100),
        TAR180=float(np.mean(G > 180) * 100),
        TAR250=float(np.mean(G > 250) * 100),
        CV=float(G.std() / G.mean() * 100),
        Gmin=float(G.min()),
        Gmax=float(G.max()),
    )


def standard_meals(days, rng, err=0.15, forget_lunch_day=None):
    plan = []
    for d in range(days):
        base = d * 1440
        for hh, cho in [(7.5, 50), (12.5, 70), (19.0, 80)]:
            t0 = base + int((hh + rng.uniform(-0.25, 0.25)) * 60)
            t0 -= t0 % TS
            ann = round(cho * (1 + rng.normal(0, err)))
            plan.append((t0, cho, ann))
        t0 = base + int(15.5 * 60)
        plan.append((t0, 20, None))  # bữa phụ không khai báo
    if forget_lunch_day is not None:  # quên khai báo bữa trưa
        plan = [(a, b, None if (a // 1440 == forget_lunch_day and 11 * 60 < a % 1440 < 14 * 60) else c)
                for a, b, c in plan]
    return plan
