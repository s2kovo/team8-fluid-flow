import math
import streamlit as st
import matplotlib.pyplot as plt

G = 9.81
BASE_PARAMS = {"rho": 998.0, "mu": 9.82e-4, "L": 10.0, "D": 0.0266446,
               "epsilon": 4.572e-5, "K_total": 1.4}


def validate_params(params):
    for key in BASE_PARAMS:
        if key not in params or not math.isfinite(params[key]):
            raise ValueError(f"{key}: 유효한 숫자를 입력하세요.")
    for key in ("rho", "mu", "L", "D"):
        if params[key] <= 0:
            raise ValueError(f"{key}는 0보다 커야 합니다.")
    for key in ("epsilon", "K_total"):
        if params[key] < 0:
            raise ValueError(f"{key}는 0 이상이어야 합니다.")


def get_flow_regime(Re):
    return "Laminar" if Re < 2100 else "Transition" if Re <= 4000 else "Turbulent"


def swamee_jain(Re, epsilon, D):
    x = epsilon / (3.7 * D) + 5.74 / Re ** 0.9
    return 0.25 / math.log10(x) ** 2


def friction_factor(Re, epsilon, D, method="colebrook"):
    if method not in ("colebrook", "swamee_jain"):
        raise ValueError("알 수 없는 마찰계수 계산방법입니다.")
    if Re <= 0 or D <= 0 or epsilon < 0:
        raise ValueError("마찰계수 입력값이 잘못되었습니다.")
    if Re < 2100:
        return {"f": 64 / Re, "iterations": 0, "converged": True,
                "method": "Laminar: 64/Re", "warning": None}

    def colebrook(re_val):
        f_old = swamee_jain(re_val, epsilon, D)
        for iteration in range(1, 101):
            x = epsilon / (3.7 * D) + 2.51 / (re_val * math.sqrt(f_old))
            f_new = 1 / (-2 * math.log10(x)) ** 2
            if abs(f_new - f_old) < 1e-7:
                return f_new, iteration, True
            f_old = f_new
        return f_old, 100, False

    if Re > 4000:
        if method == "swamee_jain":
            return {"f": swamee_jain(Re, epsilon, D), "iterations": 0,
                    "converged": True, "method": "Swamee–Jain (Explicit)", "warning": None}
        f, n, ok = colebrook(Re)
        return {"f": f, "iterations": n, "converged": ok,
                "method": "Colebrook-White (Iterative)",
                "warning": None if ok else "Colebrook 수렴 실패"}

    f_low = 64 / 2100
    if method == "swamee_jain":
        f_high, n, ok = swamee_jain(4000, epsilon, D), 0, True
    else:
        f_high, n, ok = colebrook(4000)
    f = f_low + (Re - 2100) / 1700 * (f_high - f_low)
    return {"f": f, "iterations": n, "converged": ok,
            "method": "Transition interpolation",
            "warning": "천이영역: 마찰계수는 선형보간한 추정값입니다."}


def calculate_flow(Q, params, method="colebrook"):
    validate_params(params)
    if not math.isfinite(Q) or Q <= 0:
        raise ValueError("Q는 유한한 양수여야 합니다.")
    rho, mu, L, D, epsilon, K_total = (params[k] for k in BASE_PARAMS)
    A = math.pi * D ** 2 / 4
    V = Q / A
    Re = rho * V * D / mu
    friction = friction_factor(Re, epsilon, D, method)
    if not friction["converged"]:
        raise RuntimeError("마찰계수 계산이 수렴하지 않았습니다.")
    f = friction["f"]
    velocity_head = V ** 2 / (2 * G)
    h_major = f * L / D * velocity_head
    h_minor = K_total * velocity_head
    return {"Q": Q, "A": A, "V": V, "Re": Re, "regime": get_flow_regime(Re),
            "relative_roughness": epsilon / D, "f": f,
            "friction_iterations": friction["iterations"],
            "friction_converged": friction["converged"],
            "friction_method": friction["method"], "friction_warning": friction["warning"],
            "velocity_head": velocity_head, "h_major": h_major,
            "h_minor": h_minor, "h_total": h_major + h_minor}


def solve_forward(delta_z, params, tolerance=1e-8, max_iterations=100, method="colebrook"):
    validate_params(params)
    if not math.isfinite(delta_z) or delta_z <= 0:
        raise ValueError("고도차는 유한한 양수여야 합니다.")
    q_low, q_high = 0.0, 1e-6
    for _ in range(100):
        if calculate_flow(q_high, params, method)["h_total"] >= delta_z:
            break
        q_high *= 2
    else:
        raise RuntimeError("유량 탐색 범위를 찾지 못했습니다.")
    for iteration in range(1, max_iterations + 1):
        q_mid = (q_low + q_high) / 2
        result = calculate_flow(q_mid, params, method)
        residual = result["h_total"] - delta_z
        if abs(residual) < tolerance:
            return {**result, "delta_z": delta_z, "residual": residual,
                    "solver_iterations": iteration, "solver_converged": True}
        if residual > 0:
            q_high = q_mid
        else:
            q_low = q_mid
    return {**result, "delta_z": delta_z, "residual": residual,
            "solver_iterations": max_iterations, "solver_converged": False}


def solve_inverse(Q_target, params):
    result = calculate_flow(Q_target, params)
    return {**result, "Q_target": Q_target, "delta_z_required": result["h_total"]}


def sweep_diameter_natural_flow(values, delta_z, params):
    rows = []
    for D in values:
        p = {**params, "D": D}
        r = solve_forward(delta_z, p)
        rows.append({"D_mm": D * 1000, "Q_Ls": r["Q"] * 1000,
                     "Re": r["Re"], "f": r["f"]})
    return rows


def sweep_length_natural_flow(values, delta_z, params):
    rows = []
    for L in values:
        r = solve_forward(delta_z, {**params, "L": L})
        rows.append({"L": L, "Q_Ls": r["Q"] * 1000, "Re": r["Re"], "f": r["f"]})
    return rows


def sweep_roughness_natural_flow(values, delta_z, params):
    rows = []
    for epsilon in values:
        r = solve_forward(delta_z, {**params, "epsilon": epsilon})
        rows.append({"epsilon": epsilon, "Q_Ls": r["Q"] * 1000,
                     "Re": r["Re"], "f": r["f"]})
    return rows


def sweep_diameter_inverse(values, Q_target, params):
    rows = []
    for D in values:
        r = solve_inverse(Q_target, {**params, "D": D})
        rows.append({"D_mm": D * 1000, "delta_z": r["delta_z_required"],
                     "Re": r["Re"], "f": r["f"]})
    return rows


def validation_crosscheck(delta_z, params):
    main = solve_forward(delta_z, params, method="colebrook")
    ref = solve_forward(delta_z, params, method="swamee_jain")
    if not main["solver_converged"] or not ref["solver_converged"]:
        raise RuntimeError("검증 유량 계산이 수렴하지 않았습니다.")
    error = abs(main["Q"] - ref["Q"]) / abs(ref["Q"]) * 100
    return main, ref, error


def validation_ideal_outlet(delta_z, params, scales, K_free_outlet=0.4):
    """별도 모델: 큰 저장조 자유표면 -> 대기 중 자유출구.
    Δz = [1 + alpha*(f L/D + K_free_outlet)] V²/(2g).
    K_free_outlet은 자유출구 이전의 입구/밸브/굴곡 손실만 포함한다.
    기존 두 저장조의 출구손실 K_exit=1은 포함하지 않는다.
    """
    validate_params(params)
    if not math.isfinite(K_free_outlet) or K_free_outlet < 0:
        raise ValueError("자유출구 손실계수는 유한한 0 이상의 값이어야 합니다.")
    if delta_z <= 0 or not math.isfinite(delta_z):
        raise ValueError("고도차는 유한한 양수여야 합니다.")
    area = math.pi * params["D"] ** 2 / 4
    ideal_q = area * math.sqrt(2 * G * delta_z)
    rows = []
    for alpha in scales:
        if not 0 <= alpha <= 1:
            raise ValueError("손실 배율은 0~1 사이여야 합니다.")
        if alpha == 0:
            q = ideal_q
        else:
            low, high = 0.0, ideal_q
            for _ in range(100):
                q = (low + high) / 2
                r = calculate_flow(q, params)
                head = (1 + alpha * (r["f"] * params["L"] / params["D"] + K_free_outlet)) * r["velocity_head"]
                if head > delta_z:
                    high = q
                else:
                    low = q
                if abs(head - delta_z) < 1e-9:
                    break
        rows.append({"loss_scale": alpha, "Q_Ls": q * 1000,
                     "relative_difference_pct": abs(q - ideal_q) / ideal_q * 100})
    return ideal_q, rows


def draw_chart(rows, x_key, y_key, xlabel, ylabel, title, ideal=None):
    fig, ax = plt.subplots()
    ax.plot([r[x_key] for r in rows], [r[y_key] for r in rows], marker="o")
    if ideal is not None:
        ax.axhline(ideal, linestyle="--", label="Ideal Bernoulli")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def verify_roundtrip(q_target, base_params, selected):
    """각 관경에서 역설계 Δz를 정방향에 대입해 재현성을 검증한다."""
    rows = []
    for name in selected:
        D = SCH40_INNER_DIAMETERS_MM[name] / 1000
        p = {**base_params, "D": D}
        inverse = solve_inverse(q_target, p)
        dz = inverse["delta_z_required"]
        forward = solve_forward(dz, p, tolerance=1e-10, max_iterations=160)
        if not forward["solver_converged"]:
            raise RuntimeError(f"{name}: 정방향 계산이 수렴하지 않았습니다.")
        rows.append({
            "표준관경": name, "내경 (mm)": D * 1000,
            "목표 유량 (L/s)": q_target * 1000,
            "역설계 Δz (m)": dz,
            "정방향 재계산 Q (L/s)": forward["Q"] * 1000,
            "유량 상대오차 (%)": abs(forward["Q"] - q_target) / q_target * 100,
            "손실수두 잔차 (m)": forward["h_total"] - dz,
            "정방향 반복 횟수": forward["solver_iterations"],
        })
    return rows


def recommend_pipe(q_target, max_head, max_velocity, selected, base_params):
    """모든 선택 관경을 검사한 후, 제약조건을 만족하는 최소 내경 반환."""
    rows = compare_standard_pipes(q_target, max_head, max_velocity, selected, base_params)
    eligible = [r for r in rows if r["최종 판정"] == "충족"]
    return min(eligible, key=lambda r: r["내경 D (mm)"]) if eligible else None


def sensitivity_sweep(variable, values, q_target, max_head, max_velocity, selected, base_params):
    """한 번에 한 변수만 변경해 표준관경 추천 결과를 추적."""
    rows = []
    for value in values:
        q, head, p = q_target, max_head, base_params.copy()
        if variable == "head":
            head = value
        elif variable == "flow":
            q = value / 1000  # L/s -> m³/s
        elif variable == "length":
            p["L"] = value
        else:
            raise ValueError("알 수 없는 민감도 분석 변수")
        chosen = recommend_pipe(q, head, max_velocity, selected, p)
        rows.append({
            "변수 값": value,
            "추천 관경": chosen["표준관경"] if chosen else "적합 관경 없음",
            "추천 내경 (mm)": chosen["내경 D (mm)"] if chosen else None,
            "필요 Δz (m)": chosen["필요 Δz (m)"] if chosen else None,
            "유속 (m/s)": chosen["유속 V (m/s)"] if chosen else None,
        })
    return rows


def draw_sensitivity(rows, xlabel, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = [r["변수 값"] for r in rows]
    ys = [float("nan") if r["추천 내경 (mm)"] is None else r["추천 내경 (mm)"] for r in rows]
    ax.step(xs, ys, where="post", linewidth=2)
    ax.scatter(xs, ys, s=15)
    ax.set(xlabel=xlabel, ylabel="Minimum eligible Sch40 inside diameter (mm)", title=title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    if any(r["추천 내경 (mm)"] is None for r in rows):
        st.info("그래프의 빈 구간은 선택된 표준관경 중 설계조건을 만족하는 관경이 없는 경우입니다.")


# ASME B36.10M Schedule 40 / 40S nominal sizes: reference inner diameters.
# Values below are representative dimensions; check vendor tolerances before procurement.
SCH40_INNER_DIAMETERS_MM = {
    "1/2 in (DN15)": 15.80,
    "3/4 in (DN20)": 20.93,
    "1 in (DN25)": 26.64,
    "1-1/4 in (DN32)": 35.05,
    "1-1/2 in (DN40)": 40.89,
    "2 in (DN50)": 52.50,
    "2-1/2 in (DN65)": 62.71,
    "3 in (DN80)": 77.93,
    "4 in (DN100)": 102.26,
}


def compare_standard_pipes(q_target, max_head, max_velocity, selected, base_params):
    """Return hydraulic comparisons and eligible options for selected standard IDs."""
    results = []
    for name in selected:
        diameter_mm = SCH40_INNER_DIAMETERS_MM[name]
        r = solve_inverse(q_target, {**base_params, "D": diameter_mm / 1000.0})
        head_ok = r["delta_z_required"] <= max_head + 1e-9
        velocity_ok = r["V"] <= max_velocity + 1e-9
        results.append({
            "표준관경": name,
            "내경 D (mm)": diameter_mm,
            "필요 Δz (m)": r["delta_z_required"],
            "유속 V (m/s)": r["V"],
            "Re": r["Re"],
            "유동영역": r["regime"],
            "Darcy f": r["f"],
            "주손실 (m)": r["h_major"],
            "부차손실 (m)": r["h_minor"],
            "고도차 충족": head_ok,
            "유속 조건 충족": velocity_ok,
            "최종 판정": "충족" if head_ok and velocity_ok else "미충족",
        })
    return sorted(results, key=lambda row: row["내경 D (mm)"])



# ============================================================
# Streamlit 웹 화면: 계산 함수 정의가 모두 끝난 다음에 배치
# ============================================================
st.set_page_config(page_title="team 7 Natural Flow Design", page_icon="💧", layout="wide")
st.title("두 저장조 사이 자연유동 및 최소 고도차 설계")
st.caption("team 7 | Mechanical Energy Balance · Colebrook · Inverse Design · Validation")

st.sidebar.header("계산 조건")
rho = st.sidebar.number_input("밀도 ρ (kg/m³)", min_value=1e-6, value=998.0, format="%.4f")
mu = st.sidebar.number_input("점도 μ (Pa·s)", min_value=1e-9, value=9.82e-4, format="%.8f")
L = st.sidebar.number_input("관 길이 L (m)", min_value=1e-4, value=10.0, format="%.4f")
D = st.sidebar.number_input("관 내경 D (m)", min_value=1e-6, value=0.0266446, format="%.7f")
epsilon = st.sidebar.number_input("절대조도 ε (m)", min_value=0.0, value=4.572e-5, format="%.8f")
K_total = st.sidebar.number_input("ΣK", min_value=0.0, value=1.4, format="%.4f")
params = {"rho": rho, "mu": mu, "L": L, "D": D, "epsilon": epsilon, "K_total": K_total}

with st.expander("적용한 지배방정식과 가정"):
    st.write("정상상태 · 비압축성 · 양쪽 저장조 대기압 · 자유표면 속도 무시 · 펌프 없음 · Darcy 마찰계수")
    st.latex(r"\Delta z=\left[f_D\frac{L}{D}+\sum K\right]\frac{V^2}{2g}")
    st.latex(r"V=\frac{4Q}{\pi D^2},\quad Re=\frac{\rho VD}{\mu}")

tab_forward, tab_inverse, tab_roundtrip, tab_sweep, tab_design, tab_sensitivity, tab_validation = st.tabs(
    ["정방향 계산", "역설계", "정·역방향 재검증", "자연유량 곡선", "D-Δz 설계 비교", "설계 민감도 분석", "계산 결과 검증"]
)

with tab_forward:
    st.subheader("정방향 계산 : Δz → Q")
    dz = st.number_input("실제 고도차 Δz (m)", min_value=1e-4, value=3.0, format="%.4f", key="forward_dz")
    if st.button("정방향 계산", type="primary"):
        try:
            r = solve_forward(dz, params)
            if not r["solver_converged"]:
                st.warning("유량 반복계산이 수렴하지 않았습니다.")
            else:
                st.success("계산이 완료되었습니다.")
                a, b, c, d = st.columns(4)
                a.metric("자연유량 Q", f"{r['Q']*1000:.4f} L/s")
                b.metric("유속 V", f"{r['V']:.4f} m/s")
                c.metric("Reynolds 수", f"{r['Re']:.3e}")
                d.metric("유동영역", r["regime"])
                st.divider()
                a, b, c = st.columns(3)
                a.metric("Darcy f", f"{r['f']:.6f}")
                b.metric("주손실", f"{r['h_major']:.4f} m")
                c.metric("부차손실", f"{r['h_minor']:.4f} m")
                st.metric("총 손실수두", f"{r['h_total']:.6f} m")
                st.write("**마찰계수 계산법:**", r["friction_method"])
                st.write("**Colebrook 반복횟수:**", r["friction_iterations"])
                st.write("**Colebrook 수렴:**", r["friction_converged"])
                st.write("**유량 Solver 반복횟수:**", r["solver_iterations"])
                st.write("**유량 Solver 수렴:**", r["solver_converged"])
                st.write("**잔차:**", f"{r['residual']:.3e} m")
                if r["friction_warning"]:
                    st.warning(r["friction_warning"])
        except Exception as e:
            st.error(str(e))

with tab_inverse:
    st.subheader("역설계 : 목표유량 → 최소 Δz")
    qt = st.number_input("목표유량 Q target (m³/s)", min_value=1e-7, value=0.0013, format="%.8f", key="inverse_q")
    if st.button("필요 고도차 계산", type="primary"):
        try:
            r = solve_inverse(qt, params)
            st.success("역설계 계산이 완료되었습니다.")
            a, b, c, d = st.columns(4)
            a.metric("필요 고도차 Δz", f"{r['delta_z_required']:.4f} m")
            b.metric("유속 V", f"{r['V']:.4f} m/s")
            c.metric("Reynolds 수", f"{r['Re']:.3e}")
            d.metric("유동영역", r["regime"])
            a, b, c = st.columns(3)
            a.metric("Darcy f", f"{r['f']:.6f}")
            b.metric("주손실", f"{r['h_major']:.4f} m")
            c.metric("부차손실", f"{r['h_minor']:.4f} m")
        except Exception as e:
            st.error(str(e))

with tab_roundtrip:
    st.subheader("정·역방향 재검증: 역설계 Δz → 정방향 Q 재계산")
    st.write("각 표준관경에서 목표 유량으로 필요 고도차를 계산한 뒤, 동일한 조건의 정방향 계산에 다시 대입합니다.")
    check_q_ls = st.number_input("재검증 목표 유량 (L/s)", min_value=0.001, value=1.30,
                                 step=0.10, format="%.3f", key="roundtrip_q")
    check_selected = st.multiselect("재검증할 Sch40 표준관경", list(SCH40_INNER_DIAMETERS_MM),
                                    default=list(SCH40_INNER_DIAMETERS_MM), key="roundtrip_pipes")
    st.caption("밀도·점도·길이·조도·ΣK는 사이드바 조건을 사용합니다. 동일 계산식을 이용한 수치적 일관성 검증입니다.")
    if st.button("표준관경별 정·역방향 재검증 실행", type="primary", key="roundtrip_run"):
        if not check_selected:
            st.warning("검증할 표준관경을 선택하세요.")
        else:
            try:
                check_rows = verify_roundtrip(check_q_ls / 1000, params, check_selected)
                st.session_state["roundtrip_results"] = check_rows
                st.session_state["roundtrip_settings"] = (check_q_ls, params.copy(), check_selected[:])
            except Exception as e:
                st.error(f"재검증 계산 오류: {e}")
    if "roundtrip_results" in st.session_state:
        rows = st.session_state["roundtrip_results"]
        if st.session_state["roundtrip_settings"] != (check_q_ls, params, check_selected):
            st.warning("입력값이 변경되었습니다. 최신 결과를 보려면 다시 실행하세요.")
        a, b, c = st.columns(3)
        a.metric("검증한 표준관경", f"{len(rows)}개")
        b.metric("최대 유량 상대오차", f"{max(r['유량 상대오차 (%)'] for r in rows):.6g}%")
        c.metric("최대 절대 손실수두 잔차", f"{max(abs(r['손실수두 잔차 (m)']) for r in rows):.3e} m")
        st.latex(r"\mathrm{Relative\ Error}(\%)=\frac{|Q_{\rm forward}-Q_{\rm target}|}{Q_{\rm target}}\times100")
        st.latex(r"r_h=h_{\rm major}+h_{\rm minor}-\Delta z_{\rm inverse}")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        import csv, io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("재검증 결과 CSV 다운로드", buf.getvalue().encode("utf-8-sig"),
                           "team7_roundtrip_validation.csv", "text/csv")


with tab_sweep:
    st.subheader("관경·길이·조도에 따른 자연유량 곡선")
    dzs = st.number_input("곡선 계산에 사용할 고정 Δz (m)", min_value=1e-4, value=3.0, format="%.4f", key="sweep_dz")
    choice = st.radio("변화시킬 변수", ["관경 D", "관 길이 L", "조도 ε"], horizontal=True)
    if st.button("자연유량 곡선 계산", type="primary"):
        try:
            if choice == "관경 D":
                rows = sweep_diameter_natural_flow([0.020 + 0.002*i for i in range(20)], dzs, params)
                x, xlabel, title = "D_mm", "Pipe Inside Diameter D (mm)", "Diameter vs Natural Flow Rate"
            elif choice == "관 길이 L":
                rows = sweep_length_natural_flow([5.0*i for i in range(1, 21)], dzs, params)
                x, xlabel, title = "L", "Pipe Length L (m)", "Length vs Natural Flow Rate"
            else:
                rows = sweep_roughness_natural_flow([1e-6 + i*9.9e-5/19 for i in range(20)], dzs, params)
                x, xlabel, title = "epsilon", "Absolute Roughness ε (m)", "Roughness vs Natural Flow Rate"
            draw_chart(rows, x, "Q_Ls", xlabel, "Natural Flow Rate Q (L/s)", title)
            st.dataframe(rows, use_container_width=True)
        except Exception as e:
            st.error(str(e))


with tab_design:
    st.subheader("표준관경별 목표유량–최소 고도차 설계 비교")
    st.caption("두 대기개방 저장조 · 펌프 없음 · 정상 비압축성 유동 · Darcy–Weisbach + Colebrook")
    st.info("표준관경은 Sch 40의 대표적인 실제 내경(mm)을 사용합니다. 배관 재질·규격·제조사에 따라 내경과 조도가 달라질 수 있습니다.")

    left, right = st.columns(2)
    with left:
        design_q_ls = st.number_input("목표유량 Q (L/s)", min_value=0.001, value=1.30,
                                      step=0.10, format="%.3f", key="standard_design_q")
        max_head = st.number_input("설치 가능한 최대 고도차 Δz (m)", min_value=0.001,
                                   value=3.0, step=0.5, format="%.3f", key="standard_design_head")
    with right:
        max_velocity = st.number_input("설계 허용 최대 유속 (m/s)", min_value=0.01,
                                       value=3.0, step=0.5, format="%.2f", key="standard_design_vmax")
        selected = st.multiselect("비교할 Sch 40 표준관경", options=list(SCH40_INNER_DIAMETERS_MM),
                                  default=list(SCH40_INNER_DIAMETERS_MM), key="standard_pipe_selection")
    st.caption("허용 유속은 사용자가 지정하는 설계 조건이며, 3.0 m/s는 예시값입니다. 관 길이·밀도·점도·조도·ΣK는 왼쪽 입력값을 사용합니다.")

    if st.button("표준관경 설계안 비교 실행", type="primary", key="standard_design_run"):
        if not selected:
            st.warning("비교할 표준관경을 하나 이상 선택하세요.")
        else:
            try:
                results = compare_standard_pipes(design_q_ls / 1000.0, max_head,
                                                 max_velocity, selected, params)
                st.session_state["standard_design_results"] = results
                st.session_state["standard_design_inputs"] = {
                    "q_ls": design_q_ls, "head": max_head, "vmax": max_velocity,
                    "params": params.copy(), "selected": selected[:]
                }
            except Exception as e:
                st.error(f"표준관경 계산 오류: {e}")

    if "standard_design_results" in st.session_state:
        results = st.session_state["standard_design_results"]
        settings = st.session_state["standard_design_inputs"]
        if (settings["q_ls"] != design_q_ls or settings["head"] != max_head or
            settings["vmax"] != max_velocity or settings["params"] != params or
            settings["selected"] != selected):
            st.warning("입력값이 변경되었습니다. 최신 조건을 반영하려면 '표준관경 설계안 비교 실행'을 다시 누르세요.")
        eligible = [r for r in results if r["최종 판정"] == "충족"]
        st.divider()
        a, b, c = st.columns(3)
        a.metric("비교한 표준관경", f"{len(results)}개")
        b.metric("두 설계조건 모두 충족", f"{len(eligible)}개")
        if eligible:
            recommended = min(eligible, key=lambda r: r["내경 D (mm)"])
            c.metric("최소 적합 표준관경", recommended["표준관경"])
            st.success(f"추천(수리학적 조건 기준): {recommended['표준관경']} / "
                       f"내경 {recommended['내경 D (mm)']:.2f} mm / "
                       f"필요 고도차 {recommended['필요 Δz (m)']:.3f} m / "
                       f"유속 {recommended['유속 V (m/s)']:.3f} m/s")
        else:
            c.metric("최소 적합 표준관경", "없음")
            st.warning("선택한 표준관경 중 고도차와 유속 조건을 동시에 만족하는 설계안이 없습니다.")
        st.caption("추천은 '조건을 만족하는 최소 내경' 기준이며, 실제 배관 가격·압력등급·시공비의 경제적 최적화는 포함하지 않습니다.")

        st.markdown("#### ① 표준관경–필요 고도차 비교")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xs = [r["내경 D (mm)"] for r in results]
        ys = [r["필요 Δz (m)"] for r in results]
        ax.plot(xs, ys, "o-", label="Required elevation difference")
        ax.axhline(settings["head"], linestyle="--", color="tab:red", label="Available elevation difference")
        ax.set(xlabel="Actual inside diameter D (mm)", ylabel="Required Δz (m)",
               title="Schedule 40 pipe diameter vs required elevation difference")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### ② 주손실·부차손실 비교")
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        major = [r["주손실 (m)"] for r in results]
        minor = [r["부차손실 (m)"] for r in results]
        ax2.bar(xs, major, width=3.5, label="Major loss")
        ax2.bar(xs, minor, bottom=major, width=3.5, label="Minor loss")
        ax2.set(xlabel="Actual inside diameter D (mm)", ylabel="Head loss (m)",
                title="Major and minor head loss by standard pipe diameter")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("#### ③ 표준관경별 상세 비교표")
        st.dataframe(results, use_container_width=True, hide_index=True,
                     column_config={"Re": st.column_config.NumberColumn(format="%.0f"),
                                    "Darcy f": st.column_config.NumberColumn(format="%.5f"),
                                    "필요 Δz (m)": st.column_config.NumberColumn(format="%.4f"),
                                    "유속 V (m/s)": st.column_config.NumberColumn(format="%.4f"),
                                    "주손실 (m)": st.column_config.NumberColumn(format="%.4f"),
                                    "부차손실 (m)": st.column_config.NumberColumn(format="%.4f")})
        import csv
        import io
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        st.download_button("비교 결과 CSV 다운로드", data=csv_buffer.getvalue().encode("utf-8-sig"),
                           file_name="team7_standard_pipe_design.csv", mime="text/csv")
        if any(r["유동영역"] == "Transition" for r in results):
            st.warning("일부 관경은 천이영역(Re 2100~4000)입니다. 마찰계수 선형보간 결과는 추정값입니다.")
        st.caption("주의: ΣK는 선택한 모든 관경에서 일정하다고 가정했습니다. 실제 관경 변경 시 밸브·피팅의 K와 배관 조도를 재검토해야 합니다.")

with tab_sensitivity:
    st.subheader("설계 민감도 분석: 조건 변화에 따른 최소 적합 표준관경")
    st.write("허용 고도차, 목표 유량, 배관 길이를 각각 변경하면서 추천 관경의 변화를 분석합니다.")
    a, b, c = st.columns(3)
    with a:
        sens_q = st.number_input("기준 목표 유량 (L/s)", min_value=0.001, value=1.30,
                                 step=0.1, format="%.3f", key="sens_q")
        sens_vmax = st.number_input("허용 최대 유속 (m/s)", min_value=0.01, value=3.0,
                                    step=0.5, key="sens_vmax")
    with b:
        sens_head = st.number_input("기준 허용 고도차 (m)", min_value=0.001, value=3.0,
                                    step=0.1, key="sens_head")
        sens_count = st.number_input("변수별 계산 조건 수", min_value=20, max_value=100,
                                     value=20, step=1, key="sens_count")
    with c:
        sens_selected = st.multiselect("분석할 Sch40 표준관경", list(SCH40_INNER_DIAMETERS_MM),
                                       default=list(SCH40_INNER_DIAMETERS_MM), key="sens_pipes")
    st.caption("기준 배관 길이는 사이드바 L을 사용하며, 한 번에 하나의 변수만 바꾸는 OFAT 민감도 분석입니다.")
    st.markdown("#### 변수별 탐색 범위")
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    with h1:
        head_min = st.number_input("고도차 최소 (m)", min_value=0.001, value=1.0, key="sens_head_min")
    with h2:
        head_max = st.number_input("고도차 최대 (m)", min_value=0.001, value=5.0, key="sens_head_max")
    with h3:
        flow_min = st.number_input("유량 최소 (L/s)", min_value=0.001, value=0.5, key="sens_flow_min")
    with h4:
        flow_max = st.number_input("유량 최대 (L/s)", min_value=0.001, value=3.0, key="sens_flow_max")
    with h5:
        length_min = st.number_input("길이 최소 (m)", min_value=0.001, value=5.0, key="sens_length_min")
    with h6:
        length_max = st.number_input("길이 최대 (m)", min_value=0.001, value=100.0, key="sens_length_max")
    if st.button("민감도 분석 실행 (3개 변수)", type="primary", key="sens_run"):
        if not sens_selected:
            st.warning("분석할 표준관경을 선택하세요.")
        elif not (head_min < head_max and flow_min < flow_max and length_min < length_max):
            st.error("모든 변수의 최대값은 최소값보다 커야 합니다.")
        else:
            try:
                ranges = {
                    "head": (head_min, head_max, "Available elevation difference (m)", "Available head vs recommended pipe"),
                    "flow": (flow_min, flow_max, "Target flow (L/s)", "Target flow vs recommended pipe"),
                    "length": (length_min, length_max, "Pipe length (m)", "Pipe length vs recommended pipe"),
                }
                all_rows = {}
                for var, (lo, hi, _, _) in ranges.items():
                    vals = [lo + (hi-lo)*i/(sens_count-1) for i in range(sens_count)]
                    all_rows[var] = sensitivity_sweep(var, vals, sens_q/1000, sens_head,
                                                       sens_vmax, sens_selected, params)
                st.session_state["sensitivity_results"] = all_rows
                st.session_state["sensitivity_settings"] = (sens_q, sens_head, sens_vmax,
                    sens_count, sens_selected[:], params.copy(), head_min, head_max,
                    flow_min, flow_max, length_min, length_max)
            except Exception as e:
                st.error(f"민감도 분석 오류: {e}")
    if "sensitivity_results" in st.session_state:
        current_settings = (sens_q, sens_head, sens_vmax, sens_count, sens_selected, params,
                            head_min, head_max, flow_min, flow_max, length_min, length_max)
        if st.session_state["sensitivity_settings"] != current_settings:
            st.warning("입력값이 변경되었습니다. 최신 조건을 반영하려면 다시 실행하세요.")
        for var, label, xlabel, title in (
            ("head", "① 허용 고도차 변화", "Available elevation difference (m)", "Available head vs recommended pipe"),
            ("flow", "② 목표 유량 변화", "Target flow (L/s)", "Target flow vs recommended pipe"),
            ("length", "③ 배관 길이 변화", "Pipe length (m)", "Pipe length vs recommended pipe"),
        ):
            rows = st.session_state["sensitivity_results"][var]
            st.markdown(f"#### {label}")
            draw_sensitivity(rows, xlabel, title)
            st.dataframe(rows, use_container_width=True, hide_index=True)
        import csv, io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["변화 변수", "변수 값", "추천 관경", "추천 내경 (mm)", "필요 Δz (m)", "유속 (m/s)"])
        writer.writeheader()
        for var, rows in st.session_state["sensitivity_results"].items():
            for row in rows:
                writer.writerow({"변화 변수": var, **row})
        st.download_button("민감도 분석 결과 CSV 다운로드", buf.getvalue().encode("utf-8-sig"),
                           "team7_design_sensitivity.csv", "text/csv")
        st.caption("추천 기준: 허용 고도차와 유속 제한을 모두 만족하는 가장 작은 Sch40 내경. 비용 최적화는 포함하지 않습니다.")


with tab_validation:
    st.subheader("계산 결과 검증")
    st.markdown("#### ① Colebrook ↔ Swamee–Jain 교차검증")
    st.write("동일한 두 저장조 모델에서 마찰계수 계산방법만 변경하여 자연유량과 상대차이를 비교합니다.")
    dzv = st.number_input("검증 고도차 Δz (m)", min_value=1e-4, value=3.0, format="%.4f", key="validation_dz")
    if st.button("교차검증 실행", type="primary"):
        try:
            main, ref, error = validation_crosscheck(dzv, params)
            if main["regime"] != "Turbulent" or ref["regime"] != "Turbulent":
                st.warning("현재 조건은 완전 난류가 아닙니다. Swamee–Jain 검증 해석에 주의하세요.")
            a, b, c = st.columns(3)
            a.metric("Colebrook Q", f"{main['Q']*1000:.6f} L/s")
            b.metric("Swamee–Jain Q", f"{ref['Q']*1000:.6f} L/s")
            c.metric("유량 상대차이", f"{error:.4f}%")
            rows = []
            for name, r in (("Colebrook", main), ("Swamee–Jain", ref)):
                rows.append({"계산법": name, "Q (m³/s)": r["Q"], "V (m/s)": r["V"],
                             "Re": r["Re"], "Darcy f": r["f"], "주손실 (m)": r["h_major"],
                             "부차손실 (m)": r["h_minor"], "총손실 (m)": r["h_total"],
                             "고도차 잔차 (m)": r["residual"]})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.latex(r"\mathrm{Relative\ Difference}(\%)=\frac{|Q_{\rm Colebrook}-Q_{\rm SJ}|}{|Q_{\rm SJ}|}\times100")
            st.caption("두 상관식의 교차검증이며 실험 측정값과의 비교는 아닙니다. Swamee–Jain은 Colebrook의 초기 추정값으로도 사용됩니다.")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.markdown("#### ② Bernoulli 이상유동 한계 검증 (별도 경계조건)")
    st.info("이 검증은 두 저장조와 경계조건이 다른 자유출구 모델입니다. 출구 운동에너지 항 1을 별도로 포함하므로 두 저장조의 출구손실 K=1은 중복 적용하지 않습니다.")
    K_free_outlet = st.number_input("자유출구 모델 손실계수 ΣK (출구손실 제외)", min_value=0.0, value=0.4, step=0.1, format="%.4f", key="free_outlet_k", help="기본값 0.4는 입구손실만 포함합니다. 관 출구의 운동에너지는 식의 1 항에 이미 반영되어 있으므로 기존 두 저장조의 출구손실 K=1은 넣지 마세요.")
    st.caption("기존 두 저장조 계산용 ΣK는 사이드바 값을 그대로 사용합니다. 이 입력은 자유출구 검증에만 적용됩니다.")
    st.latex(r"\Delta z=\left[1+\alpha\left(f_D\frac{L}{D}+\sum K_{\mathrm{free}}\right)\right]\frac{V^2}{2g}")
    st.latex(r"\alpha\to0\quad\Rightarrow\quad Q\to Q_{\rm ideal}=A\sqrt{2g\Delta z}")
    st.write("여기서 α는 **검증용 손실계수 배율**입니다. 1이면 손실을 반영하고 0이면 이상유동입니다. 실제 조도를 0으로 만드는 것과는 다릅니다.")
    dzi = st.number_input("이상유동 검증 고도차 Δz (m)", min_value=1e-4, value=3.0, format="%.4f", key="ideal_dz")
    if st.button("Bernoulli 한계 검증 실행", type="primary"):
        try:
            ideal_q, rows = validation_ideal_outlet(dzi, params, [i/20 for i in range(21)], K_free_outlet=K_free_outlet)
            st.metric("이상 Bernoulli 유량", f"{ideal_q*1000:.6f} L/s")
            draw_chart(rows, "loss_scale", "Q_Ls", "Loss coefficient scale α",
                       "Flow Rate Q (L/s)", "Approach to Ideal Bernoulli Limit", ideal=ideal_q*1000)
            st.dataframe([{"손실 배율 α": r["loss_scale"], "유량 Q (L/s)": r["Q_Ls"],
                           "이상유량 대비 차이 (%)": r["relative_difference_pct"]} for r in rows],
                         use_container_width=True, hide_index=True)
            st.success("α = 0에서 이상 Bernoulli 유량과 일치합니다. α가 감소할수록 이상유량에 접근합니다.")
        except Exception as e:
            st.error(str(e))

st.divider()
st.caption("team 7 | Natural gravity flow between two open reservoirs")
