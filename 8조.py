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
    return "Laminar" if Re < 2300 else "Transition" if Re <= 4000 else "Turbulent"


def swamee_jain(Re, epsilon, D):
    x = epsilon / (3.7 * D) + 5.74 / Re ** 0.9
    return 0.25 / math.log10(x) ** 2


def friction_factor(Re, epsilon, D, method="colebrook"):
    if method not in ("colebrook", "swamee_jain"):
        raise ValueError("알 수 없는 마찰계수 계산방법입니다.")
    if Re <= 0 or D <= 0 or epsilon < 0:
        raise ValueError("마찰계수 입력값이 잘못되었습니다.")
    if Re < 2300:
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

    f_low = 64 / 2300
    if method == "swamee_jain":
        f_high, n, ok = swamee_jain(4000, epsilon, D), 0, True
    else:
        f_high, n, ok = colebrook(4000)
    f = f_low + (Re - 2300) / 1700 * (f_high - f_low)
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

def validation_ideal_outlet(delta_z, params, scales):
    """별도 모델: 큰 저장조 자유표면 -> 대기 중 자유출구.
    Δz = [1 + alpha*(f L/D + ΣK)] V²/(2g).
    """
    validate_params(params)
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
                head = (1 + alpha * (r["f"] * params["L"] / params["D"] + params["K_total"])) * r["velocity_head"]
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
    # ============================================================
# Streamlit 웹 화면: 계산 함수 정의가 모두 끝난 다음에 배치
# ============================================================
st.set_page_config(page_title="Team 8 Natural Flow Design", page_icon="💧", layout="wide")
st.title("두 저장조 사이 자연유동 및 최소 고도차 설계")
st.caption("Team 8 | Mechanical Energy Balance · Colebrook · Inverse Design · Validation")

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

tab_forward, tab_inverse, tab_sweep, tab_design, tab_validation = st.tabs(
    ["정방향 계산", "역설계", "자연유량 곡선", "D-Δz 설계 비교", "계산 결과 검증"]
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


with tab_sweep:
    st.subheader("관경·길이·조도에 따른 자연유량 곡선")

    dzs = st.number_input(
        "곡선 계산에 사용할 고정 Δz (m)",
        min_value=1e-4,
        value=3.0,
        format="%.4f",
        key="sweep_dz"
    )

    choice = st.radio(
        "변화시킬 변수",
        ["관경 D", "관 길이 L", "조도 ε"],
        horizontal=True
    )

    if st.button("자연유량 곡선 계산", type="primary"):
        try:
            if choice == "관경 D":
                rows = sweep_diameter_natural_flow(
                    [0.020 + 0.002*i for i in range(20)], dzs, params
                )
                x, xlabel, title = "D_mm", "Pipe Inside Diameter D (mm)", "Diameter vs Natural Flow Rate"

            elif choice == "관 길이 L":
                rows = sweep_length_natural_flow(
                    [5.0*i for i in range(1, 21)], dzs, params
                )
                x, xlabel, title = "L", "Pipe Length L (m)", "Length vs Natural Flow Rate"

            else:
                rows = sweep_roughness_natural_flow(
                    [1e-6 + i*9.9e-5/19 for i in range(20)], dzs, params
                )
                x, xlabel, title = "epsilon", "Absolute Roughness ε (m)", "Roughness vs Natural Flow Rate"

            draw_chart(rows, x, "Q_Ls", xlabel, "Natural Flow Rate Q (L/s)", title)
            st.dataframe(rows, use_container_width=True)

        except Exception as e:
            st.error(str(e))

with tab_design:
    st.subheader("목표유량을 위한 D-Δz 설계 비교")
    st.info("현재 관경 값은 민감도 분석용입니다. 최종 설계에서는 실제 표준관의 내경으로 교체하세요.")
    qd = st.number_input("목표유량 Q target (m³/s)", min_value=1e-7, value=0.0013, format="%.8f", key="design_q")
    if st.button("D-Δz 조합 계산", type="primary"):
        try:
            rows = sweep_diameter_inverse([0.020 + 0.002*i for i in range(20)], qd, params)
            draw_chart(rows, "D_mm", "delta_z", "Pipe Inside Diameter D (mm)",
                       "Required Elevation Difference Δz (m)", "Required Δz vs Pipe Diameter")
            st.dataframe(rows, use_container_width=True)
        except Exception as e:
            st.error(str(e))

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
    st.warning("이 검증은 '큰 저장조 자유표면 → 대기 중 자유출구' 모델입니다. 기존 '큰 저장조 → 큰 저장조' 주 계산식과 경계조건이 다릅니다.")
    st.latex(r"\Delta z=\left[1+\alpha\left(f_D\frac{L}{D}+\sum K\right)\right]\frac{V^2}{2g}")
    st.latex(r"\alpha\to0\quad\Rightarrow\quad Q\to Q_{\rm ideal}=A\sqrt{2g\Delta z}")
    st.write("여기서 α는 **검증용 손실계수 배율**입니다. 1이면 손실을 반영하고 0이면 이상유동입니다. 실제 조도를 0으로 만드는 것과는 다릅니다.")
    dzi = st.number_input("이상유동 검증 고도차 Δz (m)", min_value=1e-4, value=3.0, format="%.4f", key="ideal_dz")
    if st.button("Bernoulli 한계 검증 실행", type="primary"):
        try:
            ideal_q, rows = validation_ideal_outlet(dzi, params, [i/20 for i in range(21)])
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
st.caption("Team 8 | Natural gravity flow between two open reservoirs")
