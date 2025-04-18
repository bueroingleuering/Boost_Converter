import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.signal import square
from scipy.linalg import solve_discrete_are
"""Parameters"""
f_Switch = 50.0e3  # 50 kHz
Duty = 0.5  # 50 %
U_0 = 100.0  # 100 V
I_max = 10.0  # 10 A
L_0 = 250.0e-6  # 0.25 mH
R_L = 10.0e-3  # 0.01 Ω
R_DS = 20.0e-3  # 0.02 Ω
C_0 = 25.0e-6  # 25 µF
R_Load =  U_0 / ((1 - Duty) * I_max)  # 20 Ω
U_C_Init = U_0
"""Simulation Parameters"""
T_Switch = 1 / f_Switch  # 0.02 ms
T_Sim = 500 * T_Switch  # 10 ms
T_Step = T_Switch / 100.0  # 0.0002 ms
"""switch mode"""
t = np.arange(start=0, stop=T_Sim, step=T_Step)
gate_pulse = signal = square(2 * np.pi * f_Switch * t, duty=Duty)
i = U_C_Init / R_Load * np.ones_like(t)
U_C = U_C_Init * np.ones_like(t)
"""state space"""
A_ss = np.array([[-(R_L + Duty * R_DS) / L_0, -(1 - Duty) / L_0], [(1 - Duty) / C_0, -1 / (R_Load * C_0)]])
B_ss = np.array([[1 / L_0], [0.0]])
C_ss = np.array([[0.0, 1.0]])
D_ss = np.array([0.0])
Qo = ct.obsv(A_ss, C_ss)  # Observability matrix

if np.linalg.matrix_rank(Qo) == A_ss.shape[0]:
    h = 20000 * np.array([0.1, 1.0])
    x_o = np.array([i[0], U_C[0]])
    X_o = np.zeros((2, len(t)))
    u = U_0 * np.ones(len(t))
    X_o[:, 0] = x_o
    controller_counter = 0
    """Simulation"""
    for index in range(1, len(t)):
        controller_counter += 1
        """switch mode"""
        if gate_pulse[index] > 0:
            didt = (U_0 - (R_DS + R_L) * i[index - 1]) / L_0
            dU_Cdt = (-U_C[index - 1] / R_Load) / C_0
        else:
            didt = (U_0 - R_L * i[index - 1] - U_C[index - 1]) / L_0
            dU_Cdt = (i[index - 1] - U_C[index - 1] / R_Load) / C_0
        i[index] = i[index - 1] + didt * T_Step
        U_C[index] = U_C[index - 1] + dU_Cdt * T_Step
        """state space"""
        if controller_counter >= 400:
            x_dot_delta = h * (C_ss @ x_o - U_C[index])
            x_dot = A_ss @ x_o + B_ss.flatten() * u[index - 1] - x_dot_delta
            x_o = x_o + x_dot * T_Step * controller_counter
            controller_counter = 0
        X_o[:, index] = x_o
i_o = X_o[0, :]
U_Co = X_o[1, :]
"""Plots"""
fig_1 = plt.figure(num=1, figsize=(10, 6))
plt.plot(t, i, label="current switched", color='#284b64', linewidth=3)
plt.plot(t, i_o, label="current observer", color='#893636', linewidth=3)
plt.xlabel('Time [s]', fontsize=16), plt.ylabel('Current [A]', fontsize=16), plt.legend(fontsize=16), plt.grid()
plt.savefig('current observer.png', format="png", dpi=300)

fig_2 = plt.figure(num=2, figsize=(10, 6))
plt.plot(t, U_C, c='#284b64', linewidth=4, label='voltage switched')
plt.plot(t, U_Co, c='#893636', linewidth=4, label='voltage observer')
plt.xlabel('Time [s]', fontsize=16), plt.ylabel('Voltage [V]', fontsize=16), plt.legend(fontsize=16), plt.grid()
plt.savefig('voltage observer.png', format="png", dpi=300)