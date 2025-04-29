import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.signal import square
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
"""Pole Placement"""
zeta = 1 / np.sqrt(2)
T_s = 0.1 * T_Sim
omega_n = 4 / (zeta * T_s)
poles_real_part = -zeta * omega_n
poles_imag_part = omega_n * np.sqrt(1 - zeta**2)
desired_poles = np.array([poles_real_part + 1j * poles_imag_part,
                          poles_real_part - 1j * poles_imag_part])
"""Switch Mode"""
t = np.arange(start=0, stop=T_Sim, step=T_Step)
gate_pulse = signal = square(2 * np.pi * f_Switch * t, duty=Duty)
i = U_C_Init / R_Load * np.ones_like(t)
U_C = U_C_Init * np.ones_like(t)
"""State Space"""
A_ss = np.array([[-(R_L + Duty * R_DS) / L_0, -(1 - Duty) / L_0], [(1 - Duty) / C_0, -1 / (R_Load * C_0)]])
B_ss = np.array([[1 / L_0], [0.0]])
C_ss = np.array([[0.0, 1.0]])
D_ss = np.array([0.0])
"""Controller"""
R_cl = np.array([[0, 0]])
h = 10000 * np.array([0.1, 1.0])
v = 0
eta = 0.5
controller_counter = 0
w = U_0 / (1 - Duty) * np.ones_like(t)
D = Duty * np.ones_like(t)
"""Simulation"""
x_o = np.array([i[0], U_C[0]])
X_o = np.zeros((2, len(t)))
u = np.zeros(len(t))
X_o[:, 0] = x_o
for index in range(1, len(t)):
    controller_counter += 1
    """Switch Mode"""
    if gate_pulse[index] > 0:
        didt = (U_0 - (R_DS + R_L) * i[index - 1]) / L_0
        dU_Cdt = (-U_C[index - 1] / R_Load) / C_0
    else:
        didt = (U_0 - R_L * i[index - 1] - U_C[index - 1]) / L_0
        dU_Cdt = (i[index - 1] - U_C[index - 1] / R_Load) / C_0
    i[index] = i[index - 1] + didt * T_Step
    U_C[index] = U_C[index - 1] + dU_Cdt * T_Step
    """State Space"""
    if controller_counter >= 100:
        """Optimization"""
        while np.abs(u[index] - U_0) > eta:
            u[index] = v * w[index] - R_cl[0] @ x_o
            if u[index] < U_0 - eta:
                Duty -= 0.0025
            elif u[index] > U_0 + eta:
                Duty += 0.0025
            if Duty <= 0 or Duty >= 0.9:
                break
            A_ss = np.array([[-(R_L + Duty * R_DS) / L_0, -(1 - Duty) / L_0], [(1 - Duty) / C_0, -1 / (R_Load * C_0)]])
            G_ss = ct.ss(A_ss, B_ss, C_ss, D_ss)
            R_cl = ct.acker(A_ss, B_ss, desired_poles)
            G_cl = ct.ss(A_ss - B_ss * R_cl, B_ss, C_ss, D_ss) / (1 - Duty)
            v = G_ss.dcgain() / G_cl.dcgain()
        x_dot_delta = h * (C_ss @ x_o - U_C[index])
        x_dot = A_ss @ x_o + B_ss.flatten() * u[index] - x_dot_delta
        x_o = x_o + x_dot * T_Step * controller_counter
        controller_counter = 0
        gate_pulse = signal = square(2 * np.pi * f_Switch * t, duty=Duty)
    X_o[:, index] = x_o
    D[index] = Duty
i_o = X_o[0, :]
U_Co = X_o[1, :]
"""Plots"""
fig_1 = plt.figure(num=1, figsize=(10, 6))
plt.plot(t, i, label="current switched", color='#284b64', linewidth=3)
plt.plot(t, i_o, label="current observer", color='#893636', linewidth=3)
plt.ylim([0.0, 50.0])
plt.xlabel('Time [s]', fontsize=16), plt.ylabel('Current [A]', fontsize=16), plt.legend(fontsize=16), plt.grid()
plt.savefig('current mpc.png', format="png", dpi=300)

fig_2 = plt.figure(num=2, figsize=(10, 6))
plt.plot(t, U_C, c='#284b64', linewidth=3, label='voltage switched')
plt.plot(t, U_Co, c='#893636', linewidth=3, label='voltage observer')
plt.ylim([100, 275.0])
plt.xlabel('Time [s]', fontsize=16), plt.ylabel('Voltage [V]', fontsize=16), plt.legend(fontsize=16), plt.grid()
plt.savefig('voltage mpc.png', format="png", dpi=300)

fig_3 = plt.figure(num=3, figsize=(10, 6))
plt.plot(t, 100 * D[0] * np.ones_like(t), c='#284b64', linewidth=3, label='duty cycle constant')
plt.plot(t,  np.floor(100 * D), c='#893636', linewidth=3, label='duty cycle mpc')
plt.xlabel('Time [s]', fontsize=16), plt.ylabel('Duty Cycle [%]', fontsize=16), plt.legend(fontsize=16), plt.grid()
plt.savefig('duty cycle mpc.png', format="png", dpi=300)