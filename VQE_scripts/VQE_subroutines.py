from qiskit import *
from qiskit import Aer
from qiskit import IBMQ

from qiskit.aqua                         import QuantumInstance
from qiskit.aqua                         import aqua_globals
from qiskit.providers.aer.noise          import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

from qiskit.circuit             import ParameterVector, QuantumCircuit
from qiskit.circuit.library     import TwoLocal
from qiskit.aqua.operators      import X, Z, I
from qiskit.quantum_info.states import Statevector

import numpy             as np
import matplotlib.pyplot as plt


# defines quantum instance based on the chosen run type
def quantum_instance(run_type):
    seed = aqua_globals.random_seed

    if run_type['backend'] == 'statevector':
        backend = BasicAer.get_backend('statevector_simulator')
        return QuantumInstance(backend=backend, seed_transpiler=seed, seed_simulator=seed)

    if run_type['backend'] == 'qasm':
        backend = BasicAer.get_backend('qasm_simulator')
        return QuantumInstance(backend=backend, seed_transpiler=seed, seed_simulator=seed)

    if run_type['backend'] == 'qasm_noise_model':
        backend = Aer.get_backend('qasm_simulator')
        provider = IBMQ.load_account()
        device = provider.get_backend(run_type['device'])
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates

        if run_type['noise_mitigation'] == False:
            return QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed, coupling_map=coupling_map, basis_gates=basis_gates,
                               noise_model=noise_model, shots=8000)

        if run_type['noise_mitigation'] == True:
            return QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed, coupling_map=coupling_map, basis_gates=basis_gates,
                               noise_model=noise_model, shots=8000, measurement_error_mitigation_cls=CompleteMeasFitter,
                               measurement_error_mitigation_shots=8000)

    if run_type['backend'] == 'hardware':
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = provider.get_backend(run_type['device'])
        return QuantumInstance(backend=backend, shots=8190, measurement_error_mitigation_cls=CompleteMeasFitter,
                               measurement_error_mitigation_shots=8000)

# returns variational form for the VQE.
def ansatz(type, symmetric, n):
    if type == 'unary' and symmetric == False:
        p = ParameterVector('p', n)
        qc = QuantumCircuit(n + 1)

        qc.ry(p[0], 0)
        for k in range(n - 1):
            qc.cry(p[k + 1], k, k + 1)

        for k in reversed(range(n)):
            qc.cx(k, k + 1)

        qc.x(0)
    if type == 'unary' and symmetric == True:
        qc = symmetric_unary_ansatze(n)

    if type == 'gray' and symmetric == False:
        qc = TwoLocal(2, 'ry', 'cx', reps=1)

    if type == 'gray' and symmetric == True:
        p = ParameterVector('p', 1)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.ry(p[0], 0)

    return qc


# symmetric unary ansatze
def symmetric_unary_ansatze(n):
    p = ParameterVector('p', int(np.floor(n / 2)))
    qc = QuantumCircuit(n + 1)
    if n == 2:
        qc.ry(p[0], 1)
        qc.cry(np.pi / 2, 1, 2)
        qc.cx(2, 0)
        qc.cx(1, 2)
        qc.x(1)

    if n == 3:
        qc.ry(np.pi - 2 * np.arctan(np.cos(p[0] / 2)), 0)
        qc.cry(p[0], 0, 3)
        qc.cry(np.pi / 2, 3, 1)
        qc.cx(1, 2)
        qc.cx(3, 1)
        qc.cx(0, 3)
        qc.x(0)

    if n == 4:
        qc.cry(p[0], 0, 4)
        qc.cry(p[1], 4, 2)
        qc.cry(np.pi / 2, 2, 1)
        qc.cx(1, 3)
        qc.cx(2, 1)
        qc.cx(4, 2)
        qc.cx(0, 4)
        qc.x(0)
    return qc


# Hamiltonians for different boson interaction values U, the hopping term J is normalized to 1
def Hamiltonian(type, n, U):
    J = 1

    if n == 2 and type == 'unary':
        return (U / 2) * (I ^ Z ^ I) - (J / 2) * np.sqrt(2) * (
                    (X ^ X ^ I) + ((Z @ X) ^ (X @ Z) ^ I) + (I ^ X ^ X) + (I ^ (Z @ X) ^ (X @ Z)))

    if n == 3 and type == 'unary':
        return (U / 2) * ((2 * I ^ Z ^ I ^ I) + (2 * I ^ I ^ Z ^ I)) - (J / 2) * (
                    2 * ((I ^ X ^ X ^ I) + (I ^ (Z @ X) ^ (X @ Z) ^ I)) + np.sqrt(3) * (
                        (X ^ X ^ I ^ I) + ((Z @ X) ^ (X @ Z) ^ I ^ I) + (I ^ I ^ X ^ X) + (I ^ I ^ (Z @ X) ^ (X @ Z))))
    if n == 4 and type == 'unary':
        return (U / 2) * ((3 * I ^ Z ^ I ^ I ^ I) + (4 * I ^ I ^ Z ^ I ^ I) + (3 * I ^ I ^ I ^ Z ^ I)) - (J / 2) * (
                    2 * ((X ^ X ^ I ^ I ^ I) + ((Z @ X) ^ (X @ Z) ^ I ^ I ^ I)) + 2 * (
                        (I ^ I ^ I ^ X ^ X) + (I ^ I ^ I ^ (Z @ X) ^ (X @ Z))) + np.sqrt(6) * (
                                (I ^ X ^ X ^ I ^ I) + (I ^ (Z @ X) ^ (X @ Z) ^ I ^ I) + (I ^ I ^ X ^ X ^ I) + (
                                    I ^ I ^ (Z @ X) ^ (X @ Z) ^ I)))

    if type == 'gray':
        return -(2 * J) * (np.sqrt(3) * (I ^ X) + (X ^ I) - (X ^ Z)) + (2 * U) * (I ^ Z)


# computes entanglement entropy
def entanglement_entropy(state):
    ent = 0

    for prob in state.probabilities():
        if prob != 0:
            ent = ent - prob * np.log2(prob)
    return ent


# computes visibility
def visibility(type, state, n):
    if n == 2:
        V = np.sqrt(2) * ((X ^ X ^ I) + ((Z @ X) ^ (X @ Z) ^ I) + (I ^ X ^ X) + (I ^ (Z @ X) ^ (X @ Z)))

    if n == 3:
        V = 4 / 3 * ((I ^ X ^ X ^ I) + (I ^ (Z @ X) ^ (X @ Z) ^ I) + (I ^ (Z @ X) ^ X ^ I) + (
                    I ^ X ^ (X @ Z) ^ I)) + np.sqrt(3) * 2 / 3 * (
                        (X ^ X ^ I ^ I) + ((Z @ X) ^ (X @ Z) ^ I ^ I) + (I ^ I ^ X ^ X) + (I ^ I ^ (Z @ X) ^ (X @ Z)))

    if n == 4:
        V = (X ^ X ^ I ^ I ^ I) + ((Z @ X) ^ (X @ Z) ^ I ^ I ^ I) + (I ^ I ^ I ^ X ^ X) + (
                    I ^ I ^ I ^ (Z @ X) ^ (X @ Z)) + np.sqrt(6) / 2 * (
                        (I ^ X ^ X ^ I ^ I) + (I ^ (Z @ X) ^ (X @ Z) ^ I ^ I) + (I ^ I ^ X ^ X ^ I) + (
                            I ^ I ^ (Z @ X) ^ (X @ Z) ^ I))

    if type == 'gray':
        V = 4 / 3 * (np.sqrt(3) * ((I ^ X) + (Z ^ (X @ Z))) + (X ^ I) + ((X @ Z) ^ I) - (X ^ Z) - ((X @ Z) ^ Z))

    return np.real(state.expectation_value(V) / 4)


# direct diagonalization ground states and correlation metrics for different boson interaction values U
def analytic_solution(type, n, U):
    sol = {}

    if n == 4:
        theta = 0
        i = 6 * np.sqrt(3) * np.sqrt(9 * (U ** 6) + 412 * (U ** 4) + 64 * (U ** 2) + 1024)
        j = 288 * U - 35 * (U ** 3)
        k = 13 * (U ** 2) + 48
        US = 12 * np.sqrt(2 / 35)

        if U <= -US or 0 < U and U <= US:
            theta = np.arctan(i / j)

        if U > US or -US < U and U < 0:
            theta = np.arctan(i / j) + np.pi

        E = (11 * U - 2 * np.sqrt(k) * np.cos(theta / 3)) / 3
        if U == 0:
            E = -4

        A = 2 * np.sqrt(6 / (48 + 12 * ((E - 6 * U) ** 2) + (18 * (U ** 2) - 9 * E * U + E ** 2 - 4) ** 2))
        pr0 = A
        pr1 = A * (3 * U - E / 2)
        pr2 = A / (2 * np.sqrt(6)) * (18 * (U ** 2) - 9 * E * U + E ** 2 - 4)
        amplitudes = [0, pr0, pr1, 0, pr2, 0, 0, 0, pr1, 0, 0, 0, 0, 0, 0, 0, pr0]

    if n == 2:
        k = 16 + (U ** 2)
        A = 2 / np.sqrt(k + U * np.sqrt(k))
        pr0 = A
        pr1 = A * (U + np.sqrt(k)) / (2 * np.sqrt(2))
        amplitudes = [0, pr0, pr1, 0, pr0]

    if n == 3 or type == 'gray':
        A = 1 / np.sqrt(2 + 2 / 3 * (1 + U + np.sqrt(4 + U * (2 + U))) ** 2)
        pr0 = A
        pr1 = A * (1 + U + np.sqrt(4 + U * (2 + U))) / np.sqrt(3)
        if type == 'unary':
            amplitudes = [0, pr0, pr1, 0, pr1, 0, 0, 0, pr0]
        elif type == 'gray':
            amplitudes = [pr0, pr1, pr0, pr1]

    if type == 'unary':
        amplitudes.extend(np.zeros((2 ** n) - 1))

    sol['State'] = Statevector(amplitudes)
    sol['Energy'] = np.real(sol['State'].expectation_value(Hamiltonian(type, n, U)))
    sol['Entropy'] = entanglement_entropy(sol['State'])
    sol['Visibility'] = visibility(type, sol['State'], n)
    return sol


# creates a Statevector object from a list of outcomes' counts
def make_state(counts):
    amplitudes = counts.copy()
    amplitudes[:] = [np.sqrt(x / sum(amplitudes)) for x in amplitudes]
    return Statevector(amplitudes)


# creates a Statevector object from VQE results
def state_for_qasm(type, res, n):
    counts = []
    if type == 'gray':
        n = 1
    for k in range(2 ** (n + 1)):
        key = str(format(k, 'b').zfill(n + 1))
        if key not in res:
            counts.append(0)
        else:
            counts.append(res[key])
    return make_state(counts)


# creates a Statevector object from VQE results, discarding counts in non-physical states.
def reduced_state(res, n):
    counts = []
    for k in range(2 ** (n + 1)):
        if k in [1, 2, 4, 8, 16]:
            key = str(format(k, 'b').zfill(n + 1))
            if key not in res:
                counts.append(0)
            else:
                counts.append(res[key])
        else:
            counts.append(0)
    return make_state(counts)


def graph(to_plot, exact, data, U, run_type):
    fig, ax = plt.subplots()
    ax.plot(U, exact[to_plot], 'g', label='Exact values')
    ax.plot(U, data[to_plot], 'r--o', label=run_type['backend'])

    ax.set_xlabel('Boson interaction U')
    ax.set_ylabel(to_plot)
    ax.legend()
    ax.grid(True)
    plt.show()
