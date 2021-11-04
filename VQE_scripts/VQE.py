from qiskit.aqua.components.optimizers import SLSQP, SPSA, COBYLA
from qiskit.aqua.algorithms            import VQE
from qiskit.quantum_info               import state_fidelity

from   VQE_subroutines import *


VQE_data = {'optimal_points': [], 'entropies': [], 'visibilities': [], 'fidelities': [], 'energy_residuals': []}
Exact_values = {'entropies': [], 'visibilities': []}

slsqp  = SLSQP(maxiter=1000)
spsa   = SPSA(maxiter=300)
cobyla = COBYLA(maxiter=1000)


run_type={'backend': 'qasm_noise_model', 'device': 'ibmq_manila', 'noise_mitigation': True}
# possible types of run: choose between different backends and devices
# backend: 'statevector', 'quasm', 'quasm_noise_model', 'hardware'
# device: insert the name of one of IBMQ quantum computers
# noise_mitigation: implements CompleteMeasFitter if True

encoding  = 'unary'  # choose between 'unary' or 'gray' encoding. Gray only works for 3 bosons.
Optimizer = cobyla   # choose the classical optimizer used by the VQE.

symmetric_ansatz = False  # use symmetric ansatze if True
reduce_states    = True   # discards counts in qubit states without physical meaning if True, useful for unary encoding.

n_bosons = 3   # total number of bosons, choose from 2, 3 or 4 for unary encoding.
min_u    = 0   # minimum boson interaction value
max_u    = 10  # maximum boson interaction value
n_points = 20  # number of points computed

Qi = quantum_instance(run_type)
Ansatz = ansatz(encoding, symmetric_ansatz, n_bosons)

u = []
for i in range(n_points):

    u.append(i * (max_u - min_u) / n_points + min_u)

    vqe = VQE(operator=Hamiltonian(encoding, n_bosons, u[i]), var_form=Ansatz, optimizer=Optimizer, quantum_instance=Qi)

    result = vqe.run()

    if run_type['backend'] == 'statevector':
        State = Statevector(result.eigenstate)
    elif reduce_states == False:
        State = state_for_qasm(encoding, result.eigenstate, n_bosons)
    elif reduce_states == True and encoding == 'unary':
        State = reduced_state(result.eigenstate, n_bosons)

    print('Optimizer time ', result.optimizer_time)
    print('Optimizer evals ', result.optimizer_evals)
    initial_pt=result.optimal_point

    Sol = analytic_solution(encoding, n_bosons, u[i])
    Exact_values['entropies'].append(Sol['Entropy'])
    Exact_values['visibilities'].append(Sol['Visibility'])

    VQE_data['optimal_points'].append(list(result.optimal_point))
    VQE_data['entropies'].append(entanglement_entropy(State))
    VQE_data['visibilities'].append(visibility(encoding, State, n_bosons))
    VQE_data['fidelities'].append(state_fidelity(State, Sol['State']))
    VQE_data['energy_residuals'].append(np.real(result.optimal_value) - Sol['Energy'])



print("VQE_data=", VQE_data)
print("Exact_values=", Exact_values)

graph('entropies', Exact_values, VQE_data, u, run_type)
graph('visibilities', Exact_values, VQE_data, u, run_type)

