import qiskit
import scipy
import os
import numpy as np
from pyqsp.completion import CompletionError
from pyqsp.angle_sequence import AngleFindingError
from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.providers.aer.noise import NoiseModel, errors, coherent_unitary_error
import qiskit.quantum_info as qi
from qiskit.converters import circuit_to_dag
import scipy.sparse as sp

from utils_gsp import construct_ising_local_term, approx_polynomial, get_phis, qc_U, qc_U_Strang, qc_QETU_cf_R, qc_QETU_R, get_phis




def qetu_rqc_oneLayer(L, J, g, t, mu, c2=0, d=30, c=0.95,
    steep=0.01, max_iter_for_phis=10, RQC_layers=5, 
    init_state=None, split_U=1, reuse_RQC=0, qc_U_custom=None,
    custom_qc_QETU_cf_R=None, qc_cU_custom=None, hamil=None,
    H1=None, H2=None, heis_c2=0, eps=0, lambda_est=None, reverse=False,
    multi_ancilla=1, a_assess=None
    ):
    # Implements one QETU Layer.

    t = t/split_U
    V_list = []
    dirname = os.path.dirname(os.path.realpath(__file__))
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(V_list))]

    print("Constructing QETU sequence")
    qcs_rqc = []
    for V in V_list:
        qc_rqc = qiskit.QuantumCircuit(2)
        qc_rqc.unitary(V, [0, 1])
        qcs_rqc.append(qc_rqc)

    backend = Aer.get_backend("statevector_simulator")
    phis = []
    i = 0
    while True:
        try:
            phis = get_phis(mu, d, steep, c=c, reverse=reverse)
            break
        except CompletionError:
            print("Completion Error encountered!")
            if i>max_iter_for_phis:
                raise Exception("Max Iteration for estimating the phis breached!")
            i = i + 1
            c = c - 0.01
            print(f"c updated to {c}!")
            if i > max_iter_for_phis / 2:
                print(f"QSP did not work for d = {d}, updating d to {d-4}")
                d = d - 4
        except AngleFindingError:
            print("AngleFindingError encountered!")
            if i>max_iter_for_phis:
                raise Exception("Max Iteration for estimating the phis breached!")
            i = i + 1
            
            if i > max_iter_for_phis / 2:
                print(f"QSP did not work for d = {d}, updating d to {d-4}")
                d = d - 4
    phis_list = [phi+eps for phi in phis[0]]


    qc_U_ins = qiskit.QuantumCircuit(L)
    if qc_U_custom is None and lambda_est is None:
        for i in range(split_U):
            qc_U_ins.append(qc_U(qcs_rqc, L, perms).to_gate(), [i for i in range(L)])
    elif lambda_est is not None:
        qc_Phase = qiskit.QuantumCircuit(L)
        for i in range(L):
            qc_Phase.p(-t * split_U * lambda_est, i)
            qc_Phase.x(i)
            qc_Phase.p(-t * split_U * lambda_est, i)
            qc_Phase.x(i)
        qc_U_ins.append(qc_Phase.to_gate(), [j for j in range(L)])
    else:
        qc_U_ins = qc_U_custom
    
    if custom_qc_QETU_cf_R is not None:
        qc_QETU = custom_qc_QETU_cf_R(qc_U_ins, phis_list, c2, split_U=split_U)
    elif qc_cU_custom is not None:
        qc_QETU = qc_QETU_R(qc_cU_custom, phis_list, c2, multi_ancilla=multi_ancilla)
    else:
        qc_QETU = qc_QETU_cf_R(qc_U_ins, phis_list, c2)

    qc_ins = qiskit.QuantumCircuit(L+multi_ancilla)
    qc_ins.append(qc_QETU.to_gate(), [i for i in range(L+multi_ancilla)])


    return qc_ins, (phis_list, phis[1], phis[2])

