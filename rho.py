import numpy as np
from scipy import linalg

def generate_density_matrix(n):
    """
    生成一个n维的随机密度矩阵
    
    参数:
    n (int): 矩阵维数
    
    返回:
    numpy.ndarray: 一个满足密度矩阵所有条件的n维矩阵
    """
    # 生成随机复数矩阵, 元素服从标准正态分布
    A = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))
    matx = np.dot(A, A.conj().T)
    
    # 精确归一化
    tr = np.trace(matx).real
    if abs(tr) < 1e-10:  # 避免除以接近零的数
        # 如果迹接近零，重新生成一次
        return generate_density_matrix(n)
    
    matx = matx / tr
    
    # 确保厄米性（消除可能的数值误差）
    matx = 0.5 * (matx + matx.conj().T)

    return matx


def verify_density_matrix(matx, tol=1e-10):
    """
    验证生成的矩阵是否满足密度矩阵的所有条件
    
    参数:
    matx (numpy.ndarray): 待验证的矩阵
    tol (float): 数值误差容忍度
    
    返回:
    numpy.ndarray 或 dict: 若是有效密度矩阵则打印并返回该矩阵, 否则返回包含验证结果的字典
    """
    # 检查是否为厄米矩阵
    hermitian_diff = np.max(np.abs(matx - matx.conj().T))
    is_hermitian = hermitian_diff < tol
    
    # 检查迹是否为1
    trace = np.trace(matx).real
    trace_diff = abs(trace - 1.0)
    has_unit_trace = trace_diff < tol
    
    # 检查是否半正定(所有特征值非负)
    eigenvalues = np.linalg.eigvalsh(matx)
    min_eigenvalue = np.min(eigenvalues)
    is_positive_semidefinite = min_eigenvalue > -tol
    
    # 检查密度矩阵是否满足所有条件
    is_valid = is_hermitian and has_unit_trace and is_positive_semidefinite
    
    if is_valid:
        print(matx)
        return matx
    else:
        return {
            "is_valid": is_valid,
            "is_hermitian": is_hermitian,
            "hermitian_diff": hermitian_diff,
            "has_unit_trace": has_unit_trace,
            "trace": trace,
            "trace_diff": trace_diff,
            "is_positive_semidefinite": is_positive_semidefinite,
            "min_eigenvalue": min_eigenvalue,
            "eigenvalues": eigenvalues
        }
    

def Bures_distance(rho, sigma):
    """
    计算两个密度矩阵之间的Bures距离
    B(ρ, σ) = arccos(sqrt(rho^(1/2)·σ·ρ^(1/2)))
    
    参数:
    rho, sigma (numpy.ndarray): 两个密度矩阵
    
    返回: 
    float: Bures距离
    """
    # 对ρ特征值分解, 生成特征值和特征向量数组
    rho_eigenvalues, rho_eigenvectors = np.linalg.eigh(rho)
    # 确保特征值非负
    rho_eigenvalues = np.maximum(rho_eigenvalues, 0)
    
    # 计算ρ^(1/2)
    rho_sqrt = np.dot(rho_eigenvectors * np.sqrt(rho_eigenvalues), rho_eigenvectors.conj().T)
    
    # 计算ρ^(1/2)·σ·ρ^(1/2)
    current_matrix = rho_sqrt @ sigma @ rho_sqrt
    
    # 确保特征值非负
    current_matrix_eigenvalues = np.linalg.eigvalsh(current_matrix)
    current_matrix_eigenvalues = np.maximum(current_matrix_eigenvalues, 0)
    
    # 计算tr(ρ^(1/2)·σ·ρ^(1/2))
    fidelity = np.sum(np.sqrt(current_matrix_eigenvalues))
    
    # 确保保真度数值有效
    fidelity = min(fidelity, 1.0)
    
    # 计算Bures距离
    Bures_dist = np.arccos(fidelity)
    
    return Bures_dist


# 随机生成两个密度矩阵
denmatx_1 = generate_density_matrix(2) 
denmatx_2 = generate_density_matrix(2)

# 验证密度矩阵有效性
verify_density_matrix(denmatx_1)
verify_density_matrix(denmatx_2)

#计算Bures距离并打印
distB = Bures_distance(denmatx_1, denmatx_2)
print(distB)