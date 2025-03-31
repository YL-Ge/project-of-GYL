import numpy as np
from scipy import linalg

def generate_random_density_matrix(n, method='bures'):
    """
    生成一个n×n的随机密度矩阵
    
    参数:
    n (int): 矩阵维度
    method (str): 生成方法，'ginibre'或'bures'
    
    返回:
    numpy.ndarray: 一个满足密度矩阵所有条件的n×n矩阵
    """
    if method == 'ginibre':
        # Ginibre ensemble方法
        A = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))
        rho = np.dot(A, A.conj().T)
        trace = np.trace(rho).real
        if abs(trace) < 1e-10:
            return generate_random_density_matrix(n, method)
        rho = rho / trace
        rho = 0.5 * (rho + rho.conj().T)
        
    elif method == 'bures':
        # Bures measure方法
        A = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))
        B = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))
        
        AA = np.dot(A, A.conj().T)
        
        eval, evec = np.linalg.eigh(AA)
        eval = np.maximum(eval, 0)
        sqrt_eval = np.sqrt(eval)
        sqrt_AA = np.dot(evec * sqrt_eval, evec.conj().T)
        
        temp = np.dot(sqrt_AA, B)
        rho = np.dot(temp, temp.conj().T)
        
        trace = np.trace(rho).real
        if abs(trace) < 1e-10:
            return generate_random_density_matrix(n, method)
        
        rho = rho / trace
        rho = 0.5 * (rho + rho.conj().T)
    
    else:
        raise ValueError("方法必须是'ginibre'或'bures'")
    
    return rho

def bures_angle_distance(rho, sigma):
    """
    计算两个密度矩阵之间的Bures角距离
    B(ρ,σ) = arccos(tr(sqrt(ρ^(1/2)·σ·ρ^(1/2))))
    
    参数:
    rho, sigma (numpy.ndarray): 两个密度矩阵
    
    返回:
    float: Bures角距离
    """
    # 计算ρ的平方根
    rho_eigenvalues, rho_eigenvectors = np.linalg.eigh(rho)
    # 确保特征值为非负
    rho_eigenvalues = np.maximum(rho_eigenvalues, 0)
    rho_sqrt = np.dot(rho_eigenvectors * np.sqrt(rho_eigenvalues), rho_eigenvectors.conj().T)
    
    # 计算ρ^(1/2)·σ·ρ^(1/2)
    temp = np.dot(rho_sqrt, sigma)
    product = np.dot(temp, rho_sqrt)
    
    # 计算tr(sqrt(ρ^(1/2)·σ·ρ^(1/2)))
    product_eigenvalues = np.linalg.eigvalsh(product)
    product_eigenvalues = np.maximum(product_eigenvalues, 0)  # 确保特征值为非负
    fidelity = np.sum(np.sqrt(product_eigenvalues))
    
    # 确保fidelity在有效范围内（数值误差可能导致fidelity略大于1）
    fidelity = min(fidelity, 1.0)
    
    # 计算Bures角距离
    bures_angle = np.arccos(fidelity)
    
    return bures_angle

# 设置随机密度矩阵的维度
n = 4

# 使用Bures方法生成两个随机密度矩阵
rho = generate_random_density_matrix(n, method='bures')
sigma = generate_random_density_matrix(n, method='bures')

# 计算Bures角距离
distance = bures_angle_distance(rho, sigma)

# 输出结果
print("随机生成的密度矩阵 ρ:")
print(rho)
print("\n随机生成的密度矩阵 σ:")
print(sigma)
print(f"\nBures角距离 B(ρ,σ) = {distance:.6f}")