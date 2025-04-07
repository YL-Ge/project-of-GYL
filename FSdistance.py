import numpy as np
from numpy import linalg

def generate_a_ket(n):
    """
    随机生成一个包含 n 个元素的归一化复数列向量

    参数:
    n(int): 列向量维数, 必须是正整数.
    
    返回:
    np.ndarray: 形状为(n, 1)的复数列向量
    """
    # 生成随机复数列向量, 元素服从标准正态分布
    ket = np.random.normal(size=(n, 1)) + 1j * np.random.normal(size=(n, 1))
    
    # 归一化
    norm_ket = np.linalg.norm(ket)
    normalized_ket = ket / norm_ket
    
    # 确保归一化后向量范数为1, 并返回归一化列向量
    if abs(np.linalg.norm(normalized_ket) - 1.0) < 1e-10: 
        return normalized_ket


def pure_density_matrix(normalized_ket):
    """
    将归一化列向量张成纯态密度矩阵

    参数:
    normalized_ket (numpy.ndarray): 归一化的列向量
    
    返回:
    pure_denmatx (np.ndarray): 一个纯态密度矩阵
    """
    # 张成纯态密度矩阵
    normalized_bra = normalized_ket.conj().T
    pure_denmatx = np.outer(normalized_ket, normalized_bra)
    
    return pure_denmatx
    
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
        

def Fubini_Study_distance(rho, sigma):
    """
    计算两个纯态的Fubini-Study距离
    
    参数:
    rho, sigma (numpy.ndarray): 两个纯态密度矩阵
    
    返回: 
    float: Fubini-Study距离
    """
    # 计算希尔伯特-施密特内积
    # 对厄米矩阵, <A,B>=tr(AB)
    inner_product = np.trace(rho @ sigma)
    
    # 计算Fubini-Study距离
    distFS = np.arccos(abs(inner_product))
    
    return distFS
    
        
# 随机生成两个归一化列向量
ket_1 = generate_a_ket(4)
ket_2 = generate_a_ket(4)

# 张成两个纯态密度矩阵
pure_denmatx_1 = pure_density_matrix(ket_1)
pure_denmatx_2 = pure_density_matrix(ket_2)

# 验证密度矩阵有效性
verify_density_matrix(pure_denmatx_1)
verify_density_matrix(pure_denmatx_2)

# 计算Fubini-Study距离并打印
distFS = Fubini_Study_distance(pure_denmatx_1, pure_denmatx_2)
print(distFS)