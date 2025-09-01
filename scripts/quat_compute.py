import numpy as np

def normalize_quaternion(q):
    """归一化四元数"""
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("零长度四元数无法归一化")
    return q / norm

def inverse_quaternion(q):
    """四元数求逆（假设输入已归一化）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def multiply_quaternion(q1, q2):
    """四元数相乘 (q1 ⊗ q2)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def input_quaternion(prompt):
    """输入四元数，格式 w x y z"""
    while True:
        s = input(prompt).strip()
        if s.lower() in ['exit', 'quit']:
            return None
        parts = s.split()
        if len(parts) != 4:
            print("请输入四个数字，以空格分开，例如: 0.972 0 0.233 0")
            continue
        try:
            q = np.array(list(map(float, parts)))
            return normalize_quaternion(q)
        except ValueError:
            print("输入无效，请输入数字。")

if __name__ == "__main__":
    print("循环计算 q1_inv ⊗ q2 并归一化，输入 'exit' 或 'quit' 退出")
    while True:
        q1 = input_quaternion("请输入 q1 (w x y z): ")
        if q1 is None:
            break
        q2 = input_quaternion("请输入 q2 (w x y z): ")
        if q2 is None:
            break

        q1_inv = inverse_quaternion(q1)
        result = multiply_quaternion(q1, q2) 
        result_normalized = normalize_quaternion(result)

        print("q1_inv =", q1_inv)
        print("乘积结果（归一化） =", result_normalized)
        print("-" * 40)
