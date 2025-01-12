import numpy as np
import math
import matplotlib.pyplot as plt

# Определяем функцию для системы уравнений
# p0 - p_value, m0 - m_value
def system(t: float, Y: np.ndarray, m_value: float, p_value: float) -> list:
    # Сохраняем повторяющиеся вычисления
    pressure_term = p_value - 8.3938 * pl(Y[1])
    mass_term = m_value - 1377 * t
    angle_value = angle(t)

    # Основное уравнение
    dz_dt = (pressure_term - F(Y[1], Y[0])) * angle_value / mass_term - g(Y[1])  # производная z
    dy_dt = Y[0]  # производная y

    # Изохронная производная по m0
    dz_dm = - (pressure_term - F(Y[1], Y[0])) * angle_value / (mass_term ** 2)  # производная zm
    dy_dm = Y[2]  # производная ym

    # Изохронная производная по p0
    dz_dp = 1 / mass_term  # производная zp
    dy_dp = Y[4]  # производная yp

    return [dz_dt, dy_dt, dz_dm, dy_dm, dz_dp, dy_dp]

def angle(t):
    # Переводим градусы в радианы
    degrees = 0.516949 * t
    radians = math.radians(degrees)

    # Вычисляем косинус
    return math.cos(radians)

def F(h: float, v: float) -> float:
    """
    Вычисляет силу сопротивления воздуха.

    Параметры:
    h (float): Высота.
    v (float): Скорость.

    Возвращает:
    float: Сила сопротивления.
    """
    return 0.8 * 0.3 * 7.9 * d(h) * v ** 2

def d(h: float) -> float:
    """
    Вычисляет плотность воздуха на высоте.

    Параметры:
    h (float): Высота.

    Возвращает:
    float: Плотность воздуха.
    """
    return pl(h) * 0.029 / T(h) / 8.314

def T(h: float) -> float:
    """
    Вычисляет температуру на высоте.

    Параметры:
    h (float): Высота.

    Возвращает:
    float: Температура (К).
    """
    h += 100

    if h <= 10000:
        return 288.15 - 0.0065 * h
    return 223

def g(h: float) -> float:
    """
    Вычисляет ускорение свободного падения на высоте.

    Параметры:
    h (float): Высота.

    Возвращает:
    float: Ускорение свободного падения (м/с²).
    """
    h += 100

    return 9.8 * 6_371_000 ** 2 / (6_371_000 + h) ** 2

def pl(h: float) -> float:
    """
    Вычисляет давление воздуха на высоте.

    Параметры:
    h (float): Высота.

    Возвращает:
    float: Давление воздуха.
    """
    h += 100

    return 101325 * math.exp(-h / 8400)


# Решение системы уравнений
def solve_model(m_value: float, p_value: float, t_span: tuple = (0, 118), step: float = 0.1) -> tuple:
    """
    Решает модель с использованием метода Рунге-Кутты 4-го порядка.

    Параметры:
    m_value (float): Значение массы.
    p_value (float): Значение давления.
    t_span (tuple): Временной интервал (начало, конец).
    step (float): Шаг интегрирования.

    Возвращает:
    tuple: Массив временных значений и массив решений.
    """
    t0, tf = t_span
    num_steps = int((tf - t0) / step)  # Количество шагов на основе фиксированного шага
    t_values: np.ndarray = np.arange(t0, tf + step, step)  # Временные точки с фиксированным шагом
    Y: np.ndarray = np.zeros((6, num_steps + 1))  # Массив для хранения значений z, y, zm, ym, zp, yp
    Y[:, 0] = [0, 0, 0, 0, 0, 0]  # Начальные условия

    for i in range(num_steps):
        t = t_values[i]
        Y_current = Y[:, i]

        # Вычисляем четыре промежуточных значения для метода Рунге-Кутты
        k1 = np.array(system(t, Y_current, m_value, p_value))
        k2 = np.array(system(t + step / 2, Y_current + k1 * step / 2, m_value, p_value))
        k3 = np.array(system(t + step / 2, Y_current + k2 * step / 2, m_value, p_value))
        k4 = np.array(system(t + step, Y_current + k3 * step, m_value, p_value))

        # Обновляем значения с использованием формулы Рунге-Кутты
        Y[:, i + 1] = Y_current + (k1 + 2 * k2 + 2 * k3 + k4) * step / 6

    return t_values, Y  # Возвращаем временные точки и значения


def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Выполняет разложение Холецкого матрицы A.

    Параметры:
    A (np.ndarray): Квадратная матрица.

    Возвращает:
    np.ndarray: Нижняя треугольная матрица L.
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # диагональные элементы
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

def back_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решает систему Lx = b методом обратной подстановки.

    Параметры:
    L (np.ndarray): Нижняя треугольная матрица.
    b (np.ndarray): Вектор свободных членов.

    Возвращает:
    np.ndarray: Решение системы уравнений.
    """
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def gauss_newton(initial_m: float, initial_p: float, tol: float = 1e-6) -> list:
    """
    Реализует метод Гаусса-Ньютона для оптимизации параметров модели.

    Параметры:
    initial_m (float): Начальное значение массы.
    initial_p (float): Начальное значение давления.
    tol (float): Порог для остановки алгоритма.

    Возвращает:
    list: Оцененные значения массы и давления.
    """
    m_value = initial_m
    p_value = initial_p
    while True:
        t_values, solution = solve_model(m_value, p_value)
        y_model = solution[1]
        rm_model = solution[3]
        rp_model = solution[5]

        # Остатки
        data = np.loadtxt('data_rak.txt', delimiter=',', skiprows=1)
        y_data = data[:, 2]
        t_data = data[:, 0]
        y_model_fix = []
        rm_model_fix = []
        rp_model_fix = []
        for i in t_data:
            y_model_fix.append(y_model[int(i * 10)])
            rm_model_fix.append(rm_model[int(i * 10)])
            rp_model_fix.append(rp_model[int(i * 10)])



        residuals = y_model_fix - y_data

        # Вычисление Якобиана
        J = np.column_stack((rm_model_fix, rp_model_fix))

        # 1. Вычисляем J^T J
        JT_J = np.dot(J.T, J)

        # 2. Выполняем разложение Холецкого
        L = cholesky_decomposition(JT_J)

        # 3. Решаем систему L * y = J^T * residuals
        y = back_substitution(L, np.dot(J.T, residuals))

        # 4. Решаем систему L^T * delta = y
        delta = back_substitution(L.T, y)

        # Обновляем параметры
        m_value = m_value - delta[0]
        p_value = p_value - delta[1]

        los = np.sum((y_model_fix - y_data) ** 2)
        print("обновленное значение m:", m_value)
        print("обновленное значение p:", p_value)
        print("разность квадратов:", los)
        print('------------------------------------------------------')

        # Проверка на сходимость
        if (los < 10) or ((abs(delta[0]) < tol) and (abs(delta[1]) < tol)):
            return [m_value, p_value]

def main_estimation(initial_m: float, initial_p: float) -> list:
    print("начальное значение:", initial_m, initial_p)
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    m_estimated: list = gauss_newton(initial_m, initial_p)
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    print(f'Оцененное значение m: {m_estimated[0]}')
    print(f'Оцененное значение p: {m_estimated[1]}')
    print("Реальное значение m, p: 313000, 5042983")
    return m_estimated


#Построение графиков
def main_plotting(m_estimated):
    # Решение модели с оцененными параметрами
    t_values, final_solution = solve_model(m_estimated[0], m_estimated[1])

    # Извлечение значений
    z_values: np.ndarray = final_solution[0]
    y_values: np.ndarray = final_solution[1]
    zm_values: np.ndarray = final_solution[2]
    ym_values: np.ndarray = final_solution[3]
    zp_values: np.ndarray = final_solution[4]
    yp_values: np.ndarray = final_solution[5]


    #data_to_save = np.column_stack((t_values, z_values, y_values))  # Объединяем данные в один массив
    #np.savetxt('test_data_apri.txt', data_to_save, header='t, z, y', delimiter=',', fmt='%.15f')

    # Построение графиков
    plt.figure(figsize=(12, 10))

    # График z
    plt.subplot(3, 2, 1)
    plt.plot(t_values, z_values, label='z(t)', color='blue')
    plt.title('График z(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('z')
    plt.grid()
    plt.legend()

    # График y
    plt.subplot(3, 2, 2)
    plt.plot(t_values, y_values, label='y(t)', color='orange')
    plt.title('График y(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('y')
    plt.grid()
    plt.legend()

    # График zm
    plt.subplot(3, 2, 3)
    plt.plot(t_values, zm_values, label='zm(t)', color='green')
    plt.title('График zm(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('zm')
    plt.grid()
    plt.legend()

    # График ym
    plt.subplot(3, 2, 4)
    plt.plot(t_values, ym_values, label='ym(t)', color='red')
    plt.title('График ym(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('ym')
    plt.grid()
    plt.legend()

    # График zp
    plt.subplot(3, 2, 5)
    plt.plot(t_values, zp_values, label='zp(t)', color='purple')
    plt.title('График zp(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('zp')
    plt.grid()
    plt.legend()

    # График yp
    plt.subplot(3, 2, 6)
    plt.plot(t_values, yp_values, label='yp(t)', color='brown')
    plt.title('График yp(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('yp')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #Реальное значение m, p: 313000, 5042983
    #Начальные параметры
    initial_m: float = 300000
    initial_p: float = 5042988
    try:
        # Для получения оцененных параметров
        m_estimated = main_estimation(initial_m, initial_p)
    except:
        ("Плохое стартовое приближение")

    #Для построения графиков
    main_plotting([313000, 5042983])
