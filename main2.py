import numpy as np
import math
import matplotlib.pyplot as plt

def system(t: float, Y: np.ndarray, m_value: float, p_value: float, derivatives: np.ndarray) -> None:
    pressure_term = p_value - 8.3938 * pl(Y[1])
    mass_term = m_value - 1377 * t

    # Основное уравнение
    derivatives[0] = (pressure_term - F(Y[1], Y[0])) / mass_term - g(Y[1])  # производная z
    derivatives[1] = Y[0]  # производная y

    # Изохронная производная по m0
    derivatives[2] = -(pressure_term - F(Y[1], Y[0])) / mass_term ** 2  # производная zm
    derivatives[3] = Y[2]

    # Изохронная производная по p0
    derivatives[4] = 1 / mass_term  # производная zp
    derivatives[5] = Y[4]  # производная yp


def F(h: float, v: float) -> float:
    """
        Вычисляет силу сопротивления воздуха.

        Параметры:
        h (float): Высота.
        v (float): Скорость.

        Возвращает:
        float: Сила сопротивления.
        """
    return 0.8 * 0.5 * 7.9 * d(h) * v ** 2

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
    return 288.15 - 0.0065 * h if h <= 10000 else 223

def m(t: float) -> float:
    """
        Вычисляет массу объекта в зависимости от времени.

        Параметры:
        t (float): Время.

        Возвращает:
        float: Масса.
        """
    return 313000 - 1377 * t

def g(h: float) -> float:
    """
        Вычисляет ускорение свободного падения на высоте.

        Параметры:
        h (float): Высота.

        Возвращает:
        float: Ускорение свободного падения (м/с²).
        """
    return 9.8 * 6_371_000 ** 2 / (6_371_000 + h) ** 2

def P(p: float, h: float) -> float:
    """
        Вычисляет давление в зависимости от высоты.

        Параметры:
        p (float): Давление.
        h (float): Высота.

        Возвращает:
        float: Давление на высоте.
        """
    return p - 8.3938 * pl(h)

def pl(h: float) -> float:
    """
        Вычисляет плотность воздуха на высоте.

        Параметры:
        h (float): Высота.

        Возвращает:
        float: Давление воздуха.
        """
    return 101325 * math.exp(-h / 8400)

def solve_model(m_value: float, p_value: float, solution: np.ndarray, t_values: np.ndarray,num_steps, dt, half_dt, k, temp) -> tuple:
    """
        Решает модель с использованием метода Рунге-Кутты 4-го порядка.

        Параметры:
        m_value (float): Значение массы.
        p_value (float): Значение давления.
        """
    for i in range(num_steps):
        system(t_values[i], solution[:, i], m_value, p_value, k[:, 0])  # k1
        # Вычисляем k2
        np.multiply(k[:, 0], half_dt, out=temp)
        np.add(solution[:, i], temp, out=temp)
        system(t_values[i] + half_dt, temp, m_value, p_value, k[:, 1])  # k2

        # Вычисляем k3
        np.multiply(k[:, 1], half_dt, out=temp)
        np.add(solution[:, i], temp, out=temp)
        system(t_values[i] + half_dt, temp, m_value, p_value, k[:, 2])  # k3

        # Вычисляем k4
        np.multiply(k[:, 2], dt, out=temp)
        np.add(solution[:, i], temp, out=temp)
        system(t_values[i] + dt, temp, m_value, p_value, k[:, 3])  # k4

        #solution[:, i + 1] = solution[:, i] + (k[:, 0] + 2 * k[:, 1] + 2 * k[:, 2] + k[:, 3]) * dt / 6
        solution[:, i + 1] = k[:, 0]
        np.add(solution[:, i + 1], k[:, 1], out=solution[:, i + 1])
        np.add(solution[:, i + 1], k[:, 1], out=solution[:, i + 1])
        np.add(solution[:, i + 1], k[:, 2], out=solution[:, i + 1])
        np.add(solution[:, i + 1], k[:, 2], out=solution[:, i + 1])
        np.add(solution[:, i + 1], k[:, 3], out=solution[:, i + 1])
        np.multiply(solution[:, i + 1], dt/6, out=solution[:, i + 1])
        np.add(solution[:, i + 1], solution[:, i], out=solution[:, i + 1])


def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Выполняет разложение Холецкого матрицы A.

    Параметры:
    A (np.ndarray): Квадратная матрица.

    Возвращает:
    np.ndarray: Нижняя треугольная матрица L.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            sum_l = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_l)
            else:
                L[i, j] = (A[i, j] - sum_l) / L[j, j]
    return L

def back_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решает систему Lx = b методом обратной подстановки.

    Параметры:
    L (np.ndarray): треугольная матрица.
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
    Реализует метод Гаусса-Ньютона для уточнения параметров модели.

    Параметры:
    initial_m (float): Начальное значение массы.
    initial_p (float): Начальное значение давления.
    tol (float): Порог для остановки алгоритма.

    Возвращает:
    list: Оцененные значения массы и давления.
    """
    m_value = initial_m
    p_value = initial_p
    data = np.loadtxt('solution_data_shum.txt', delimiter=',', skiprows=1)

    residuals = np.zeros_like(data[:, 2])
    delta = np.zeros(2)
    J = np.zeros((len(data[:, 2]), 2))

    steps_count = 1000
    # Выделяем память для solution
    solution = np.zeros((6, steps_count))
    t_values = np.linspace(0, 118, steps_count)
    num_steps = len(t_values) - 1
    dt = t_values[1] - t_values[0]
    half_dt = dt / 2
    k = np.zeros((6, 4))
    temp = np.empty_like(solution[:, 0])

    while True:
        solve_model(m_value, p_value, solution, t_values, num_steps, dt, half_dt, k, temp)

        residuals[:] = solution[1] - data[:, 2]

        J[:, 0] = solution[3]
        J[:, 1] = solution[5]

        JT_J = np.dot(J.T, J)
        L = cholesky_decomposition(JT_J)
        y = back_substitution(L, np.dot(J.T, residuals))
        delta[:] = back_substitution(L.T, y)

        m_value -= delta[0]
        p_value -= delta[1]

        #Разность квадратов
        los = np.sum(residuals ** 2)
        print("обновленное значение m:", m_value)
        print("обновленное значение p:", p_value)
        print("разность квадратов:", los)
        print('------------------------------------------------------')

        # Завершение если разность квадратов < 10 или изменения аргументов незначительны
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
    print("Реальное значение m, p: 313000, 4997904")
    return m_estimated

def main_init(flag, m = 313000, p = 4997904):
    final_solution = np.zeros((6, 1000))  # Здесь 1000 - максимальное количество шагов, можно изменить по необходимости
    t_values = np.linspace(0, 118, 1000)
    num_steps = len(t_values) - 1
    dt = t_values[1] - t_values[0]
    half_dt = dt / 2
    k = np.zeros((6, 4))
    temp = np.empty_like(final_solution[:, 0])
    solve_model(m, p, final_solution, t_values, num_steps, dt, half_dt, k, temp)

    z_values = final_solution[0]
    y_values = final_solution[1]

     #Генерация шума
    if flag:
        noise = 0.01 * y_values * np.random.randn(*y_values.shape)  # Генерируем случайный шум
        y_values = y_values + noise
    data_to_save = np.column_stack((t_values, z_values, y_values))  # Объединяем данные в один массив
    np.savetxt('solution_data_shum.txt', data_to_save, header='t, z, y_noisy', delimiter=',', fmt='%.15f')


def main_plotting(m_estimated):
    final_solution = np.zeros((6, 1000))  # Здесь 1000 - максимальное количество шагов, можно изменить по необходимости
    t_values = np.linspace(0, 118, 1000)
    num_steps = len(t_values) - 1
    dt = t_values[1] - t_values[0]
    half_dt = dt / 2
    k = np.zeros((6, 4))
    temp = np.empty_like(final_solution[:, 0])
    solve_model(m_estimated[0], m_estimated[1], final_solution, t_values, num_steps, dt, half_dt, k, temp)

    z_values = final_solution[0]
    y_values = final_solution[1]
    zm_values = final_solution[2]
    ym_values = final_solution[3]
    zp_values = final_solution[4]
    yp_values = final_solution[5]

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(t_values, z_values, label='z(t)', color='blue')
    plt.title('График z(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('z')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(t_values, y_values, label='y(t)', color='orange')
    plt.title('График y(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('y')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(t_values, zm_values, label='zm(t)', color='green')
    plt.title('График zm(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('zm')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(t_values, ym_values, label='ym(t)', color='red')
    plt.title('График ym(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('ym')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(t_values, zp_values, label='zp(t)', color='purple')
    plt.title('График zp(t)')
    plt.xlabel('Время (t)')
    plt.ylabel('zp')
    plt.grid()
    plt.legend()

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
    # Реальное значение m, p: 313000, 4997904
    # Начальные параметр

    # Создание данных
    shum = True # Для создания данных с шумом True. Без шума False
    graph = False  # Для построение графика
    main_init(shum)

    initial_m: float = 313000
    initial_p: float = 4997904
    try:
        m_estimated = main_estimation(initial_m, initial_p)
    except Exception as e:
        print("Плохое стартовое приближение:", e)


    if graph:
        main_plotting(m_estimated)
