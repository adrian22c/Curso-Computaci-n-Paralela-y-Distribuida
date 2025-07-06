# Primera parte (Tarea 1)

from re import Pattern
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

try:
    profile
except NameError:
    def profile(func): return func


# Crear carpetas para guardar salida
os.makedirs("graficas", exist_ok=True)
os.makedirs("animaciones", exist_ok=True)
os.makedirs("perfilado", exist_ok=True)

#  Implementación orientada a objetos
'''
 Desarrolle una clase GameOfLife en Python que contenga al menos los siguientes métodos:
 • __init__(self, rows, cols, initial_state=None): para inicializar el tablero.
 • step(self): que actualiza el estado del tablero según las reglas de Conway.
 • run(self, steps): para ejecutar múltiples iteraciones del juego.
 • get_state(self): que devuelve el estado actual del tablero.
'''
class GameOfLife:
    def __init__(self, rows, cols, initial_state=None):
        self.rows = rows
        self.cols = cols
        if initial_state is None:
            self.board = np.random.randint(2, size=(rows, cols))
        else:
            self.board = np.array(initial_state, dtype=int)

    @profile
    def count_neighbors(self, r, c):
        total = 0
        for i in range(r-1, r+2):
            for j in range(c-1, c+2):
                if (i == r and j == c) or i < 0 or j < 0 or i >= self.rows or j >= self.cols:
                    continue
                total += self.board[i, j]
        return total
    
    @profile
    def step(self):
        new_board = np.copy(self.board)
        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = self.count_neighbors(r, c)
                if self.board[r, c] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_board[r, c] = 0
                else:
                    if neighbors == 3:
                        new_board[r, c] = 1
        self.board = new_board

    def run(self, steps):
        for _ in range(steps):
            self.step()

    def get_state(self):
        return self.board

# Visualización
'''
Utilice matplotlib o matplotlib.animation para visualizar la evolución del juego.
Cree animaciones o secuencias de imágenes para representar gráficamente la evolución de patrones clásicos (como Glider, Blinker, Toad, etc.).
La visualización debe ser capaz de funcionar en diferentes tamaños de grilla (e.g., 32x32, 128x128, 512x512).
'''
def animate_game(game, steps=100, interval=200, save=False, filename="glider_32x32.gif"):
    fig, ax = plt.subplots()
    img = ax.imshow(game.get_state(), cmap='binary', interpolation='nearest')
    ax.set_title("Evolución del Juego de la Vida")

    def update(frame):
        game.step()
        img.set_data(game.get_state())
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=steps, interval=interval, blit=True, repeat=False
    )

    if save:
        ani.save(f"animaciones/{filename}", writer='pillow')
        plt.close()
    else:
        plt.show()

# Glider en una grilla 32x32
glider = np.zeros((32, 32), dtype=int)
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1

# Inicialización del juego con patrón Glider
game = GameOfLife(32, 32, glider)
animate_game(game, steps=100, save=True, filename="glider_32x32.gif")

# Patrones clasiicos
def glider_pattern(size=32):
    board = np.zeros((size, size), dtype=int)
    board[1, 2] = board[2, 3] = board[3, 1] = board[3, 2] = board[3, 3] = 1
    return board

def blinker_pattern(size=32):
    board = np.zeros((size, size), dtype=int)
    center = size // 2
    board[center, center-1:center+2] = 1
    return board

def toad_pattern(size=32):
    board = np.zeros((size, size), dtype=int)
    center = size // 2
    board[center, center:center+3] = 1
    board[center+1, center-1:center+2] = 1
    return board

def block_pattern(size=32):
    board = np.zeros((size, size), dtype=int)
    center = size // 2
    board[center:center+2, center:center+2] = 1
    return board

pattern = glider_pattern(32)  # o blinker_pattern(), toad_pattern(), block_pattern()
game = GameOfLife(32, 32, pattern)
animate_game(game, steps=100)

# Medición de rendimiento y complejidad empírica
'''
Realice pruebas de rendimiento empíricas variando el tamaño de la grilla (por ejemplo: 32x32, 64x64, 128x128, ..., 1024x1024).
Para cada tamaño, mida el tiempo promedio de ejecución por iteración del juego.
Presente una gráfica de tiempo vs tamaño de entrada (número de celdas) y compare con curvas teóricas de complejidad (O(n), O(nlogn), O(n2), etc.).
Incluya al menos una visualización log-log.
'''
def medir_rendimiento(tamanos, steps=10):
    tiempos = []

    print("\n--- Medición de Rendimiento ---")
    for size in tamanos:
        print(f"\n Ejecutando simulación para grilla {size}x{size}")
        game = GameOfLife(size, size)
        inicio = time.perf_counter()
        game.run(steps)
        fin = time.perf_counter()
        tiempo_total = fin - inicio
        tiempo_promedio = tiempo_total / steps
        print(f"Tiempo total para {steps} pasos: {tiempo_total:.4f} s")
        print(f"Tiempo promedio por iteración: {tiempo_promedio:.6f} s")
        tiempos.append(tiempo_promedio)

    return tiempos

tamanos = [32, 64, 128, 256, 512, 1024]
tiempos = medir_rendimiento(tamanos, steps=5)
celulas = [n * n for n in tamanos]

# Gráfica lineal
plt.figure(figsize=(10, 5))
plt.plot(celulas, tiempos, marker='o', label="Medido")
plt.xlabel("Número de celdas (n x n)")
plt.ylabel("Tiempo promedio por iteración (s)")
plt.title("Rendimiento del Juego de la Vida")
plt.grid(True)
plt.legend()
plt.savefig("graficas/rendimiento_lineal.png")
plt.close()

# Gráfica log-log
plt.figure(figsize=(10, 5))
plt.loglog(celulas, tiempos, marker='o', label="Medido")
plt.xlabel("Número de celdas (log escala)")
plt.ylabel("Tiempo promedio (log escala)")
plt.title("Escala Log-Log del Rendimiento")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("graficas/rendimiento_loglog.png")
plt.close()

#Segunda parte (Tarea 2)

# =======================================
# PARTE 1 – ANÁLISIS CON cProfile
# =======================================
"""
Esta sección ejecuta una simulación del Juego de la Vida en una grilla de 512x512 durante 100 pasos,
utilizando la herramienta de perfilado cProfile para identificar funciones costosas.
El resultado se guarda en el archivo 'perfil_cprofile.txt'.
"""
import cProfile
import pstats

def ejecutar_simulacion():
    pattern = glider_pattern(512)
    juego = GameOfLife(512, 512, pattern)
    juego.run(100)

if __name__ == "__main__":
    # Activar esta sección si desea ejecutar el análisis con cProfile
    activar_perfilado = True

    if activar_perfilado:
        with open("perfilado/perfil_cprofile.txt", "w") as f:
            profile = cProfile.Profile()
            profile.enable()
            ejecutar_simulacion()
            profile.disable()
            stats = pstats.Stats(profile, stream=f)
            stats.sort_stats("cumtime")  # Ordenar por tiempo acumulado
            stats.print_stats()
        print("\n Análisis con cProfile guardado en 'perfil_cprofile.txt'")

# =======================================
# PARTE 2 – ANÁLISIS CON line_profiler
# =======================================
"""
Esta sección está diseñada para ser ejecutada con la herramienta line_profiler, 
la cual permite analizar el tiempo de ejecución línea por línea en funciones críticas 
como 'step' y 'count_neighbors'. 

Estas funciones han sido decoradas previamente con @profile, requisito indispensable 
para que kernprof pueda analizarlas correctamente.

Nota: Esta sección no se ejecuta automáticamente con el script normal, 
ya que debe ejecutarse con kernprof para ser útil.
"""

def prueba_line_profiler():
    """
    Ejecuta una simulación moderada del Juego de la Vida sobre una grilla 256x256
    durante 50 pasos. Esta prueba es suficientemente representativa para observar 
    el comportamiento interno de las funciones clave usando line_profiler.
    """
    pattern = glider_pattern(256)
    juego = GameOfLife(256, 256, pattern)
    juego.run(50)




# =======================================
# PARTE 3 – ANÁLISIS DE ESCALABILIDAD
# =======================================

from multiprocessing import Process, Array, Barrier, cpu_count
import ctypes

'''
Clase paralela del Juego de la Vida
En esta parte desarrollé una versión del Game of Life que permite ejecutarse usando múltiples procesos
gracias al módulo multiprocessing de Python.
La idea es dividir el tablero entre varios procesos que trabajen simultáneamente,
lo que permite evaluar cómo se comporta el rendimiento del programa conforme se agregan más recursos.
'''

class GameOfLifeParallel:
    def __init__(self, rows, cols, steps, num_procs):
        # Inicializo los parámetros del tablero, cantidad de pasos y procesos
        self.rows = rows
        self.cols = cols
        self.steps = steps
        self.num_procs = num_procs

        # Creo una estructura compartida de memoria para el tablero, usando un Array de ctypes enteros
        # Esto permite que todos los procesos puedan leer y escribir sobre el mismo tablero sin conflictos
        self.shared_board = Array(ctypes.c_int, rows * cols, lock=False)

        # Creo una barrera para sincronizar todos los procesos en cada iteración del juego
        self.barrier = Barrier(num_procs)

        # Inicializo el tablero con el patrón Glider en una grilla del tamaño indicado
        board = glider_pattern(rows)
        flat = board.flatten()
        for i in range(len(flat)):
            self.shared_board[i] = flat[i]

    def run(self):
        # Esta función lanza todos los procesos y espera a que terminen
        procesos = []
        for pid in range(self.num_procs):
            p = Process(target=self.worker, args=(pid,))
            procesos.append(p)
            p.start()

        for p in procesos:
            p.join()

    def worker(self, pid):
        # Cada proceso va a manejar una porción del tablero, calculada dividiendo las filas equitativamente
        chunk = self.rows // self.num_procs
        start = pid * chunk
        end = (pid + 1) * chunk if pid != self.num_procs - 1 else self.rows

        for _ in range(self.steps):
            new_data = []  # Almacena las celdas que deben cambiar de estado

            # Recorro solo la parte del tablero que le toca a este proceso
            for r in range(start, end):
                for c in range(self.cols):
                    neighbors = self.count_neighbors(r, c)
                    idx = r * self.cols + c
                    cell = self.shared_board[idx]

                    # Aplico las reglas del Juego de la Vida
                    if cell == 1 and (neighbors < 2 or neighbors > 3):
                        new_data.append((idx, 0))  # Muerte por soledad o sobrepoblación
                    elif cell == 0 and neighbors == 3:
                        new_data.append((idx, 1))  # Nace una nueva célula

            # Espero a que todos los procesos terminen de analizar su parte
            self.barrier.wait()

            # Ahora se aplican todos los cambios al tablero compartido
            for idx, val in new_data:
                self.shared_board[idx] = val

            # Espero nuevamente antes de pasar a la siguiente iteración
            self.barrier.wait()

    def count_neighbors(self, r, c):
        # Función auxiliar para contar los vecinos vivos de una celda (r, c)
        total = 0
        for i in range(r - 1, r + 2):
            for j in range(c - 1, c + 2):
                if i == r and j == c:
                    continue  # No se cuenta a sí misma
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    idx = i * self.cols + j
                    total += self.shared_board[idx]
        return total

'''
ESCALAMIENTO FUERTE

En esta prueba, mantengo constante el tamaño de la grilla (512x512)
y evalúo cómo mejora el tiempo de ejecución al aumentar la cantidad de procesos.
'''
def test_escalamiento_fuerte(fixed_size=512, steps=50):
    procesos = [1, 2, 4, 8]  # Número de procesos a probar
    tiempos = []

    for p in procesos:
        print(f"\n>> Ejecutando con {p} procesos")
        start = time.perf_counter()
        juego = GameOfLifeParallel(fixed_size, fixed_size, steps, p)
        juego.run()
        end = time.perf_counter()
        tiempo = end - start
        print(f"Tiempo: {tiempo:.4f} s")
        tiempos.append(tiempo)

    # Cálculo del speedup y eficiencia comparando contra el tiempo con 1 proceso
    t1 = tiempos[0]
    speedup = [t1 / t for t in tiempos]
    eficiencia = [s / p for s, p in zip(speedup, procesos)]

    return procesos, tiempos, speedup, eficiencia

'''
Esta función grafica los resultados del escalamiento fuerte:
Tiempo total, Speedup y Eficiencia en función de la cantidad de procesos
'''
def graficar_escalamiento_fuerte(procesos, tiempos, speedup, eficiencia):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(procesos, tiempos, marker='o')
    plt.title("Tiempo vs Procesos")
    plt.xlabel("Procesos")
    plt.ylabel("Tiempo (s)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(procesos, speedup, marker='o')
    plt.title("Speedup")
    plt.xlabel("Procesos")
    plt.ylabel("Speedup")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(procesos, eficiencia, marker='o')
    plt.title("Eficiencia")
    plt.xlabel("Procesos")
    plt.ylabel("Eficiencia")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("graficas/escalamiento_fuerte.png")
    plt.close()

'''
ESCALAMIENTO DÉBIL

En esta prueba, aumento el tamaño de la grilla proporcionalmente a la cantidad de procesos,
manteniendo constante la cantidad de celdas por proceso.
Esto permite evaluar si el sistema mantiene el rendimiento al escalar la carga total.
'''
def test_escalamiento_debil(celdas_por_proc=100*100, steps=50):
    procesos = [1, 2, 4, 8]
    tiempos = []

    for p in procesos:
        n_celdas = celdas_por_proc * p
        lado = int(np.sqrt(n_celdas))
        lado = lado + (lado % 2)  # Me aseguro que sea par
        print(f"\n>> {p} procesos con grilla {lado}x{lado}")
        start = time.perf_counter()
        juego = GameOfLifeParallel(lado, lado, steps, p)
        juego.run()
        end = time.perf_counter()
        tiempo = end - start
        print(f"Tiempo: {tiempo:.4f} s")
        tiempos.append(tiempo)

    return procesos, tiempos

# Esta función grafica los tiempos obtenidos en la prueba de escalamiento débil
def graficar_escalamiento_debil(procesos, tiempos):
    plt.figure(figsize=(6, 4))
    plt.plot(procesos, tiempos, marker='o')
    plt.title("Escalamiento Débil")
    plt.xlabel("Procesos")
    plt.ylabel("Tiempo Total (s)")
    plt.grid(True)
    plt.savefig("graficas/escalamiento_debil.png")
    plt.close()