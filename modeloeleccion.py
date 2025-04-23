from datetime import datetime

class Empleado:
    def __init__(self, nombre, antiguedad_anos, fecha_solicitud, tiene_hijos, tuvo_vacaciones_deseadas_ano_pasado,
                 dias_acumulados, dias_disfrutados, rol_critico):
        self.nombre = nombre
        self.antiguedad_anos = antiguedad_anos
        self.fecha_solicitud = datetime.strptime(fecha_solicitud, "%Y-%m-%d")
        self.tiene_hijos = tiene_hijos
        self.tuvo_vacaciones_deseadas_ano_pasado = tuvo_vacaciones_deseadas_ano_pasado
        self.dias_acumulados = dias_acumulados
        self.dias_disfrutados = dias_disfrutados
        self.rol_critico = rol_critico

    def prioridad(self):
        score = 0

        # Antigüedad
        score += self.antiguedad_anos * 2

        # Situación familiar
        if self.tiene_hijos:
            score += 3

        # Equidad
        if not self.tuvo_vacaciones_deseadas_ano_pasado:
            score += 2

        # Días acumulados vs disfrutados
        if self.dias_disfrutados == 0:
            score += 4  # No ha tenido vacaciones este año
        score += (self.dias_acumulados - self.dias_disfrutados) * 0.5

        # Rol crítico (penalización)
        if self.rol_critico:
            score -= 3

        # Fecha de solicitud (más temprano = más puntos)
        dias_desde_solicitud = (datetime.now() - self.fecha_solicitud).days
        score += dias_desde_solicitud * 0.1

        return score


def ordenar_por_prioridad(empleados):
    return sorted(empleados, key=lambda e: e.prioridad(), reverse=True)


# Ejemplo de uso
empleados = [
    Empleado("Ana", 5, "2025-03-01", True, False, 15, 0, False),
    Empleado("Luis", 2, "2025-04-10", False, True, 10, 5, True),
    Empleado("Marta", 7, "2025-02-15", True, False, 20, 10, False),
    Empleado("Carlos", 1, "2025-04-01", False, False, 12, 0, False)
]

prioridad = ordenar_por_prioridad(empleados)
for e in prioridad:
    print(f"{e.nombre}: prioridad {e.prioridad():.2f}")