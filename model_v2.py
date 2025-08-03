import pandas as pd
from ortools.sat.python import cp_model
from collections import defaultdict

# 1. Definición exacta de los datos iniciales
classrooms = {
    '101': {'capacity': 15, 'schedule': {}},
    '102': {'capacity': 25, 'schedule': {}},
    '103': {'capacity': 20, 'schedule': {
        # B01AM (Interdiario L-M-V)
        ('LUN', '07:00-08:30'): 'B01AM',
        ('MIE', '07:00-08:30'): 'B01AM',
        ('VIE', '07:00-08:30'): 'B01AM',
        ('LUN', '08:45-10:15'): 'B01AM',
        ('MIE', '08:45-10:15'): 'B01AM',
        ('VIE', '08:45-10:15'): 'B01AM',
        # B00TT (Interdiario M-J)
        ('MAR', '07:00-08:30'): 'B00TT',
        ('JUE', '07:00-08:30'): 'B00TT',
        ('MAR', '08:45-10:15'): 'B00TT',
        ('JUE', '08:45-10:15'): 'B00TT'
    }},
    '104': {'capacity': 25, 'schedule': {}},
    '105': {'capacity': 20, 'schedule': {
        # B01RP (Diario)
        ('LUN', '08:45-10:15'): 'B01RP',
        ('MAR', '08:45-10:15'): 'B01RP',
        ('MIE', '08:45-10:15'): 'B01RP',
        ('JUE', '08:45-10:15'): 'B01RP',
        ('VIE', '08:45-10:15'): 'B01RP'
    }}
}

# Mapeo exacto de frecuencias
frequency_mapping = {
    "Diario": ["LUN", "MAR", "MIE", "JUE", "VIE"],
    "Interdiario L-M-V": ["LUN", "MIE", "VIE"],
    "Interdiario M-J": ["MAR", "JUE"]
}

# Horarios compuestos
time_slots = ['07:00-08:30', '08:45-10:15', '10:30-12:00']
composite_slots = {
    '07:00-10:15': ['07:00-08:30', '08:45-10:15'],
    '07:00-08:30': ['07:00-08:30'],
    '08:45-10:15': ['08:45-10:15'],
    '10:30-12:00': ['10:30-12:00']
}

# Cursos a asignar
courses_to_assign = [
    {'code': 'B02RP', 'schedule': '07:00-08:30', 'students': 15, 'frequency': 'Diario'},
    {'code': 'B03RP', 'schedule': '08:45-10:15', 'students': 25, 'frequency': 'Diario'},
    {'code': 'B03AM', 'schedule': '07:00-10:15', 'students': 15, 'frequency': 'Interdiario L-M-V'},
    {'code': 'B03TT', 'schedule': '07:00-10:15', 'students': 15, 'frequency': 'Interdiario M-J'}
]

# 2. Creación del modelo de optimización
model = cp_model.CpModel()

# Variables de decisión
assignments = {course['code']: model.NewIntVar(0, len(classrooms) - 1, f"assign_{course['code']}")
               for course in courses_to_assign}

# Variables auxiliares para la función objetivo
utilization_vars = []
utilization_coeffs = []

# 3. Definición de restricciones
for course in courses_to_assign:
    assigned_room = assignments[course['code']]
    required_slots = composite_slots[course['schedule']]
    days = frequency_mapping[course['frequency']]

    for j, (room, data) in enumerate(classrooms.items()):
        # Restricción de capacidad
        if data['capacity'] < course['students']:
            model.Add(assigned_room != j)
            continue

        # Verificación de disponibilidad
        conflict = False
        for slot in required_slots:
            for day in days:
                if (day, slot) in data['schedule']:
                    conflict = True
                    break
            if conflict:
                break

        if conflict:
            model.Add(assigned_room != j)
        else:
            # Contribución a la función objetivo
            utilization = course['students'] / data['capacity']
            var = model.NewBoolVar(f'util_{course["code"]}_{room}')
            model.Add(assigned_room == j).OnlyEnforceIf(var)
            model.Add(assigned_room != j).OnlyEnforceIf(var.Not())
            utilization_vars.append(var)
            utilization_coeffs.append(int(utilization * 100))

# 4. Función objetivo: maximizar la utilización
model.Maximize(sum(
    var * coeff for var, coeff in zip(utilization_vars, utilization_coeffs)
))

# 5. Resolución del modelo
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
status = solver.Solve(model)

# 6. Visualización de resultados completos
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("=== SOLUCIÓN ÓPTIMA ENCONTRADA ===")
    print(f"Valor objetivo (utilización total): {solver.ObjectiveValue() / 100:.2f}\n")

    # Mostrar asignaciones existentes
    print("=== ASIGNACIONES EXISTENTES ===")
    for room, data in classrooms.items():
        if data['schedule']:
            print(f"\nAula {room} (Capacidad: {data['capacity']}):")

            # Agrupar por curso
            course_slots = defaultdict(list)
            for (day, slot), course in data['schedule'].items():
                course_slots[course].append((day, slot))

            for course, slots in course_slots.items():
                # Determinar frecuencia del curso
                freq = next((freq for freq, days in frequency_mapping.items()
                             if all(slot[0] in days for slot in slots)), "Desconocida")
                print(f"  Curso {course} ({freq}):")
                for day, slot in sorted(slots):
                    print(f"    {day} {slot}")

    # Procesar nuevas asignaciones
    print("\n=== NUEVAS ASIGNACIONES ===")
    new_assignments = []
    for course in courses_to_assign:
        room_idx = solver.Value(assignments[course['code']])
        room = list(classrooms.keys())[room_idx]
        days = frequency_mapping[course['frequency']]
        slots = composite_slots[course['schedule']]

        # Registrar la asignación
        for day in days:
            for slot in slots:
                classrooms[room]['schedule'][(day, slot)] = course['code']

        new_assignments.append({
            'Curso': course['code'],
            'Aula': room,
            'Capacidad': classrooms[room]['capacity'],
            'Estudiantes': course['students'],
            'Utilización': f"{course['students'] / classrooms[room]['capacity']:.1%}",
            'Horario': course['schedule'],
            'Frecuencia': f"{course['frequency']} ({', '.join(days)})",
            'Franjas': [f"{day} {slot}" for day in days for slot in slots]
        })

    # Mostrar nuevas asignaciones en formato tabla
    df_new = pd.DataFrame(new_assignments)
    print(df_new[['Curso', 'Aula', 'Capacidad', 'Estudiantes', 'Utilización', 'Frecuencia']].to_string(index=False))

    # Mostrar detalles de franjas horarias
    print("\nDetalle de franjas horarias asignadas:")
    for assignment in new_assignments:
        print(f"\nCurso: {assignment['Curso']}")
        print(f"Aula: {assignment['Aula']}")
        for slot in assignment['Franjas']:
            print(f"  {slot}")

    # Resumen de utilización
    print("\n=== RESUMEN FINAL DE UTILIZACIÓN ===")
    utilization_data = []
    for room, data in classrooms.items():
        total_slots = len(time_slots) * 5  # 5 días por semana
        used_slots = len(data['schedule'])
        util_percent = used_slots / total_slots
        utilization_data.append({
            'Aula': room,
            'Capacidad': data['capacity'],
            'Franjas Ocupadas': f"{used_slots}/{total_slots}",
            'Utilización': f"{util_percent:.1%}"
        })

    df_util = pd.DataFrame(utilization_data)
    print(df_util.to_string(index=False))

else:
    print("No se encontró una solución factible para todas las asignaciones requeridas.")