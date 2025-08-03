import numpy as np
import random
from collections import defaultdict, deque
import torch
import pandas as pd
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def read_json(path_file:str):
    with open(path_file, 'r') as file:
        data_file = json.load(file)
    return data_file


def read_csv(path_file:str):
    df = pd.read_csv(path_file)
    return df.to_dict(orient='records')


# 1. Environment Definition (Fixed)
class ClassroomEnv:
    def __init__(self):
        sede = 'Ica'
        periodo = '202506'
        classrooms = read_json('classrooms.json')

        self.classrooms = classrooms[periodo][sede]

        self.frequency_mapping = {
            "Diario": ["LUN", "MAR", "MIE", "JUE", "VIE"],
            "Interdiario L-M-V": ["LUN", "MIE", "VIE"],
            "Interdiario M-J": ["MAR", "JUE"],
            "Sabatino": ["SAB"]
        }

        lv = ['07:00 - 08:30', '08:45 - 10:15', '10:30 - 12:00', '12:30 - 14:00',
              '12:30 - 14:00', '14:15 - 15:45', '16:00 - 17:30', '17:45 - 19:15',
              '19:30 - 21:00', '21:15 - 22:45']

        sab = ['07:00 - 08:41', '08:55 - 10:36',
               '10:50 - 12:31', '12:45 - 14:26',
               '14:40 - 16:21', '16:35 - 18:16']

        self.time_slots = {
            "Diario": lv,
            "Interdiario L-M-V": lv,
            "Interdiario M-J": lv,
            "Sabatino": sab
        }

        self.composite_slots = {
            '07:00 - 10:15': ['07:00 - 08:30', '08:45 - 10:15'],
            '07:00 - 08:30': ['07:00 - 08:30'],
            '08:45 - 10:15': ['08:45 - 10:15'],
            '10:30 - 12:00': ['10:30 - 12:00']
        }

        self.courses_to_assign = read_csv('courses_to_assign.csv')

        self.reset()

    def reset(self):
        """Reset environment while keeping existing assignments"""
        # Create a fresh copy of classrooms preserving existing schedules
        self.temp_classrooms = {
            room: {
                'capacity': data['capacity'],
                'schedule': defaultdict(str, data['schedule'].copy())
            }
            for room, data in self.classrooms.items()
        }
        self.current_course_idx = 0
        return self._get_state()

    def _get_state(self):
        """Improved state representation"""
        if self.current_course_idx >= len(self.courses_to_assign):
            return None

        current_course = self.courses_to_assign[self.current_course_idx]
        days = self.frequency_mapping[current_course['frequency']]
        slots = self.composite_slots[current_course['schedule']]

        # State features per classroom
        state_features = []
        for room, data in self.temp_classrooms.items():
            # Feature 1: Capacity ratio (current course students / room capacity)
            capacity_ratio = current_course['students'] / data['capacity']

            # Feature 2: Current utilization percentage
            current_util = len(data['schedule']) / (len(self.time_slots) * 5)

            # Feature 3: Conflict flag (1 if no conflict, 0 if conflict)
            conflict = any(
                (day, slot) in data['schedule']
                for day in days
                for slot in slots
            )
            no_conflict = float(not conflict)

            # Feature 4: Perfect fit (1 if exact capacity match)
            perfect_fit = float(current_course['students'] == data['capacity'])

            state_features.extend([capacity_ratio, current_util, no_conflict, perfect_fit])

        # Course features
        state_features.extend([
            current_course['students'] / 30,  # Normalized student count
            len(days) / 5,  # Normalized day count
            len(slots)  # Number of slots
        ])

        return np.array(state_features, dtype=np.float32)

    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        if self.current_course_idx >= len(self.courses_to_assign):
            return None, 0, True, {}

        current_course = self.courses_to_assign[self.current_course_idx]
        room = list(self.temp_classrooms.keys())[action]
        room_data = self.temp_classrooms[room]

        days = self.frequency_mapping[current_course['frequency']]
        slots = self.composite_slots[current_course['schedule']]

        # Check validity
        valid = True
        reward = 0

        # Capacity check
        if room_data['capacity'] < current_course['students']:
            reward = -10
            valid = False
        else:
            # Availability check
            conflict = any(
                (day, slot) in room_data['schedule']
                for day in days
                for slot in slots
            )

            if conflict:
                reward = -5
                valid = False
            else:
                # Valid assignment
                utilization = current_course['students'] / room_data['capacity']
                reward = 10 + 5 * utilization  # Base reward + utilization bonus

                # Make the assignment
                for day in days:
                    for slot in slots:
                        room_data['schedule'][(day, slot)] = current_course['code']

        # Move to next course
        self.current_course_idx += 1
        next_state = self._get_state()
        done = next_state is None

        return next_state, reward, done, {
            'valid': valid,
            'course': current_course,
            'room': room
        }


# 2. DQN Model (Improved)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 3. DQN Agent (Fixed)
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = len(env._get_state())
        self.action_size = len(env.classrooms)

        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 100
        self.steps = 0

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]

        target = rewards + (1 - dones) * self.gamma * next_q
        loss = F.mse_loss(current_q.squeeze(), target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def train(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                if next_state is not None:
                    self.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                self.steps += 1

                loss = self.replay()

                if self.steps % self.update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            if e % 10 == 0:
                print(f"Episode: {e}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

    def get_assignments(self):
        """Get optimal assignments using current policy"""
        assignments = []
        state = self.env.reset()
        done = False

        while not done:
            with torch.no_grad():
                action = torch.argmax(self.model(torch.FloatTensor(state).unsqueeze(0))).item()

            next_state, _, done, info = self.env.step(action)

            if info['valid']:
                course = info['course']
                room = info['room']
                room_data = self.env.temp_classrooms[room]

                assignments.append({
                    'course': course['code'],
                    'room': room,
                    'capacity': room_data['capacity'],
                    'students': course['students'],
                    'utilization': course['students'] / room_data['capacity'],
                    'schedule': course['schedule'],
                    'frequency': course['frequency'],
                    'days': self.env.frequency_mapping[course['frequency']],
                    'slots': self.env.composite_slots[course['schedule']]
                })

            state = next_state

        return assignments


# 4. Training and Evaluation (Fixed)
env = ClassroomEnv()
agent = DQNAgent(env)

# Train the agent
print("Training...")
agent.train(episodes=200)

# Get final assignments
print("\nGetting optimal assignments...")
final_assignments = agent.get_assignments()

# Display results
if not final_assignments:
    print("\nNo valid assignments could be found. Possible reasons:")
    print("- Insufficient classroom capacity")
    print("- Scheduling conflicts that couldn't be resolved")
    print("- Not enough training episodes")
else:
    print("\n=== ASIGNACIONES ÓPTIMAS ===")
    for i, assignment in enumerate(final_assignments, 1):
        print(f"\nAsignación #{i}:")
        print(f"Curso: {assignment['course']}")
        print(f"Aula: {assignment['room']} (Capacidad: {assignment['capacity']})")
        print(f"Estudiantes: {assignment['students']} (Utilización: {assignment['utilization']:.1%})")
        print(f"Horario: {assignment['schedule']}")
        print(f"Frecuencia: {assignment['frequency']} ({', '.join(assignment['days'])})")
        print("Franjas horarias asignadas:")
        for day in assignment['days']:
            for slot in assignment['slots']:
                print(f"  {day} {slot}")

    # Calculate utilization statistics
    total_slots = len(env.time_slots) * 5  # 5 days per week
    print("\n=== ESTADÍSTICAS DE UTILIZACIÓN ===")
    for room, data in env.classrooms.items():
        # Count existing assignments
        existing = len(data['schedule'])

        # Count new assignments
        new = sum(
            1 for a in final_assignments
            if a['room'] == room
            for day in a['days']
            for slot in a['slots']
        )

        total = existing + new
        print(f"Aula {room}: {total}/{total_slots} franjas ({total / total_slots:.1%})")

    # Verify all courses were assigned
    assigned_courses = {a['course'] for a in final_assignments}
    all_courses = {c['code'] for c in env.courses_to_assign}
    if assigned_courses != all_courses:
        print("\nAdvertencia: No se pudieron asignar los siguientes cursos:")
        for course in all_courses - assigned_courses:
            print(f"- {course}")