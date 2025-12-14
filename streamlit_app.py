import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from joblib import load
import time
import os

# ==========================================
# 1. COPY ENVIRONMENT CLASS & BUILDERS
# ==========================================
# (Ideally import this, but for standalone we define it here)
class ParkingGrid:
    def __init__(self, size=10, start=(0,0), parking_spots=[(9,9)],
                 obstacles=None, moving_humans=None,
                 move_penalty=-2, collision_penalty=-50, park_reward=200,
                 boundary_penalty=-20, reward_shaping=True, shaping_coeff=0.1,
                 slip_prob=0.0):
        self.size = size
        self.start = start
        self.parking_spots = set(parking_spots)
        self.static_obstacles = set(obstacles) if obstacles else set()
        self.moving_humans = moving_humans if moving_humans else []
        self.obstacles = self.static_obstacles | {h["pos"] for h in self.moving_humans}
        self.move_penalty = move_penalty
        self.collision_penalty = collision_penalty
        self.park_reward = park_reward
        self.boundary_penalty = boundary_penalty
        self.reward_shaping = reward_shaping
        self.shaping_coeff = shaping_coeff
        self.slip_prob = slip_prob
        self.reset()

    def reset(self):
        if hasattr(self, "start_candidates"):
            self.start = np.random.choice(len(self.start_candidates)) # simplified choice
            self.start = self.start_candidates[self.start]

        if hasattr(self, "goal_candidates") and hasattr(self, "use_random_goals") and self.use_random_goals:
             # Just pick one for visualization stability
             idx = np.random.randint(len(self.goal_candidates))
             self.parking_spots = {self.goal_candidates[idx]}

        self.state = self.start
        self.steps_taken = 0
        self.prev_action = None
        self.visit_count = {}
        return self.state

    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def _nearest_goal_distance(self, pos):
        if not self.parking_spots: return 0
        return min(abs(pos[0]-g[0]) + abs(pos[1]-g[1]) for g in self.parking_spots)

    def _update_moving_humans(self):
        new_positions = set()
        for h in self.moving_humans:
            x, y = h["pos"]
            if h["axis"] == "h":
                ny = y + h["dir"]
                if ny < h["min"] or ny > h["max"]:
                    h["dir"] *= -1
                    ny = y + h["dir"]
                h["pos"] = (x, ny)
            else:
                nx = x + h["dir"]
                if nx < h["min"] or nx > h["max"]:
                    h["dir"] *= -1
                    nx = x + h["dir"]
                h["pos"] = (nx, y)
            new_positions.add(h["pos"])
        self.obstacles = self.static_obstacles | new_positions
        if hasattr(self, "visual_objects"):
            self.visual_objects["human"] = new_positions

    def step(self, action):
        self._update_moving_humans()
        self.steps_taken += 1
        x, y = self.state
        if action == 0: nx, ny = x-1, y
        elif action == 1: nx, ny = x+1, y
        elif action == 2: nx, ny = x, y-1
        elif action == 3: nx, ny = x, y+1
        else: nx, ny = x, y
        
        info = {"is_collision": False, "is_boundary": False, "is_parked": False}
        done = False

        if not self._in_bounds(nx, ny):
            info["is_boundary"] = True
            return self.state, self.boundary_penalty, done, info
        next_state = (nx, ny)
        if next_state in self.obstacles:
            info["is_collision"] = True
            return self.state, self.collision_penalty, done, info
        if next_state in self.parking_spots:
            self.state = next_state
            info["is_parked"] = True
            return next_state, self.park_reward, True, info

        reward = self.move_penalty
        self.state = next_state
        return next_state, reward, done, info

    def render_map(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        for p in self.obstacles: grid[p] = -1
        for p in self.parking_spots: grid[p] = 2
        grid[self.start] = 3
        return grid

# --- Builders (Must Match Export) ---
def env_builder_easy():
    # Same as export, but we only need visuals here really
    obstacle_map = {
        "parking_slots": {(2,0),(3,0),(4,0),(5,0),(7,0),(2,9),(4,9),(5,9),(6,9),(7,9),(8,9),(2,4),(2,5),(3,3),(4,3),(5,3),(6,3),(3,6),(4,6),(5,6),(6,6),(9,9)},
        "storage": {(0,8), (0,9)},
        "pillar": {(6,0), (3,9), (9,0),(7,3), (7,6),(2,3), (2,6)},
        "bush": {(3,4),(4,4),(5,4),(6,4),(3,5),(4,5),(5,5),(6,5)},
        "guard": {(9,2), (9,3)},
        "parked_car": {(2,0),(3,0),(4,0),(7,0),(2,9),(4,9),(5,9),(6,9),(8,9),(2,5),(3,3),(4,3),(5,3),(6,3),(3,6),(4,6),(5,6),(6,6)},
        "female": {(7,4), (7,5)},
        "waiting": {(4,1)},
        "exiting": {(1,4),(7,8)},
        "empty_soon": {(2,4), (5,0),(7,9)}
    }
    obstacles = set().union(*[v for k, v in obstacle_map.items() if k != "parking_slots"])
    env = ParkingGrid(size=10, start=(0,0), parking_spots=[(9,9)], obstacles=obstacles)
    env.visual_objects = obstacle_map
    return env

def env_builder_medium():
    obstacle_map = {
        "parking_slots": set(),
        "ticket_machine": {(0,17),(0,18),(0,19),(1,17),(1,18),(1,19),(2,17),(2,18),(2,19),(3,17),(3,18),(3,19)},
        "water_leak": {(13,14),(13,15),(13,16),(13,17),(13,18),(14,14),(14,15),(14,16),(14,17),(14,18),(15,17),(15,18)},
        "barrier_cone": {(15,13),(15,14),(15,15),(15,16),(16,17),(16,18),(16,19),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(12,19),(13,13),(14,13),(13,19),(14,19),(15,19)},
        "wall": {(15,2),(15,3),(15,4),(15,5),(15,6),(15,7),(15,8),(15,9),(15,10),(15,11),(15,12),(6,14),(6,17),(7,14),(7,17)},
        "entrance": {(17,18),(18,18),(19,18),(17,19),(18,19),(19,19),(17,17)},
        "bush": {(16,2),(16,3),(16,4),(16,5),(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12),(17,2),(17,3),(17,4),(17,5),(17,6),(17,7),(17,8),(17,9),(17,10),(17,11),(17,12),(7,4),(7,5),(7,6),(7,7),(3,6),(3,7),(3,8),(3,11),(3,12),(3,13),(3,14)},
        "parked_car": {(3,2),(6,2),(7,2),(8,2),(9,2),(10,2),(11,2),(12,2),(13,2),(14,2),(3,3),(4,3),(5,2),(6,3),(7,3),(8,3),(9,3),(11,3),(12,3),(13,3),(14,3),(2,6),(2,7),(2,8),(2,11),(2,13),(2,14),(4,6),(4,7),(4,8),(4,11),(4,12),(4,13),(4,14),(9,12),(9,13),(9,14),(9,17),(9,18),(9,19),(10,12),(10,13),(10,14),(10,18),(10,19),(8,7),(10,7),(13,7),(8,8),(9,8),(10,8),(12,8),(13,8)},
        "female": {(6,12),(6,13),(6,15),(6,16),(6,18),(6,19),(7,12),(7,13),(7,15),(7,16),(7,18),(7,19)},
        "waiting": {(3,1)},
        "exiting": {(10,4),(5,4),(1,12),(5,8)},
        "empty_soon": {(4,2),(10,3),(5,3),(2,12),(4,8)}
    }
    obstacles = set().union(*[v for k, v in obstacle_map.items() if k != "parking_slots"])
    env = ParkingGrid(size=20, start=(17,16), parking_spots=[(9,9)], obstacles=obstacles)
    env.visual_objects = obstacle_map
    env.goal_candidates = [(10,17), (9,7), (12,7)]
    env.use_random_goals = True 
    return env

def env_builder_hard():
    moving_humans = [
        {"pos": (25,29), "axis": "h", "min": 23, "max": 29, "dir": -1},
        {"pos": (26,29), "axis": "h", "min": 23, "max": 29, "dir": -1},
        {"pos": (24,23), "axis": "v", "min": 17, "max": 24, "dir":  1},
        {"pos": (24,24), "axis": "v", "min": 17, "max": 24, "dir": -1},
        {"pos": (16,23), "axis": "v", "min": 9,  "max": 16, "dir": -1},
        {"pos": (16,24), "axis": "v", "min": 9,  "max": 16, "dir":  1},
        {"pos": (26,7),  "axis": "v", "min": 18, "max": 26, "dir": -1},
        {"pos": (9,7),   "axis": "v", "min": 9,  "max": 15, "dir":  1},
        {"pos": (9,8),   "axis": "v", "min": 9,  "max": 15, "dir": -1},
        {"pos": (29,20), "axis": "v", "min": 22, "max": 29, "dir": -1},
        {"pos": (29,21), "axis": "v", "min": 22, "max": 29, "dir":  1},
        {"pos": (29,22), "axis": "v", "min": 22, "max": 29, "dir": -1},
    ]
    obstacle_map = {
        "human": {h["pos"] for h in moving_humans},
        "ticket_machine": {(27,4),(28,4),(29,4),(27,5),(28,5),(29,5),(27,6),(28,6),(29,6)},
        "guard": {(27,0),(28,0),(29,0),(27,1),(28,1),(29,1),(27,2),(28,2),(29,2)},
        "wall": {(0,0),(1,0),(16,7),(17,7),(16,14),(17,14),(12,10),(13,10),(12,17),(13,17),(20,10),(21,10),(20,17),(21,17),(10,27),(10,28),(14,27),(14,28),(18,27),(18,28),(29,8),(29,9),(29,10),(29,11),(29,12),(29,13),(29,14),(29,15),(29,16),(29,17),(29,18),(13,2),(14,2),(15,2),(16,2),(17,2),(18,2),(19,2),(20,2),(21,2),(22,2)},
    }
    obstacles = set().union(*[v for k, v in obstacle_map.items() if k not in ("human", "parking_slots")])
    env = ParkingGrid(size=30, start=(1,4), parking_spots=[(19,28)], obstacles=obstacles, moving_humans=moving_humans)
    env.start_candidates = [(12,3), (1,4), (1,19)]
    # FIXED GOAL FOR HARD MODE
    env.use_random_goals = False 
    env.visual_objects = obstacle_map
    return env

ENV_BUILDERS = {
    "easy": env_builder_easy,
    "medium": env_builder_medium,
    "hard": env_builder_hard
}

# ==========================================
# 2. VISUALIZATION FUNCTION (Adapted from Notebook)
# ==========================================
def draw_grid(env, title="Parking Grid"):
    fig, ax = plt.subplots(figsize=(6, 6))
    grid = env.render_map()
    
    # Base grid
    ax.imshow(grid, cmap='tab20', origin='upper')

    # Draw Icons/Colors based on testing3.ipynb Cell 31 logic
    if hasattr(env, "visual_objects"):
        icon_fs = 12 if env.size <= 10 else 8
        for k, cells in env.visual_objects.items():
            for (x, y) in cells:
                # RECTANGLE: (y, x) because matplotlib is (x, y) but array is (row, col)
                rect_args = ((y-0.5, x-0.5), 1, 1)
                
                if k == "storage":
                    ax.add_patch(Rectangle(*rect_args, facecolor="khaki", alpha=0.7))
                    ax.text(y, x, "▧", ha='center', va='center', color="dimgray", fontsize=icon_fs)
                elif k == "pillar":
                    ax.add_patch(Rectangle(*rect_args, facecolor="khaki", alpha=0.7))
                    ax.text(y, x, "●", ha='center', va='center', color="black", fontsize=icon_fs)
                elif k == "bush":
                    ax.add_patch(Rectangle(*rect_args, facecolor="#b6e3a8", alpha=0.8))
                    ax.text(y, x, "♣", ha='center', va='center', color="darkgreen", fontsize=icon_fs)
                elif k == "guard":
                    ax.add_patch(Rectangle(*rect_args, facecolor="khaki", alpha=0.7))
                    ax.text(y, x, "⌂", ha='center', va='center', color="red", fontsize=icon_fs, fontweight='bold')
                elif k == "parked_car":
                    ax.add_patch(Rectangle(*rect_args, facecolor="#cce5ff", alpha=0.8))
                elif k == "female":
                    ax.add_patch(Rectangle(*rect_args, facecolor="#f7c6d0", alpha=0.8))
                    ax.text(y, x, "♀", ha='center', va='center', color="magenta", fontsize=icon_fs, fontweight='bold')
                elif k == "ticket_machine":
                    ax.add_patch(Rectangle(*rect_args, facecolor="khaki", alpha=0.75))
                    ax.text(y, x, "▣", ha='center', va='center', color="saddlebrown", fontsize=icon_fs)
                elif k == "water_leak":
                    ax.add_patch(Rectangle(*rect_args, facecolor="#cceeff", alpha=0.8))
                    ax.text(y, x, "≈", ha='center', va='center', color="steelblue", fontsize=14, fontweight='bold')
                elif k == "barrier_cone":
                    ax.add_patch(Rectangle(*rect_args, facecolor="#f8caca", alpha=0.8))
                    ax.text(y, x, "▲", ha='center', va='center', color="darkred", fontsize=icon_fs)
                elif k == "wall":
                    ax.add_patch(Rectangle(*rect_args, facecolor="#6e6e6e", alpha=0.85))
                    ax.text(y, x, "✖", ha='center', va='center', color="white", fontsize=icon_fs)
                elif k == "human":
                     ax.add_patch(Rectangle((y-0.4, x-0.4), 0.8, 0.8, facecolor="red", edgecolor="black", alpha=0.9))

    # Agent
    ax.scatter([env.state[1]], [env.state[0]], marker='s', s=150, color='blue', edgecolor='white', label='Agent')
    
    # Goal
    for g in env.parking_spots:
        ax.scatter([g[1]], [g[0]], marker='*', s=200, color='gold', edgecolor='black', label='Goal')

    ax.invert_yaxis() # Match notebook orientation
    ax.set_title(title)
    ax.axis('off')
    return fig

# ==========================================
# 3. STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Auto Parking RL", layout="wide")
st.title("🚗 Auto Parking RL (Double-Q)")
st.caption("Visualizing the Double-Q Agent on Easy, Medium, and Hard environments.")

# --- SIDEBAR ---
with st.sidebar:
    level = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    speed = st.slider("Animation Speed (seconds)", 0.0, 0.5, 0.05, 0.01)
    max_steps = st.slider("Max Steps", 50, 1000, 300, 50)
    run_btn = st.button("▶ Run Animation", type="primary")
    reset_btn = st.button("↺ Reset")

# --- LOAD MODELS ---
if "dq_tables" not in st.session_state:
    try:
        st.session_state.dq_tables = load("artifacts/dq_combined_tables.joblib")
        st.success("Models loaded!")
    except Exception as e:
        st.error(f"Error loading models. Run export_models.py first! {e}")
        st.stop()

# --- INIT ENV ---
# Reset logic if button clicked OR level changed
if reset_btn or "env" not in st.session_state or st.session_state.get("stored_level") != level:
    st.session_state.env = ENV_BUILDERS[level]()
    st.session_state.stored_level = level
    st.session_state.step = 0
    st.session_state.done = False

env = st.session_state.env
q_table = st.session_state.dq_tables.get(level, {})

col1, col2 = st.columns([3, 1])
placeholder = col1.empty()

# --- ANIMATION LOOP ---
if run_btn:
    st.session_state.done = False
    env.reset()
    
    for _ in range(max_steps):
        # Greedy Action
        if env.state in q_table:
            action = int(np.argmax(q_table[env.state]))
        else:
            action = np.random.randint(4) # Fallback

        # Step
        _, _, done, info = env.step(action)
        st.session_state.step += 1
        
        # Draw
        fig = draw_grid(env, title=f"Step {st.session_state.step} | Collision: {info['is_collision']}")
        placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig) # Memory cleanup
        
        # Info Panel
        with col2:
            st.metric("Steps", st.session_state.step)
            st.write(f"Position: {env.state}")
            st.write(f"Parked: {info['is_parked']}")

        time.sleep(speed)
        
        if done:
            if info['is_parked']:
                st.balloons()
                st.success("Parked Successfully! 🎉")
            else:
                st.error("Crashed! 💥")
            break
else:
    # Static Initial Frame
    fig = draw_grid(env, title="Ready to Start")
    placeholder.pyplot(fig, clear_figure=True)
