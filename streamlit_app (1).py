# streamlit_app.py
# Web GUI (Streamlit) to PRESENT your grid auto-parking RL (Q-learning vs Double-Q)
# - No training here (presentation only)
# - Animates greedy policy step-by-step
#
# Run:
#   pip install streamlit matplotlib numpy
#   streamlit run streamlit_app.py
#
# Files expected (export from your notebook):
#   q_tables.pkl              -> dict: level -> qdict (state -> np.array(4))
#   dq_combined_tables.pkl    -> dict: level -> qdict (state -> np.array(4))
#
# If you prefer different filenames/structures, adjust load_models().

import time
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# 1) ENVIRONMENT (grid-based)
#    If you already have ParkingGrid + env_builders in your project,
#    you can delete this section and import them instead.
# ============================================================

State = Tuple[int, int]  # (row, col)

@dataclass
class StepInfo:
    is_collision: bool = False
    is_boundary: bool = False
    is_parked: bool = False


class ParkingGrid:
    """
    Minimal grid parking environment for web demo.
    - static obstacles
    - moving obstacles (simple left-right bounce)
    - slip (action noise)
    - multi-cell parking bay (parking_spots)
    """

    def __init__(
        self,
        size: int = 10,
        start: State = (0, 0),
        parking_spots: Optional[List[State]] = None,
        obstacles: Optional[List[State]] = None,
        moving_obstacles: Optional[List[State]] = None,
        slip_prob: float = 0.0,
        move_penalty: float = -1.0,
        collision_penalty: float = -10.0,
        boundary_penalty: float = -3.0,
        park_reward: float = 20.0,
    ):
        self.size = int(size)
        self.start = tuple(start)
        self.parking_spots = set(parking_spots or [(size - 1, size - 1)])

        self.static_obstacles = set(obstacles or [])
        self.moving_obstacles = list(moving_obstacles or [])  # list for deterministic order
        self._move_dir = -1  # horizontal direction: -1 left, +1 right

        self.slip_prob = float(slip_prob)

        self.move_penalty = float(move_penalty)
        self.collision_penalty = float(collision_penalty)
        self.boundary_penalty = float(boundary_penalty)
        self.park_reward = float(park_reward)

        self.state: State = self.start

    def reset(self) -> State:
        self.state = self.start
        return self.state

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _obstacle_set(self) -> set:
        return self.static_obstacles | set(self.moving_obstacles)

    def _update_moving_obstacles(self) -> None:
        """
        Simple demo movement:
        - all moving obstacles shift horizontally together
        - bounce at borders or static obstacles
        """
        if not self.moving_obstacles:
            return

        proposed = []
        for (r, c) in self.moving_obstacles:
            proposed.append((r, c + self._move_dir))

        # if any proposed position invalid, reverse direction for all
        blocked = False
        for (r, c) in proposed:
            if (not self._in_bounds(r, c)) or ((r, c) in self.static_obstacles):
                blocked = True
                break
        if blocked:
            self._move_dir *= -1
            proposed = [(r, c + self._move_dir) for (r, c) in self.moving_obstacles]

            # second check: if still blocked, stay
            for (r, c) in proposed:
                if (not self._in_bounds(r, c)) or ((r, c) in self.static_obstacles):
                    return

        self.moving_obstacles = proposed

    def step(self, action: int):
        """
        action: 0 up, 1 down, 2 left, 3 right
        returns: next_state, reward, done, info(dict)
        """
        # move dynamic obstacles first
        self._update_moving_obstacles()

        # slip noise: sometimes take random action
        if self.slip_prob > 0 and np.random.rand() < self.slip_prob:
            action = int(np.random.randint(4))

        r, c = self.state
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r + 1, c
        elif action == 2:
            nr, nc = r, c - 1
        elif action == 3:
            nr, nc = r, c + 1
        else:
            raise ValueError("Invalid action (expected 0..3)")

        info = StepInfo()
        done = False

        # boundary
        if not self._in_bounds(nr, nc):
            info.is_boundary = True
            return self.state, self.boundary_penalty, False, info.__dict__

        # collision
        if (nr, nc) in self._obstacle_set():
            info.is_collision = True
            return self.state, self.collision_penalty, False, info.__dict__

        # normal move
        next_state = (nr, nc)
        reward = self.move_penalty

        # parked?
        if next_state in self.parking_spots:
            info.is_parked = True
            reward += self.park_reward
            done = True

        self.state = next_state
        return next_state, reward, done, info.__dict__

    def grid_for_draw(self) -> np.ndarray:
        """
        2D integer map for plotting:
        0 empty
        1 static obstacle
        2 moving obstacle
        3 parking spot
        4 agent
        """
        g = np.zeros((self.size, self.size), dtype=np.int8)
        for (r, c) in self.static_obstacles:
            g[r, c] = 1
        for (r, c) in self.moving_obstacles:
            g[r, c] = 2
        for (r, c) in self.parking_spots:
            g[r, c] = 3
        ar, ac = self.state
        g[ar, ac] = 4
        return g


def env_builder_easy() -> ParkingGrid:
    # Wider parking bay (easier)
    return ParkingGrid(
        size=10,
        start=(0, 0),
        parking_spots=[(9, 7), (9, 8), (9, 9)],
        obstacles=[(3, 3), (3, 4), (4, 4), (6, 2), (7, 2)],
        moving_obstacles=[],
        slip_prob=0.0,
        move_penalty=-1,
        collision_penalty=-10,
        boundary_penalty=-3,
        park_reward=20,
    )


def env_builder_medium() -> ParkingGrid:
    return ParkingGrid(
        size=10,
        start=(0, 0),
        parking_spots=[(9, 8), (9, 9)],
        obstacles=[(2, 2), (2, 3), (2, 4), (5, 5), (6, 5), (7, 5), (7, 6)],
        moving_obstacles=[],
        slip_prob=0.05,
        move_penalty=-1,
        collision_penalty=-10,
        boundary_penalty=-3,
        park_reward=20,
    )


def env_builder_hard() -> ParkingGrid:
    return ParkingGrid(
        size=10,
        start=(0, 0),
        parking_spots=[(9, 9)],
        obstacles=[(1, 2), (2, 2), (3, 2), (4, 2),
                   (6, 7), (6, 8), (6, 9),
                   (8, 5), (8, 6)],
        moving_obstacles=[(5, 9)],
        slip_prob=0.10,
        move_penalty=-1,
        collision_penalty=-10,
        boundary_penalty=-3,
        park_reward=20,
    )


ENV_BUILDERS = {
    "easy": env_builder_easy,
    "medium": env_builder_medium,
    "hard": env_builder_hard,
}


# ============================================================
# 2) MODEL LOADING (Q-tables)
# ============================================================

def load_models() -> Tuple[Dict[str, Dict[State, np.ndarray]], Dict[str, Dict[State, np.ndarray]]]:
    """
    Load pickles exported from your notebook.
    Expected:
      q_tables.pkl -> dict(level -> qdict)
      dq_combined_tables.pkl -> dict(level -> qdict)
    """
    with open("q_tables.pkl", "rb") as f:
        q_tables = pickle.load(f)
    with open("dq_combined_tables.pkl", "rb") as f:
        dq_tables = pickle.load(f)

    return q_tables, dq_tables


def qvals_for_state(qdict: Dict[State, np.ndarray], state: State) -> np.ndarray:
    # fallback for unseen state
    v = qdict.get(state)
    if v is None:
        return np.zeros(4, dtype=float)
    return np.asarray(v, dtype=float)


# ============================================================
# 3) DRAWING
# ============================================================

def draw_grid(env: ParkingGrid, title: str = "", show_legend: bool = True):
    g = env.grid_for_draw()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(g, origin="upper", cmap="tab20")

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # legend (simple)
    if show_legend:
        # Create dummy squares for legend
        handles = [
            plt.Line2D([0], [0], marker='s', color='w', label='Static obstacle', markerfacecolor=plt.cm.tab20(1), markersize=12),
            plt.Line2D([0], [0], marker='s', color='w', label='Moving obstacle', markerfacecolor=plt.cm.tab20(2), markersize=12),
            plt.Line2D([0], [0], marker='s', color='w', label='Parking bay', markerfacecolor=plt.cm.tab20(3), markersize=12),
            plt.Line2D([0], [0], marker='s', color='w', label='Agent', markerfacecolor=plt.cm.tab20(4), markersize=12),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize="small")

    return fig


# ============================================================
# 4) STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Auto Parking RL Demo", layout="wide")

st.title("🚗 Grid Auto-Parking (Q-Learning vs Double-Q) — Streamlit Demo")
st.caption("Presentation UI: plays a greedy episode step-by-step. No training is performed here.")

# sidebar controls
with st.sidebar:
    st.header("Controls")
    level = st.selectbox("Difficulty level", ["easy", "medium", "hard"], index=1)
    algo = st.radio("Algorithm", ["Q-Learning", "Double-Q"], index=0)
    speed = st.slider("Animation speed (seconds per step)", 0.0, 1.0, 0.25, 0.05)
    max_steps = st.slider("Max steps", 50, 600, 300, 50)
    seed = st.number_input("Random seed (for slip/moving)", min_value=0, max_value=999999, value=42, step=1)
    run_btn = st.button("▶ Run animation", type="primary")
    reset_btn = st.button("↺ Reset")

# session state init
if "q_tables" not in st.session_state:
    st.session_state.q_tables = None
    st.session_state.dq_tables = None
    st.session_state.load_error = None

if reset_btn:
    for k in ["env", "step", "total_reward", "done", "algo_used", "level_used"]:
        if k in st.session_state:
            del st.session_state[k]
    st.toast("Reset done.", icon="✅")

# try load models once
if st.session_state.q_tables is None or st.session_state.dq_tables is None:
    try:
        st.session_state.q_tables, st.session_state.dq_tables = load_models()
        st.session_state.load_error = None
    except Exception as e:
        st.session_state.load_error = str(e)

if st.session_state.load_error:
    st.error(
        "Model files not found/failed to load.\n\n"
        "Expected files in the same folder as this app:\n"
        "- q_tables.pkl\n"
        "- dq_combined_tables.pkl\n\n"
        f"Error: {st.session_state.load_error}"
    )
    st.stop()

# choose q-table
qdict = st.session_state.q_tables.get(level, {})
dqdict = st.session_state.dq_tables.get(level, {})
active_qdict = qdict if algo == "Q-Learning" else dqdict

# build environment
if "env" not in st.session_state or st.session_state.get("level_used") != level:
    np.random.seed(int(seed))
    st.session_state.env = ENV_BUILDERS[level]()
    st.session_state.step = 0
    st.session_state.total_reward = 0.0
    st.session_state.done = False
    st.session_state.level_used = level
    st.session_state.algo_used = algo

# layout columns
colA, colB = st.columns([2, 1], gap="large")
placeholder = colA.empty()

with colB:
    st.subheader("Episode info")
    info_box = st.empty()
    st.markdown("**Note:** If your Double-Q cannot reach the target, increase training episodes or add reward shaping in training.")

def render_status(env: ParkingGrid, step: int, total_reward: float, last_reward: Optional[float], last_info: Optional[dict]):
    lines = [
        f"Level: **{level}**",
        f"Algorithm: **{algo}**",
        f"Step: **{step}** / {max_steps}",
        f"Total reward: **{total_reward:.2f}**",
    ]
    if last_reward is not None:
        lines.append(f"Last reward: **{last_reward:.2f}**")
    if last_info is not None:
        lines.append(f"Collision: **{bool(last_info.get('is_collision', False))}**")
        lines.append(f"Boundary: **{bool(last_info.get('is_boundary', False))}**")
        lines.append(f"Parked: **{bool(last_info.get('is_parked', False))}**")
        lines.append(f"Agent pos: **{env.state}**")
        lines.append(f"Moving obstacles: **{env.moving_obstacles}**")
    info_box.markdown("\n\n".join(lines))

# draw initial grid
fig = draw_grid(st.session_state.env, title=f"{level.upper()} — {algo} (ready)")
placeholder.pyplot(fig, clear_figure=True)
render_status(st.session_state.env, st.session_state.step, st.session_state.total_reward, None, None)

# run animation (blocking loop, ok for demos)
if run_btn:
    np.random.seed(int(seed))  # keep repeatable
    env = st.session_state.env
    state = env.state
    total_reward = float(st.session_state.total_reward)

    last_reward = None
    last_info = None

    for _ in range(int(max_steps) - int(st.session_state.step)):
        if st.session_state.done:
            break

        qvals = qvals_for_state(active_qdict, state)
        action = int(np.argmax(qvals))

        next_state, reward, done, info = env.step(action)

        # update counters
        st.session_state.step += 1
        total_reward += float(reward)
        st.session_state.total_reward = total_reward
        st.session_state.done = bool(done)

        last_reward = reward
        last_info = info
        state = next_state

        # draw frame
        fig = draw_grid(env, title=f"{level.upper()} — {algo} (step {st.session_state.step})")
        placeholder.pyplot(fig, clear_figure=True)
        render_status(env, st.session_state.step, total_reward, last_reward, last_info)

        if speed > 0:
            time.sleep(float(speed))

        if done:
            break

    if st.session_state.done:
        st.success("Episode finished: parked ✅")
    else:
        st.warning("Episode ended: max steps reached (not parked).")
