from typing import Tuple, List, Optional

Pos = Tuple[int, int]


class GridWorld:
    def __init__(self, n: int, grid: List[List [int]]):
        self.n = n
        self.grid = [row[:] for row in grid]  # 深拷贝

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.n and 0 <= c < self.n

    def is_dirty(self, p: Pos) -> bool:
        r, c = p
        return self.grid[r][c] == 1

    def clean(self, p: Pos) -> None:
        r, c = p
        self.grid[r][c] = 0

    def all_clean(self) -> bool:
        return all(v == 0 for row in self.grid for v in row)


class Sensors:
    def dirty_here(self, env: GridWorld, pos: Pos) -> bool:
        return env.is_dirty(pos)


class Actuators:
    def move(self, env: GridWorld, pos: Pos, action: str) -> Pos:
        r, c = pos
        moves = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (-1, 0), "DOWN": (1, 0)}
        if action in moves:
            dr, dc = moves[action]
            nxt = (r + dr, c + dc)
            return nxt if env.in_bounds(nxt) else pos
        return pos

    def suck(self, env: GridWorld, pos: Pos) -> None:
        env.clean(pos)


class SnakePathPolicy:
    def __init__(self, n: int):
        self.path: List[Pos] = []
        for r in range(n):
            it = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
            for c in it:
                self.path.append((r, c))
        self.index = 0

    def reset_to(self, start: Pos):
        r0, c0 = start
        self.index = min(
            range(len(self.path)),
            key=lambda i: abs(self.path[i][0] - r0) + abs(self.path[i][1] - c0),
        )

    def next_step(self, curr: Pos, nxt: Pos) -> str:
        (r, c), (nr, nc) = curr, nxt
        if nr == r and nc == c - 1: return "LEFT"
        if nr == r and nc == c + 1: return "RIGHT"
        if nr == r - 1 and nc == c: return "UP"
        if nr == r + 1 and nc == c: return "DOWN"
        if nr != r: return "DOWN" if nr > r else "UP"
        return "RIGHT" if nc > c else "LEFT"

    def decide_move(self, pos: Pos) -> str:
        target = self.path[self.index]
        if target == pos:
            self.index = (self.index + 1) % len(self.path)
            target = self.path[self.index]
        return self.next_step(pos, target)


class SimpleReflexAgent:
    def __init__(self, sensors: Sensors, actuators: Actuators, policy: SnakePathPolicy):
        self.sensors = sensors
        self.actuators = actuators
        self.policy = policy

    def step(self, env: GridWorld, pos: Pos):
        if self.sensors.dirty_here(env, pos):
            self.actuators.suck(env, pos)
            return pos, "SUCK"
        action = self.policy.decide_move(pos)
        return self.actuators.move(env, pos, action), action


def run_episode(
    n: int,
    start: Pos,
    grid: List[List[int]],
    max_steps: Optional[int] = None,
    verbose: bool = True,
):
    env = GridWorld(n, grid)
    sensors, acts = Sensors(), Actuators()
    policy = SnakePathPolicy(n)
    policy.reset_to(start)

    agent = SimpleReflexAgent(sensors, acts, policy)
    pos = start
    steps = 0
    trace: List[str] = []
    if max_steps is None:
        max_steps = n * n * 8

    if verbose:
        print("Initial:", env.grid, "Start:", start)
    while not env.all_clean() and steps < max_steps:
        pos, action = agent.step(env, pos)
        trace.append(action)
        steps += 1
    if verbose:
        print("Final:", env.grid, "End:", pos, "Steps:", steps)
    return env.grid, pos, steps, trace


def make_grid_from_mask(n: int, mask: int) -> List[List[int]]:
    grid: List[List[int]] = []
    for r in range(n):
        row: List[int] = []
        for c in range(n):
            idx = r * n + c
            bit = (mask >> idx) & 1
            row.append(bit)
        grid.append(row)
    return grid


def run_all(n: int, max_steps: Optional[int] = None):
    total = 2 ** (n * n)
    starts = [(r, c) for r in range(n) for c in range(n)]

    for mask in range(total):
        base = make_grid_from_mask(n, mask)
        for start in starts:
            _, _, steps, trace = run_episode(n, start, base, max_steps, verbose=False)
            print(f"mask={mask:0{n*n}b}, start={start}, steps={steps}, trace={trace}")


if __name__ == "__main__":
    run_all(2)
