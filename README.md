
# Micro-Chess RL in Pure C

A from-scratch Reinforcement Learning engine and custom Automatic Differentiation (Autograd) framework written in pure C, designed to learn how to play Micro-Chess.

This project contains **zero external machine learning dependencies** — no PyTorch, no TensorFlow, no NumPy, no Python. Every single component — from tensor memory allocation and matrix multiplication to reverse-mode backpropagation, the chess move generator, and the RL training loop — is implemented natively in C.

---

## 🌟 Features

### Autograd Engine
- Dynamic computational graph construction with forward and backward function pointers.
- Reverse-mode automatic differentiation via recursive topological sorting.
- Bitwise flag system (`flags & 2`) for O(1) visited-node tracking during graph traversal, avoiding fixed-size stack arrays.
- Proper gradient accumulation for shared nodes (e.g., when a weight is used in multiple operations).

### Neural Network Primitives
- `matmul`: Full matrix multiplication with correct backward pass (`dL/dA = dL/dC @ B^T`, `dL/dB = A^T @ dL/dC`).
- `add`: Element-wise addition with gradient pass-through.
- `relu`: Rectified Linear Unit activation with subgradient gating in the backward pass.
- `sgd_step`: Stochastic Gradient Descent optimizer with automatic gradient zeroing after each update.

### Micro-Chess Environment
- 8x8 board represented as a flat 64-integer array.
- Legal move generation for Pawns (forward push + diagonal capture) and Kings (8-directional movement with edge-wrapping prevention).
- Pawn promotion to King upon reaching the back rank.
- Win detection via King capture (simplified from standard checkmate).
- Board-to-tensor conversion (`1.0` for White, `-1.0` for Black, `0.0` for empty).

### Reinforcement Learning
- Epsilon-greedy exploration policy with linear decay.
- Episodic experience tracking (stores computational graph nodes per game).
- Policy gradient updates using MSE loss derivative at the final game state.
- Full graph cleanup per episode to guarantee zero memory leaks.

---

## 📁 Project Structure

```text
micro-chess-rl/
├── autograd.h      # Header: struct definitions (matrix, Var) and function declarations
├── autograd.c      # Implementation: all tensor ops, forward/backward passes, topo sort, SGD
├── chess.h         # Header: board constants and chess function declarations
├── chess.c         # Implementation: board init, move generation, move execution, win check
├── rl.c            # Main entry point: NN setup, game loop, RL training orchestration
├── README.md       # This file
└── LICENSE         # MIT License
```

---

## 🏗️ Architecture Deep Dive

### Module 1: `autograd` — The Brain

The autograd engine is modeled after PyTorch's dynamic computation graph. Every operation creates a `Var` node that stores:
- `val`: The forward-pass result (a `matrix` of floats).
- `grad`: The accumulated gradient buffer (same shape as `val`).
- `inputs[]`: Pointers to the parent nodes in the graph.
- `forward` / `backward`: Function pointers to the operation-specific math.

When `var_backward(loss)` is called, the engine:
1. Recursively traverses the graph via `build_topo()`, marking visited nodes using bit 1 of the `flags` field.
2. Produces a topological ordering of all nodes.
3. Iterates in **reverse** topological order, calling each node's `backward` function to propagate gradients from the loss down to the leaf weights.
4. Clears the visited bits so the graph can be traversed again in the next episode.

### Module 2: `chess` — The Environment

The chess module is completely decoupled from the neural network. It knows nothing about tensors, gradients, or learning. It simply:
- Maintains a global `board[64]` array.
- Generates legal moves encoded as `(from * 64 + to)` integers.
- Executes moves and handles pawn promotion.
- Checks for game termination (King captured).
- Converts the board state into a float tensor via `board_to_tensor()`.

This separation means you can swap in a full FIDE-compliant chess engine (e.g., via UCI protocol) without touching the neural network code.

### Module 3: `rl` — The Trainer

The RL module is the glue. It:
1. Initializes a 2-layer neural network: `Input(64) → Linear(W1: 64×8) → ReLU → Linear(W2: 8×1) → Score`.
2. Plays games of Micro-Chess using an epsilon-greedy policy.
3. For each candidate move, builds a temporary computation graph, runs the forward pass to evaluate the resulting board, then immediately frees the temporary nodes.
4. At game end, computes the MSE loss gradient on the final score node and calls `var_backward()` to update `W1` and `W2`.
5. Tracks all created nodes in a `game_nodes[]` array and frees them at the end of each episode.

---

## 🚀 Getting Started

### Prerequisites
- A C compiler: GCC, Clang, or MinGW (Windows).
- Standard C library with `math.h` support.

### Compilation

**Linux / macOS:**
```bash
gcc autograd.c chess.c rl.c -o rl_game -lm
./rl_game
```

**Windows (MinGW / MSYS2):**
```bash
gcc autograd.c chess.c rl.c -o rl_game.exe -lm
.\rl_game.exe
```

> **Note:** The `-lm` flag links the math library, required for `abs()` and floating-point operations on Linux/macOS. On some Windows MinGW setups, it may be optional.

---

## 📊 Sample Output

```text
Starting Micro-Chess RL Training (Modular Autograd)...
Game    0 | W:   1 | B:   0 | D:   0 | Eps: 0.30
Game  100 | W:  52 | B:  19 | D:  30 | Eps: 0.25
Game  200 | W:  80 | B:  52 | D:  69 | Eps: 0.20
Game  300 | W: 105 | B:  76 | D: 120 | Eps: 0.15
Game  400 | W: 127 | B: 102 | D: 172 | Eps: 0.10

--- Training Complete! ---
Final -> White: 142 | Black: 126 | Draws: 232
```

**Reading the output:**
- **W / B / D**: Cumulative wins for White, Black, and Draws.
- **Eps**: The current exploration rate. Starts at 0.30 (30% random moves) and decays linearly to 0.05.
- White typically learns to capitalize on the first-move advantage.
- The increasing draw rate indicates the agent is learning to play cautiously and avoid blunders as exploration decreases.

---

## 🧠 How the RL Loop Works (Step by Step)

```text
┌─────────────────────────────────────────────────┐
│              START OF EPISODE                   │
│  init_board() → Reset to Micro-Chess start      │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│              GAME LOOP (per turn)               │
│                                                 │
│  1. get_legal_moves(player) → list of moves     │
│  2. board_to_tensor() → 64-float input          │
│  3. Forward: Score = ReLU(X @ W1) @ W2         │
│  4. Epsilon-greedy:                             │
│     - Random move (explore)                     │
│     - OR simulate all moves, pick best (exploit)│
│  5. make_move() → update board                  │
│  6. check_winner() → game over?                 │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│              END OF EPISODE                     │
│                                                 │
│  1. Compute MSE gradient on final Score node    │
│  2. var_backward(Score) → autograd propagates   │
│     gradients through ReLU → MatMul → W1, W2   │
│  3. sgd_step(W1, W2) → update weights           │
│  4. free_var() → cleanup all episode nodes      │
└─────────────────────────────────────────────────┘
```

---

## 🔧 Tuning Hyperparameters

You can experiment by editing these values in `rl.c`:

| Parameter | Default | Effect of Increasing |
|-----------|---------|----------------------|
| `learning_rate` | `0.05` | Faster learning, but may oscillate or diverge |
| `epsilon` (initial) | `0.30` | More random exploration early on |
| `epsilon` decay | `0.0005` | Slower decay = longer exploration phase |
| `total_games` | `500` | More training = better policy |
| `W1` hidden size | `8` | Larger = more expressive, but slower and may overfit |

---

## 🔮 Future Improvements

This is a foundational proof-of-concept. Potential expansions include:

- **Full Chess Rules**: Add Knights, Bishops, Rooks, Queens, castling, en passant, and check/checkmate detection.
- **Advanced RL**: Upgrade from REINFORCE to Actor-Critic, DQN, or PPO with experience replay.
- **Arena Allocator**: Replace per-node `malloc`/`free` with a bump allocator for 100x faster graph construction.
- **UCI Protocol**: Connect to Stockfish or a chess GUI via the Universal Chess Interface.
- **GPU Acceleration**: Port `matmul_forward`/`matmul_backward` to CUDA or OpenCL.
- **Self-Play with Separate Networks**: Train independent White and Black networks that compete against each other.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 How This Was Built

This project was built the old-fashioned way — **by hand, one line at a time**.

Every struct, every function, every `for` loop, and every pointer was typed out manually on a keyboard. There was no copy-pasting from Stack Overflow, no `pip install torch`, no generated boilerplate. The code was written, compiled, crashed, debugged, and rewritten — often multiple times.

The journey started with a broken `struct` missing a semicolon and a compiler that couldn't find `<stdio.h>`. From there, it grew step by step: first a matrix, then a variable, then addition, then matrix multiplication, then backpropagation through a topological sort, then a chess board, then a move generator, and finally a full reinforcement learning loop that ties it all together.

An AI assistant served as a mentor and sounding board throughout the process — explaining concepts like reverse-mode autodiff, catching subtle bugs like pawns moving backwards, and suggesting architectural patterns like header files and bitwise visited flags. But the AI didn't write the final code. **I did.** I typed every line, made the design decisions, chose the module boundaries, and wrestled with the segfaults.

The result is something I genuinely understand from the ground up. If it breaks, I know exactly where to look. If I want to add a feature, I know exactly which file to edit. That's the value of building things by hand — even when you have help along the way.

**Built with my hands, my brain, and a little guidance from AI.**
