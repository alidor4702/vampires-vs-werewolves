# gui/board.py
from ast import main
import tkinter as tk
from tkinter import simpledialog, messagebox
from core.config import GameConfig
from core.state import GameState
import random
from core.random_agent import RandomAgent
from core.mcts_agent_new import MCTSAgent
from core.heuristic_agent import HeuristicAgent
from core.simple_agent import SimpleHeuristicAgent






class GameBoard(tk.Frame):
    COLORS = {"H": "#c49a6c", "V": "#e74c3c", "W": "#3498db", "E": "#ecf0f1"}

    def __init__(self, master, config: GameConfig):
        super().__init__(master)
        self.config = config
        self.state = GameState(config.grid_rows, config.grid_cols, config.human_density)
        self.cell_size = config.cell_size
        self.zoom_factor = 1.0
        self.pack(fill="both", expand=True)
        self.master.title("Vampires vs Werewolves – Board")
        # 🔹 Keyboard controls
        self.master.bind("<space>", lambda e: self.next_turn())   # Next turn
        self.master.bind("<Command-minus>", lambda e: self.zoom_step(-1))  # ⌘ -
        self.master.bind("<Command-=>", lambda e: self.zoom_step(+1))      # ⌘ +
        self.master.bind("<Left>", lambda e: self.move_view(-50, 0))
        self.master.bind("<Right>", lambda e: self.move_view(50, 0))
        self.master.bind("<Up>", lambda e: self.move_view(0, -50))
        self.master.bind("<Down>", lambda e: self.move_view(0, 50))


        self.selected = None
        self.drag_start = None
        self.last_log_len = 0

        self.create_widgets()
        self.draw_grid()
        self.after(300, self.refresh_log)

        # 🔹 AI agent initialization
        if self.config.mode == "AI":
            self.ai_agent_w = MCTSAgent()  # Werewolves
            self.ai_agent_v = None  # Vampires (human player)
        elif self.config.mode == "AI_vs_AI":
            # AI vs AI mode
            self.ai_agent_v = MCTSAgent(time_limit=1.9)  # Vampires
            self.ai_agent_w = HeuristicAgent()  # Werewolves
        else:
            self.ai_agent_v = None
            self.ai_agent_w = None
        
        # 🔹 Auto-start if AI vs AI
        if self.config.mode == "AI_vs_AI":
            self.after(500, self.auto_play_turn)


    # --------------------------------------------------------------
    def create_widgets(self):
        top = tk.Frame(self)
        top.pack(anchor="w", fill="x")

        tk.Button(top, text="Back", command=self.back_to_menu).pack(side="left", padx=5)
        tk.Button(top, text="Restart", command=self.quick_restart).pack(side="left", padx=5)
        tk.Button(top, text="Next Turn", command=self.next_turn).pack(side="left", padx=5)
        self.status = tk.Label(top, text=f"Turn: {self.state.turn}", width=20)
        self.status.pack(side="left", padx=5)

        # Create main horizontal split
        main = tk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        # ---------------- LEFT: LOG PANEL ----------------
        log_frame = tk.Frame(main, bg="#857f70")
        tk.Label(log_frame, text="Game Log", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.log_widget = tk.Text(log_frame, width=40, height=30, state="disabled", bg="#857f70")
        self.log_widget.pack(fill="both", expand=True)
        main.add(log_frame, minsize=200)

        # ---------------- MIDDLE: BOARD CANVAS ----------------
        canvas_frame = tk.Frame(main, bg="white")
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.configure(scrollregion=(0, 0,
            self.config.grid_cols * self.cell_size,
            self.config.grid_rows * self.cell_size))
        
        # Set minimum width for board
        board_min_width = self.config.grid_cols * self.cell_size
        main.add(canvas_frame, minsize=board_min_width, stretch="always")

        # ---------------- RIGHT: HEATMAP PANEL ----------------
        heat_frame = tk.Frame(main, bg="#222")
        tk.Label(
            heat_frame,
            text="AI Heat Map",
            font=("Helvetica", 12, "bold"),
            fg="white",
            bg="#222"
        ).pack(anchor="w")

        self.heat_canvas = tk.Canvas(heat_frame, bg="black", width=250, height=250)
        self.heat_canvas.pack(fill="both", expand=True)
        self.heat_canvas.bind("<Button-1>", self.on_heat_click)

        self.heat_map_mode = "heat"
        self.last_heat = None
        self.last_path = None

        main.add(heat_frame, minsize=250)  # <-- add heatmap last (right side)

        # ---------------- BINDINGS ----------------
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)

        self.cells = {}

    # --------------------------------------------------------------
    def refresh_log(self):
        """Continuously refresh the game log every 300ms."""
        if self.last_log_len < len(self.state.log):
            new_lines = self.state.log[self.last_log_len:]
            self.log_widget.config(state="normal")
            for line in new_lines:
                self.log_widget.insert("end", line + "\n")
            self.log_widget.see("end")
            self.log_widget.config(state="disabled")
            self.last_log_len = len(self.state.log)
        self.after(300, self.refresh_log)

    # --------------------------------------------------------------
    def draw_grid(self):
        self.canvas.delete("all")
        size = int(self.cell_size * self.zoom_factor)
        for r in range(self.state.rows):
            for c in range(self.state.cols):
                x1, y1 = c * size, r * size
                x2, y2 = x1 + size, y1 + size
                h, v, w = self.state.get_counts(r, c)
                if v > 0:
                    color = self.COLORS["V"]
                elif w > 0:
                    color = self.COLORS["W"]
                elif h > 0:
                    color = self.COLORS["H"]
                else:
                    color = self.COLORS["E"]
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                txt = f"H{h}" if h > 0 else f"V{v}" if v > 0 else f"W{w}" if w > 0 else ""
                self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                        text=txt, font=("Helvetica", 10, "bold"))
                self.cells[(r, c)] = rect
        self.canvas.config(scrollregion=(0, 0,
                                         self.state.cols * size,
                                         self.state.rows * size))

    # --------------------------------------------------------------
    def check_victory(self):
        """Check if one side has won or if it's a draw."""
        result = self.state.check_end_condition()
        if result:
            messagebox.showinfo("Game Over", result)
            self.state.add_log(f"*** {result} ***")
            return True
        return False

    # --------------------------------------------------------------
    def quick_restart(self):
        """Restart the game instantly with same or randomized parameters."""
        import random

        # Copy config to mutate if random flags are on
        rows = self.config.grid_rows
        cols = self.config.grid_cols
        hum_density = self.config.human_density

        random_rows_used = random_cols_used = random_hum_used = False

        # If this config was started with random enabled → re-randomize
        if getattr(self.config, "random_rows", False):
            rows = random.randint(5, 256)
            random_rows_used = True
        if getattr(self.config, "random_cols", False):
            cols = random.randint(5, 256)
            random_cols_used = True
        if getattr(self.config, "random_hum", False):
            hum_density = round(random.uniform(0.1, 0.6), 2)
            random_hum_used = True

        # Apply new values
        self.config.grid_rows = rows
        self.config.grid_cols = cols
        self.config.human_density = hum_density

        # Create new game state
        self.state = GameState(rows, cols, hum_density,
                               self.config.start_vampires,
                               self.config.start_werewolves)

        # Reset visuals/log
        self.last_log_len = 0
        self.log_widget.config(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.config(state="disabled")
        self.status.config(text=f"Turn: {self.state.turn}")
        self.draw_grid()

        # Log restart info
        self.state.add_log("=== New game started ===")

        if random_rows_used or random_cols_used or random_hum_used:
            self.state.add_log("=== Random settings applied on restart ===")
            if random_rows_used:
                self.state.add_log(f"Random Rows chosen: {rows}")
            if random_cols_used:
                self.state.add_log(f"Random Cols chosen: {cols}")
            if random_hum_used:
                self.state.add_log(f"Random Human Density chosen: {hum_density:.2f}")
            self.state.add_log("==========================================")


    # --------------------------------------------------------------
    def on_click(self, event):
        """Handle selection and move actions."""
        if self.drag_start:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        c = int(x // (self.cell_size * self.zoom_factor))
        r = int(y // (self.cell_size * self.zoom_factor))
        if not self.state.in_bounds(r, c):
            return

        if self.selected is None:
            h, v, w = self.state.get_counts(r, c)
            if (self.state.turn == "V" and v > 0) or (self.state.turn == "W" and w > 0):
                if (r, c) in self.state.targets_used_this_turn:
                    messagebox.showinfo("Invalid",
                        "This cell received units this turn and cannot act as a source.")
                    return
                self.selected = (r, c)
                self.canvas.itemconfig(self.cells[(r, c)], outline="yellow", width=3)
            else:
                messagebox.showinfo("Invalid", "No units of your species here.")
        else:
            r1, c1 = self.selected
            if not self.state.is_adjacent(r1, c1, r, c):
                messagebox.showinfo("Invalid", "You can only move to adjacent cells (8 directions).")
                self.reset_selection()
                return
            num = simpledialog.askinteger("Move", "How many creatures to move?", minvalue=1)
            if num is None:
                self.reset_selection()
                return
            moved = self.state.move_group(r1, c1, num, r, c)
            if not moved:
                messagebox.showinfo("Invalid", "Illegal move (check adjacency or allowance).")
            self.reset_selection()
            self.draw_grid()
            self.check_victory()

    # --------------------------------------------------------------
    def draw_heatmap(self, heat_data, title="Heat Map"):
        """Draw a properly scaled heat map (2-D numpy array) in the right panel."""
        if heat_data is None:
            return
        self.heat_canvas.delete("all")

        # --- get board & heatmap sizes ---
        R, C = heat_data.shape
        board_R, board_C = self.state.rows, self.state.cols

        # --- dynamically fit heatmap into canvas ---
        canvas_w = max(50, self.heat_canvas.winfo_width() or 250)
        canvas_h = max(50, self.heat_canvas.winfo_height() or 250)

        # maintain aspect ratio consistent with board grid
        cell_ratio = board_C / board_R
        target_ratio = canvas_w / canvas_h
        if cell_ratio > target_ratio:
            # board wider than tall
            w = canvas_w
            h = int(w / cell_ratio)
        else:
            # board taller than wide
            h = canvas_h
            w = int(h * cell_ratio)

        # Center the heatmap inside the canvas safely
        x_offset = (canvas_w - w) / 2
        y_offset = (canvas_h - h) / 2

        cw = w / C
        ch = h / R

        vmax, vmin = float(heat_data.max()), float(heat_data.min())
        rng = max(1e-9, vmax - vmin)

        def color(val):
            # blue (cold) → red (hot)
            t = (val - vmin) / rng
            r = int(255 * t)
            b = int(255 * (1 - t))
            return f"#{r:02x}00{b:02x}"

        # --- draw safely inside canvas bounds ---
        for r in range(R):
            for c in range(C):
                x1 = x_offset + c * cw
                y1 = y_offset + r * ch
                x2 = x1 + cw
                y2 = y1 + ch
                self.heat_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color(heat_data[r, c]),
                    outline=""
                )

        # title
        self.heat_canvas.create_text(
            x_offset + 8, y_offset + 8,
            anchor="nw",
            text=title,
            fill="white",
            font=("Helvetica", 10, "bold")
        )


    # --------------------------------------------------------------
    def on_heat_click(self, event):
        """When user clicks heatmap, toggle path view or back."""
        if self.last_heat is None:
            return
        w = self.heat_canvas.winfo_width()
        h = self.heat_canvas.winfo_height()
        R, C = self.last_heat.shape
        cw, ch = w / C, h / R
        c = int(event.x // cw)
        r = int(event.y // ch)
        if not (0 <= r < R and 0 <= c < C):
            return

        if self.heat_map_mode == "heat":
            # show path heat for that cell
            if hasattr(self.ai_agent, "_best_path_heat"):
                self.last_path = self.ai_agent._best_path_heat(r, c)
                self.draw_heatmap(self.last_path, title=f"Path from ({r},{c})")
                self.heat_map_mode = "path"
        else:
            # back to main heat map
            self.draw_heatmap(self.last_heat, title="Heat Map")
            self.heat_map_mode = "heat"


    # --------------------------------------------------------------
    def on_pan_start(self, event):
        self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.config(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        if self.drag_start:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_pan_end(self, event):
        self.drag_start = None
        self.canvas.config(cursor="")

    # --------------------------------------------------------------
    def on_zoom(self, event):
        direction = 1 if event.delta > 0 or event.num == 4 else -1
        new_zoom = self.zoom_factor * (1.1 if direction > 0 else 0.9)
        self.zoom_factor = max(0.4, min(3.0, new_zoom))
        self.draw_grid()

    # --------------------------------------------------------------
    def reset_selection(self):
        if self.selected:
            self.canvas.itemconfig(self.cells[self.selected], outline="gray", width=1)
        self.selected = None

    # --------------------------------------------------------------
    def log_board_state(self):
        """Log a concise snapshot of the current board state for debugging AI."""
        self.state.add_log("=== Board Snapshot ===")

        H = 0; V = 0; W = 0
        ours = []
        enemies = []

        for r in range(self.state.rows):
            for c in range(self.state.cols):
                cell = self.state.grid[r,c]
                H += cell.humans
                V += cell.vampires
                W += cell.werewolves

                if cell.vampires > 0:
                    ours.append((r,c,cell.vampires))
                if cell.werewolves > 0:
                    enemies.append((r,c,cell.werewolves))

        self.state.add_log(f"Humans total: {H} | Vampires total: {V} | Werewolves total: {W}")

        # Sort by stack size (largest visible threats & power centers)
        ours_sorted = sorted(ours, key=lambda x:x[2], reverse=True)
        enemies_sorted = sorted(enemies, key=lambda x:x[2], reverse=True)

        if ours_sorted:
            self.state.add_log("Largest Our Stacks:")
            for (r,c,s) in ours_sorted[:5]:
                self.state.add_log(f"  ({r},{c}) size={s}")

        if enemies_sorted:
            self.state.add_log("Largest Enemy Stacks:")
            for (r,c,s) in enemies_sorted[:5]:
                self.state.add_log(f"  ({r},{c}) size={s}")

        self.state.add_log("===")
    # --------------------------------------------------------------

    def next_turn(self):
        """Advance turn; if AI mode, call the agent."""
        self.state.next_turn()
        self.log_board_state()
        
        self.status.config(text=f"Turn: {self.state.turn}")

        # Check for AI turns
        if self.config.mode == "AI" and self.state.turn == "W" and self.ai_agent_w:
            self.run_agent_turn(self.ai_agent_w, "Werewolves")
        elif self.config.mode == "AI_vs_AI":
            # Auto-play next turn in AI vs AI mode
            self.after(500, self.auto_play_turn)
        
        self.check_victory()

    # --------------------------------------------------------------
    def auto_play_turn(self):
        """Automatically play AI turns in AI vs AI mode."""
        if self.check_victory():
            return  # Game ended
        
        # Execute current AI's turn
        if self.state.turn == "V" and self.ai_agent_v:
            self.run_agent_turn(self.ai_agent_v, "Vampires")
        elif self.state.turn == "W" and self.ai_agent_w:
            self.run_agent_turn(self.ai_agent_w, "Werewolves")
        
        # Next turn
        self.state.next_turn()
        self.status.config(text=f"Turn: {self.state.turn}")
        self.draw_grid()
        
        # Continue if game not over
        if not self.check_victory():
            self.after(800, self.auto_play_turn)  # 800ms delay between turns

    # --------------------------------------------------------------
    def back_to_menu(self):
        from gui.menu import MainMenu
        self.destroy()
        MainMenu(self.master)
    # --------------------------------------------------------------
    def zoom_step(self, direction):
        """Keyboard zoom: +1 for zoom in, -1 for zoom out."""
        new_zoom = self.zoom_factor * (1.1 if direction > 0 else 0.9)
        self.zoom_factor = max(0.4, min(3.0, new_zoom))
        self.draw_grid()

    # --------------------------------------------------------------
    def move_view(self, dx, dy):
        """Move the canvas viewport with arrow keys."""
        self.canvas.xview_scroll(int(dx / 10), "units")
        self.canvas.yview_scroll(int(dy / 10), "units")
    # --------------------------------------------------------------
    def ai_random_move(self):
        """Perform one random valid AI move."""
        import time
        moves = []

        # Gather all legal moves for AI (Werewolves)
        for r in range(self.state.rows):
            for c in range(self.state.cols):
                cell = self.state.grid[r, c]
                if cell.werewolves > 0:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            r2, c2 = r + dr, c + dc
                            if not self.state.in_bounds(r2, c2):
                                continue
                            if self.state.is_adjacent(r, c, r2, c2):
                                moves.append((r, c, r2, c2))

        if not moves:
            self.state.add_log("AI skipped turn (no legal moves).")
            return

        # Pick one random move and a random number of creatures to move
        r1, c1, r2, c2 = random.choice(moves)
        num_to_move = random.randint(1, max(1, self.state.grid[r1, c1].werewolves))
        self.state.add_log(f"AI moving {num_to_move} Werewolves from ({r1},{c1}) → ({r2},{c2}).")
        self.state.move_group(r1, c1, num_to_move, r2, c2)
        self.draw_grid()
        self.state.add_log("AI turn complete.")
    # --------------------------------------------------------------
    def run_agent_turn(self, agent, agent_name):
        """Execute the current agent’s moves (possibly multiple)."""
        import time

        actions = agent.select_action(self.state)
        # 🔹 visualize heat map if agent provides one
        if hasattr(agent, "get_heatmap"):
            self.last_heat = agent.get_heatmap()
            if self.last_heat is not None:
                self.draw_heatmap(self.last_heat, title=f"{agent_name} Heat Map")

        if not actions:
            self.state.add_log(f"{agent_name} AI chose to skip turn.")
            return

        self.state.add_log(f"{agent_name} AI decided {len(actions)} move(s) this turn.")

        if hasattr(agent, "debug_messages"):
            for line in agent.debug_messages():
                self.state.add_log(f"[{agent_name}-DBG] {line}")

        # Display reasoning log if available
        if hasattr(agent, "log") and isinstance(agent.log, list):
            # MCTS agent: log is a list
            for line in agent.log:
                self.state.add_log(line)
        elif hasattr(agent, "_last_debug") and agent._last_debug:
            # HeuristicAgent: _last_debug is a list
            for line in agent._last_debug:
                self.state.add_log(line)


        for (r1, c1, r2, c2, num) in actions:
            if not self.state.in_bounds(r1, c1) or not self.state.in_bounds(r2, c2):
                continue
            if not self.state.is_adjacent(r1, c1, r2, c2):
                continue
            moved = self.state.move_group(r1, c1, num, r2, c2)
            if moved:
                species = "Vampires" if self.state.turn == "V" else "Werewolves"
                self.state.add_log(f"{agent_name} ({species}) moved {num} from ({r1},{c1}) → ({r2},{c2}).")
                self.draw_grid()
                self.update()
                time.sleep(0.25)
            else:
                self.state.add_log(f"{agent_name} attempted invalid move from ({r1},{c1}) → ({r2},{c2}).")

        self.state.add_log(f"{agent_name} turn complete.")
        self.state.add_log("AI turn complete.")
    