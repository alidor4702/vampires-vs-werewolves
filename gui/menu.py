# gui/menu.py
import tkinter as tk
from tkinter import ttk
import random
from core.config import GameConfig
from gui.board import GameBoard


class MainMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master.title("Vampires vs Werewolves – Menu")
        self.pack(padx=20, pady=20)
        self.create_widgets()

    # --------------------------------------------------------------
    def create_widgets(self):
        tk.Label(self, text="Vampires vs Werewolves", font=("Helvetica", 18, "bold")).pack(pady=10)

        # --- Rows ---
        frame_rows = tk.Frame(self)
        frame_rows.pack(pady=5)
        tk.Label(frame_rows, text="Rows (5–256):").pack(side="left", padx=5)
        self.rows_var = tk.IntVar(value=10)
        self.rows_entry = ttk.Entry(frame_rows, textvariable=self.rows_var, width=5)
        self.rows_entry.pack(side="left")
        self.random_rows = tk.BooleanVar(value=False)
        self.rows_check = ttk.Checkbutton(
            frame_rows, text="Random", variable=self.random_rows,
            command=lambda: self.toggle_entry(self.rows_entry, self.random_rows)
        )
        self.rows_check.pack(side="left", padx=10)

        # --- Columns ---
        frame_cols = tk.Frame(self)
        frame_cols.pack(pady=5)
        tk.Label(frame_cols, text="Cols (5–256):").pack(side="left", padx=5)
        self.cols_var = tk.IntVar(value=10)
        self.cols_entry = ttk.Entry(frame_cols, textvariable=self.cols_var, width=5)
        self.cols_entry.pack(side="left")
        self.random_cols = tk.BooleanVar(value=False)
        self.cols_check = ttk.Checkbutton(
            frame_cols, text="Random", variable=self.random_cols,
            command=lambda: self.toggle_entry(self.cols_entry, self.random_cols)
        )
        self.cols_check.pack(side="left", padx=10)

        # --- Human density ---
        frame_hum = tk.Frame(self)
        frame_hum.pack(pady=5)
        tk.Label(frame_hum, text="Human Density (0.1–0.6):").pack(side="left", padx=5)
        self.humans_var = tk.DoubleVar(value=0.3)
        self.hum_entry = ttk.Entry(frame_hum, textvariable=self.humans_var, width=5)
        self.hum_entry.pack(side="left")
        self.random_hum = tk.BooleanVar(value=False)
        self.hum_check = ttk.Checkbutton(
            frame_hum, text="Random", variable=self.random_hum,
            command=lambda: self.toggle_entry(self.hum_entry, self.random_hum)
        )
        self.hum_check.pack(side="left", padx=10)

        # --- Play Mode ---
        frame_mode = tk.Frame(self)
        frame_mode.pack(pady=10)
        tk.Label(frame_mode, text="Play Against:").pack(side="left", padx=5)
        self.mode_var = tk.StringVar(value="human")
        ttk.OptionMenu(frame_mode, self.mode_var, "human", "human", "ai").pack(side="left")

        # --- Buttons ---
        ttk.Button(self, text="Start Game", command=self.start_game).pack(pady=15)
        ttk.Button(self, text="Quit", command=self.master.quit).pack()

    # --------------------------------------------------------------
    def toggle_entry(self, entry, var):
        """Enable or disable entry box depending on random checkbox."""
        if var.get():
            entry.configure(state="disabled")
            entry.configure(style="Disabled.TEntry")
        else:
            entry.configure(state="normal")

    # --------------------------------------------------------------
    def start_game(self):
        # --- Randomize if needed ---
        random_rows_used = random_cols_used = random_hum_used = False
        if self.random_rows.get():
            self.rows_var.set(random.randint(5, 256))
            random_rows_used = True
        if self.random_cols.get():
            self.cols_var.set(random.randint(5, 256))
            random_cols_used = True
        if self.random_hum.get():
            self.humans_var.set(round(random.uniform(0.1, 0.6), 2))
            random_hum_used = True

        # --- Clamp values to allowed bounds ---
        rows = max(5, min(256, self.rows_var.get()))
        cols = max(5, min(256, self.cols_var.get()))
        hum_density = max(0.1, min(0.6, self.humans_var.get()))

        # --- Create config ---
        cfg = GameConfig(
            grid_rows=rows,
            grid_cols=cols,
            human_density=hum_density,
            random_rows=self.random_rows.get(),
            random_cols=self.random_cols.get(),
            random_hum=self.random_hum.get(),
            mode=self.mode_var.get()
        )

        # --- Start the game ---
        self.destroy()
        board = GameBoard(self.master, cfg)

        # --- Add random info to the log if applicable ---
        if random_rows_used or random_cols_used or random_hum_used:
            board.state.add_log("=== Random settings applied ===")
            if random_rows_used:
                board.state.add_log(f"Random Rows chosen: {rows}")
            if random_cols_used:
                board.state.add_log(f"Random Cols chosen: {cols}")
            if random_hum_used:
                board.state.add_log(f"Random Human Density chosen: {hum_density:.2f}")
            board.state.add_log("===============================")
