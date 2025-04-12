import tkinter as tk


def on_start():
    print("Training started with hyperparameters:")
    print(f"Learning Rate: {lr_var.get()}")
    print(f"Gamma: {gamma_var.get()}")


def on_stop():
    print("Training stopped.")


# Create the main window
root = tk.Tk()
root.title("Hyperparameter Config")
root.geometry("300x200")

# Learning Rate input
lr_frame = tk.Frame(root)
lr_frame.pack()
lr_label = tk.Label(lr_frame, text="Learning Rate:")
lr_label.pack(side=tk.LEFT)
lr_var = tk.DoubleVar(value=0.001)
lr_entry = tk.Entry(lr_frame, textvariable=lr_var)
lr_entry.pack(side=tk.RIGHT)

# Gamma input
gamma_frame = tk.Frame(root)
gamma_frame.pack()
gamma_label = tk.Label(gamma_frame, text="Gamma:")
gamma_label.pack(side=tk.LEFT)
gamma_var = tk.DoubleVar(value=0.99)
gamma_entry = tk.Entry(gamma_frame, textvariable=gamma_var)
gamma_entry.pack(side=tk.RIGHT)

# Buttons
start_button = tk.Button(root, text="Start Training", command=on_start)
start_button.pack()

stop_button = tk.Button(root, text="Stop Training", command=on_stop)
stop_button.pack()

# Run the UI
root.mainloop()
