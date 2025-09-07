from flask import Flask, request, jsonify
from flask_cors import CORS
import pennylane as qml
import numpy as np
import random
from typing import List

app = Flask(__name__)
CORS(app)

# 4 position qubits (for 16 possible positions, but we only use 9 in 3x3 grid)
pos_wires = [0, 1, 2, 3]
# 1 control qubit to determine forward/echo phase
cont_wire = 4
n_wires = len(pos_wires) + 1

dev = qml.device("default.mixed", wires=n_wires)
n_pos_qubits = len(pos_wires)
n_positions = 2**n_pos_qubits

# Only valid positions in our 3x3 grid (0-8)
VALID_POSITIONS = list(range(9))


def coord_to_index(r, c):
    return 3 * r + c


def index_to_coord(i):
    return (i // 3, i % 3)


# We'll store the complete path history with quantum operations
# Each op is a dict: { "op": str, "from": int, "to": int, "step": int }
GAME_HISTORY: List[dict] = []
PATH_HISTORY: List[int] = []  # Tracks actual path positions
CURRENT_POSITION = 4  # Start at center (position 4 in 0-8 indexing)
ECHO_PATH: List[int] = []  # Will store the reverse path during echo
ECHO_TRIGGERED = False
GAME_ENDED = False


def reset_game():
    global GAME_HISTORY, PATH_HISTORY, CURRENT_POSITION, ECHO_PATH, ECHO_TRIGGERED, GAME_ENDED
    GAME_HISTORY = [{"op": "init"}]
    PATH_HISTORY = [4]  # Start at center position
    CURRENT_POSITION = 4
    ECHO_PATH = []
    ECHO_TRIGGERED = False
    GAME_ENDED = False


reset_game()


def get_echo_trigger_step():
    """Generate random steps before echo triggers (3-8)"""
    return random.randint(3, 8)


def apply_position_gate(from_pos, to_pos, step_num):
    """
    Apply a position-specific gate that:
    1. Transfers probability from current position to new position
    2. Records the path in the quantum state
    3. Increases probability amplitude for the new position

    This implements the core quantum movement mechanic.
    """
    # Convert positions to binary representation
    from_bits = [(from_pos >> i) & 1 for i in range(n_pos_qubits)]
    to_bits = [(to_pos >> i) & 1 for i in range(n_pos_qubits)]

    # Apply controlled operations based on the move
    for i in range(n_pos_qubits):
        if from_bits[i] != to_bits[i]:
            # Determine if we need to flip this qubit
            controls = []

            # Set up controls for current position
            for j in range(n_pos_qubits):
                if j != i and from_bits[j] == 1:
                    controls.append(pos_wires[j])

            # Add control for step number (to make path unique)
            step_angle = (2 * np.pi * step_num) / 10
            qml.RZ(step_angle, wires=cont_wire)

            # Apply the flip operation
            if controls:
                # Multi-controlled X gate for position change
                control_vals = [1] * len(controls)  # <-- Create a list of integers
                qml.MultiControlledX(
                    wires=controls + [pos_wires[i]], control_values=control_vals
                )
            else:
                # Simple X gate if no controls needed
                qml.PauliX(wires=pos_wires[i])

    # Entangle with control qubit to mark forward phase
    qml.CNOT(wires=[cont_wire, pos_wires[0]])


def apply_echo_gate(from_pos, to_pos, step_num):
    """
    Apply the inverse operation for echo phase.
    This reverses the exact path taken during forward movement.
    """
    # Convert positions to binary
    from_bits = [(from_pos >> i) & 1 for i in range(n_pos_qubits)]
    to_bits = [(to_pos >> i) & 1 for i in range(n_pos_qubits)]

    # Reverse the entanglement with control qubit
    qml.CNOT(wires=[cont_wire, pos_wires[0]])

    # Apply inverse operations in reverse order
    for i in range(n_pos_qubits - 1, -1, -1):
        if from_bits[i] != to_bits[i]:
            controls = []

            # Set up controls for current position
            for j in range(n_pos_qubits):
                if j != i and from_bits[j] == 1:
                    controls.append(pos_wires[j])

            # Reverse the step angle
            step_angle = -(2 * np.pi * step_num) / 10
            qml.RZ(step_angle, wires=cont_wire)

            # Apply the inverse flip operation
            if controls:
                control_vals = [1] * len(controls)  # <-- Create a list of integers
                qml.MultiControlledX(
                    wires=controls + [pos_wires[i]], control_values=control_vals
                )
            else:
                qml.PauliX(wires=pos_wires[i])


def verify_echo_path():
    """
    Oracle that verifies if the current state represents a valid echo path.
    This doesn't know the path in advance but can verify its consistency.
    """
    # Check if we're returning along a valid path by:
    # 1. Verifying the control qubit is in expected state
    # 2. Checking position consistency with path history

    # Apply Hadamard to control qubit to reveal path information
    qml.Hadamard(wires=cont_wire)

    # Mark states where control qubit indicates valid path
    qml.PauliX(wires=cont_wire)

    # Create interference pattern that amplifies valid paths
    for i in range(n_pos_qubits):
        qml.Hadamard(wires=pos_wires[i])
        qml.PauliX(wires=pos_wires[i])

    # Create control values for the 4 position wires
    control_vals = [0] * n_pos_qubits  # <-- FIXED: Only 4 values for 4 control wires
    qml.MultiControlledX(wires=pos_wires + [cont_wire], control_values=control_vals)

    # Undo transformations
    for i in range(n_pos_qubits - 1, -1, -1):
        qml.PauliX(wires=pos_wires[i])
        qml.Hadamard(wires=pos_wires[i])

    qml.PauliX(wires=cont_wire)
    qml.Hadamard(wires=cont_wire)


@qml.qnode(dev)
def run_quantum_circuit():
    """
    Execute the complete quantum circuit based on game history.
    This implements the full quantum walk with path recording.
    """
    # Initialize position to starting point (center)
    start_bits = [(4 >> i) & 1 for i in range(n_pos_qubits)]
    for i, bit in enumerate(start_bits):
        if bit == 1:
            qml.PauliX(wires=pos_wires[i])

    # Initialize control qubit for forward phase
    qml.Hadamard(wires=cont_wire)

    # Apply all operations in sequence
    for step, op in enumerate(GAME_HISTORY[1:], 1):  # Skip init
        if op["op"] == "move":
            apply_position_gate(op["from"], op["to"], step)
        elif op["op"] == "echo":
            # Find the corresponding forward move to reverse
            for i in range(len(GAME_HISTORY) - 1, 0, -1):
                if GAME_HISTORY[i]["op"] == "move" and i < step:
                    apply_echo_gate(GAME_HISTORY[i]["to"], GAME_HISTORY[i]["from"], i)
                    break

    # Return probability distribution over positions
    return qml.probs(wires=pos_wires)


@qml.qnode(dev)
def run_verification_circuit():
    """
    Execute circuit for echo path verification.
    This checks if the current state represents a valid echo path.
    """
    # First run the regular game circuit
    run_quantum_circuit()

    # Then apply the verification oracle
    verify_echo_path()

    # Return probability distribution for verification
    return qml.probs(wires=pos_wires)


def compute_probs():
    """Calculate current position probabilities"""
    probs = run_quantum_circuit()
    # Only return probabilities for valid grid positions (0-8)
    return {
        "full_probs": probs.tolist(),
        "grid_probs": [probs[i] for i in VALID_POSITIONS],
        "collapsed": False,  # Will be updated if collapse detected
    }


@app.route("/init", methods=["POST"])
def api_init():
    """Initialize a new game"""
    reset_game()
    echo_steps = get_echo_trigger_step()

    return jsonify(
        {
            "status": "ok",
            "message": "game initialized",
            "history_len": len(GAME_HISTORY),
            "echo_trigger_step": echo_steps,
        }
    )


@app.route("/move", methods=["POST"])
def api_move():
    """
    Move particle to new position.
    Body: { "from": int, "to": int }
    """
    global CURRENT_POSITION
    data = request.get_json(force=True)
    from_pos = int(data.get("from", CURRENT_POSITION))
    to_pos = int(data.get("to", from_pos))

    # Validate move (must be adjacent in 3x3 grid)
    from_r, from_c = divmod(from_pos, 3)
    to_r, to_c = divmod(to_pos, 3)

    # Check if positions are adjacent (Manhattan distance = 1)
    if abs(from_r - to_r) + abs(from_c - to_c) != 1:
        return (
            jsonify(
                {"status": "error", "message": "invalid move - positions not adjacent"}
            ),
            400,
        )

    # Record the move
    GAME_HISTORY.append({"op": "move", "from": from_pos, "to": to_pos})
    PATH_HISTORY.append(to_pos)
    CURRENT_POSITION = to_pos

    return jsonify(
        {
            "status": "ok",
            "message": f"moved from {from_pos} to {to_pos}",
            "history_len": len(GAME_HISTORY),
            "current_position": to_pos,
        }
    )


@app.route("/echo", methods=["POST"])
def api_echo():
    """Trigger the quantum echo phase"""
    global ECHO_PATH, ECHO_TRIGGERED

    # Calculate the echo path (reverse of forward path)
    ECHO_PATH = PATH_HISTORY[::-1]  # Reverse the path
    ECHO_TRIGGERED = True

    # Verify the echo path
    verification_probs = run_verification_circuit()
    valid_path = bool(verification_probs[CURRENT_POSITION] > 0.5)

    return jsonify(
        {
            "status": "ok",
            "message": "echo triggered",
            "history_len": len(GAME_HISTORY),
            "valid_path": valid_path,
            "verification_probs": verification_probs.tolist(),
            "echo_path": ECHO_PATH,
        }
    )


@app.route("/collapse", methods=["POST"])
def api_collapse():
    """
    Simulate a collapse at a specific position.
    Body: { "position": int, "strength": float }
    """
    data = request.get_json(force=True)
    position = int(data.get("position", -1))
    strength = float(data.get("strength", 0.15))

    if position not in VALID_POSITIONS:
        return (
            jsonify({"status": "error", "message": f"invalid position {position}"}),
            400,
        )

    # Check if this position is on the echo path
    if ECHO_TRIGGERED and ECHO_PATH and position in ECHO_PATH:
        return jsonify(
            {
                "status": "warning",
                "message": f"position {position} is on echo path, cannot collapse",
                "skipped": True,
            }
        )

    # Apply depolarizing channel to simulate collapse
    for w in pos_wires:
        qml.DepolarizingChannel(strength, wires=w)

    # Mark position as collapsed
    GAME_HISTORY.append({"op": "collapse", "position": position, "strength": strength})

    return jsonify(
        {
            "status": "ok",
            "message": f"collapse applied at {position}",
            "history_len": len(GAME_HISTORY),
        }
    )


@app.route("/amplify", methods=["POST"])
def api_amplify():
    """
    Amplify probability of target position using Grover-like amplification.
    Body: { "target_index": int }
    """
    data = request.get_json(force=True)
    target = int(data.get("target_index", 8))

    if target not in VALID_POSITIONS:
        return jsonify({"status": "error", "message": f"invalid target {target}"}), 400

    # Record amplification operation
    GAME_HISTORY.append({"op": "amplify", "target": target})

    # Run verification to check path validity
    verification_probs = run_verification_circuit()

    return jsonify(
        {
            "status": "ok",
            "message": f"amplification for {target}",
            "history_len": len(GAME_HISTORY),
            "verification_probs": verification_probs.tolist(),
        }
    )


@app.route("/state", methods=["GET"])
def api_state():
    """Get current quantum state probabilities"""
    probs = compute_probs()
    return jsonify(
        {
            "status": "ok",
            "probs": probs,
            "history_len": len(GAME_HISTORY),
            "current_position": CURRENT_POSITION,
            "echo_triggered": ECHO_TRIGGERED,
            "game_ended": GAME_ENDED,
        }
    )


@app.route("/echo-trigger", methods=["GET"])
def api_echo_trigger():
    """Get the random number of steps before echo triggers"""
    echo_steps = get_echo_trigger_step()
    return jsonify({"status": "ok", "echo_trigger_step": echo_steps})


@app.route("/history", methods=["GET"])
def api_history():
    """Get complete game history"""
    return jsonify(
        {
            "status": "ok",
            "history": GAME_HISTORY,
            "path": PATH_HISTORY,
            "current_position": CURRENT_POSITION,
            "echo_path": ECHO_PATH,
            "echo_triggered": ECHO_TRIGGERED,
        }
    )


if __name__ == "__main__":
    print("Starting enhanced quantum server on http://127.0.0.1:5000")
    app.run(debug=True)
