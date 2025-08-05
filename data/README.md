### `bif_save.py`

- **Purpose**: Converts all Bayesian Networks from R into `.bif` format (Bayesian Interchange Format).

---

### `bif_filter.py`

- **Purpose**: Applies filtering criteria to the BNs.
- **Filters Applied**:
  - Removes networks with missing Conditional Probability Table (CPT) data.
  - Keeps only networks with **50 nodes or fewer**.

---

### `get_state_json.py`

- **Purpose**: Extracts and stores information about each node and its possible states across all BNs.
- **Output**: A JSON file mapping each node to its possible states.

---

### `describe_nodes.py`

- **Purpose**: Reads the JSON produced by `get_state_json.py` and describes each node along with its states.
- **Details**: Uses probability density functions (PDFs) for explanation where applicable.

---

## Example Workflow

1. **Install BnRep Repo**
2. Run `bif_save.py` to convert to `.bif`
3. Run `bif_filter.py` to remove unsuitable networks
4. Run `get_state_json.py` to map node states
5. Run `describe_nodes.py` to interpret node behavior

---
