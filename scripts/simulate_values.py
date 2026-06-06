import numpy as np
import pandas as pd
import json
from scipy.stats import truncnorm
 


class values_simulation():
    # ── Config ─────────────────────────────────────────────────────────────────
    def __init__(self, N: int=12, seed:int =42):
        self.N = N
        self.SEED = seed
        self.rng = np.random.default_rng(self.SEED)
        
        # ── Categorical columns ────────────────────────────────────────────────────
        self.cat_specs = {
            "X2":  {"values": [2, 1],
                    "probs":  [18112, 11888]},
            "X3":  {"values": [2, 1, 3, 5, 4, 6, 0],
                    "probs":  [14030, 10585, 4917, 280, 123, 51, 14]},
            "X4":  {"values": [2, 1, 3, 0],
                    "probs":  [15964, 13659, 323, 54]},
            "X6":  {"values": [0, -1,  1, -2,  2,  3,  4,  5,  8,  6,  7],
                    "probs":  [14737, 5686, 3688, 2759, 2667, 322, 76, 26, 19, 11, 9]},
            "X7":  {"values": [0, -1,  2, -2,  3,  4,  1,  5,  7,  6,  8],
                    "probs":  [15730, 6050, 3927, 3782, 326, 99, 28, 25, 20, 12, 1]},
            "X8":  {"values": [0, -1, -2,  2,  3,  4,  7,  6,  5,  1,  8],
                    "probs":  [15764, 5938, 4085, 3819, 240, 76, 27, 23, 21, 4, 3]},
            "X9":  {"values": [0, -1, -2,  2,  3,  4,  7,  5,  6,  1,  8],
                    "probs":  [16455, 5687, 4348, 3159, 180, 69, 58, 35, 5, 2, 2]},
            "X10": {"values": [0, -1, -2,  2,  3,  4,  7,  5,  6,  8],
                    "probs":  [16947, 5539, 4546, 2626, 178, 84, 58, 17, 4, 1]},
            "X11": {"values": [0, -1, -2,  2,  3,  4,  7,  6,  5,  8],
                    "probs":  [16286, 5740, 4895, 2766, 184, 49, 46, 19, 13, 2]},
        }


    
        # ── Continuous columns ─────────────────────────────────────────────────────
        # Using truncated normal for all: respects observed min/max boundaries.
        # X18-X23: payment amounts floored at 0 (no negatives).
        self.cont_specs = {
            "X1":  {"mean": 167484.32, "std": 129747.66, "min": 10000,    "max": 1000000},
            "X5":  {"mean": 35.49,     "std": 9.22,      "min": 21,       "max": 79},
            "X12": {"mean": 51223.33,  "std": 73635.86,  "min": -165580,  "max": 964511},
            "X13": {"mean": 49179.08,  "std": 71173.77,  "min": -69777,   "max": 983931},
            "X14": {"mean": 47013.15,  "std": 69349.39,  "min": -157264,  "max": 1664089},
            "X15": {"mean": 43262.95,  "std": 64332.86,  "min": -170000,  "max": 891586},
            "X16": {"mean": 40311.40,  "std": 60797.16,  "min": -81334,   "max": 927171},
            "X17": {"mean": 38871.76,  "std": 59554.11,  "min": -339603,  "max": 961664},
            "X18": {"mean": 5663.58,   "std": 16563.28,  "min": 0,        "max": 873552},
            "X19": {"mean": 5921.16,   "std": 23040.87,  "min": 0,        "max": 1684259},
            "X20": {"mean": 5225.68,   "std": 17606.96,  "min": 0,        "max": 896040},
            "X21": {"mean": 4826.08,   "std": 15666.16,  "min": 0,        "max": 621000},
            "X22": {"mean": 4799.39,   "std": 15278.31,  "min": 0,        "max": 426529},
            "X23": {"mean": 5215.50,   "std": 17777.47,  "min": 0,        "max": 528666},
        }
        
    
    # ── Helpers ────────────────────────────────────────────────────────────────
    def sample_categorical(self, values, probs, n):
        """Sample n values from a discrete distribution."""
        return self.rng.choice(values, size=n, p=probs)
    
    
    def sample_truncated_normal(self, mean, std, low, high, n):
        """Sample from a normal distribution clipped to [low, high]."""
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=self.rng)
    

    
    # ── Generate ───────────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        data = {}
        for col, spec in self.cat_specs.items():
            counts = np.array(spec["probs"], dtype=float)
            probs  = counts / counts.sum()
            data[col] = self.sample_categorical(spec["values"], probs, self.N)
        
        for col, spec in self.cont_specs.items():
            data[col] = self.sample_truncated_normal(
                spec["mean"], spec["std"], spec["min"], spec["max"], self.N
            )
    
        # ── Assemble and sort columns X1 → X23 ────────────────────────────────────
        df = pd.DataFrame(data)
        col_order = [f"X{i}" for i in range(1, 24)]
        df = df[col_order]
        
        # Round integer-natured columns
        int_cols = ["X1", "X5"] + [f"X{i}" for i in range(12, 24)]
        df[int_cols] = df[int_cols].round(0).astype(int)
        return df

    def generate_json_payload(self, as_string: bool = False):
            """
            Generates the simulation data and converts it into a valid format 
            for the Flask endpoint.
            
            :param as_string: If True, returns a JSON string. If False, returns 
                            a Python list of dicts (ready for requests.post(json=...)).
            """
            df = self.generate()
            
            # Convert to a list of records (rows as dictionaries)
            payload = df.to_dict(orient='records')
            
            if as_string:
                return json.dumps(payload)
            return payload