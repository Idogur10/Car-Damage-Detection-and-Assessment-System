"""
Cost Estimation Module for Car Damage

Maps each combination of (damage_type + severity) to an estimated repair cost range.

These are rough estimates based on US market averages for common repairs.
Actual costs vary significantly based on:
  - Car make/model (luxury vs economy)
  - Location (labor rates differ by region)
  - Paint color (metallic/pearl costs more)
  - Whether the part is repaired vs replaced
  - Shop vs dealership

This module is intentionally simple — it's a lookup table, not a prediction model.
The goal is to give the user a ballpark figure, not a quote.


=== HOW IT WORKS ===

1. Receive damage_type (from YOLO) and severity (from CLIP)
2. Look up the cost range in COST_TABLE
3. Return min/max cost + a note explaining typical repair method

Example:
    estimate = estimate_cost("scratch", "minor")
    → {"min": 50, "max": 200, "method": "Buffing/polishing", ...}
"""


# ---------------------------------------------------------------------------
# COST TABLE — (damage_type, severity) → cost range in USD
# ---------------------------------------------------------------------------
# Each entry has:
#   min: lowest reasonable cost
#   max: highest reasonable cost
#   method: typical repair approach at this severity level
#
# Sources: general US auto body repair averages.
# These are NOT precise — they're order-of-magnitude estimates.

COST_TABLE = {
    "scratch": {
        "minor": {
            "min": 50,
            "max": 200,
            "method": "Buffing and polishing",
        },
        "moderate": {
            "min": 200,
            "max": 800,
            "method": "Touch-up paint or partial respray",
        },
        "severe": {
            "min": 800,
            "max": 2500,
            "method": "Full panel repaint",
        },
    },
    "dent": {
        "minor": {
            "min": 75,
            "max": 250,
            "method": "Paintless dent repair (PDR)",
        },
        "moderate": {
            "min": 250,
            "max": 1000,
            "method": "PDR or body filler + repaint",
        },
        "severe": {
            "min": 1000,
            "max": 3500,
            "method": "Panel replacement + repaint",
        },
    },
    "crack": {
        "minor": {
            "min": 100,
            "max": 300,
            "method": "Filler and spot repair",
        },
        "moderate": {
            "min": 300,
            "max": 1200,
            "method": "Partial panel repair + repaint",
        },
        "severe": {
            "min": 1200,
            "max": 4000,
            "method": "Panel replacement",
        },
    },
    "glass_shatter": {
        "minor": {
            "min": 50,
            "max": 150,
            "method": "Windshield chip repair",
        },
        "moderate": {
            "min": 200,
            "max": 600,
            "method": "Windshield replacement (aftermarket)",
        },
        "severe": {
            "min": 400,
            "max": 1500,
            "method": "Full glass replacement (OEM)",
        },
    },
    "lamp_broken": {
        "minor": {
            "min": 50,
            "max": 200,
            "method": "Lens repair or sealant fix",
        },
        "moderate": {
            "min": 150,
            "max": 500,
            "method": "Aftermarket lamp replacement",
        },
        "severe": {
            "min": 300,
            "max": 1500,
            "method": "OEM headlight/taillight assembly replacement",
        },
    },
    "tire_flat": {
        "minor": {
            "min": 20,
            "max": 50,
            "method": "Tire patch or plug repair",
        },
        "moderate": {
            "min": 100,
            "max": 300,
            "method": "Single tire replacement",
        },
        "severe": {
            "min": 200,
            "max": 800,
            "method": "Tire + rim replacement",
        },
    },
}


def estimate_cost(damage_type, severity):
    """
    Look up the estimated repair cost for a specific damage.

    Args:
        damage_type: string — what YOLO detected ("dent", "scratch", etc.)
        severity: string — what CLIP classified ("minor", "moderate", "severe")

    Returns:
        dict with:
            - min_cost: lowest estimate in USD
            - max_cost: highest estimate in USD
            - average_cost: midpoint of min and max
            - method: typical repair method
            - damage_type: echo back for convenience
            - severity: echo back for convenience

        Returns None if damage_type or severity is not recognized.
    """

    # Normalize: YOLO uses spaces ("glass shatter") but our table uses underscores ("glass_shatter")
    damage_type = damage_type.replace(" ", "_")

    # Check if we have cost data for this damage type
    if damage_type not in COST_TABLE:
        print(f"WARNING: Unknown damage type '{damage_type}', no cost estimate available")
        return None

    # Check if we have cost data for this severity level
    severity_costs = COST_TABLE[damage_type]
    if severity not in severity_costs:
        print(f"WARNING: Unknown severity '{severity}' for {damage_type}")
        return None

    # Look up the cost entry
    cost_entry = severity_costs[severity]

    # Calculate the average (midpoint) for a single "best guess" number
    average_cost = (cost_entry["min"] + cost_entry["max"]) / 2

    return {
        "damage_type": damage_type,
        "severity": severity,
        "min_cost": cost_entry["min"],
        "max_cost": cost_entry["max"],
        "average_cost": average_cost,
        "method": cost_entry["method"],
    }


def estimate_total_cost(detections):
    """
    Estimate the total repair cost for all damages in an image.

    Args:
        detections: list of dicts, each with at least:
            - damage_type: "scratch", "dent", etc.
            - severity: "minor", "moderate", "severe"

    Returns:
        dict with:
            - total_min: sum of all minimum costs
            - total_max: sum of all maximum costs
            - total_average: sum of all average costs
            - per_damage: list of individual cost estimates
            - count: number of damages with valid cost estimates
    """

    total_min = 0
    total_max = 0
    total_average = 0
    per_damage = []

    for detection in detections:
        cost = estimate_cost(detection["damage_type"], detection["severity"])

        if cost is not None:
            total_min += cost["min_cost"]
            total_max += cost["max_cost"]
            total_average += cost["average_cost"]
            per_damage.append(cost)

    return {
        "total_min": total_min,
        "total_max": total_max,
        "total_average": total_average,
        "per_damage": per_damage,
        "count": len(per_damage),
    }


# ---------------------------------------------------------------------------
# Standalone test — see what costs look like
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("COST ESTIMATION TABLE")
    print("=" * 60)

    # Print all costs in a readable table
    print(f"\n{'Damage Type':<16} {'Severity':<10} {'Cost Range':<20} {'Method'}")
    print("-" * 80)

    for damage_type in COST_TABLE:
        for severity in ["minor", "moderate", "severe"]:
            cost = estimate_cost(damage_type, severity)
            cost_range = f"${cost['min_cost']:,} - ${cost['max_cost']:,}"
            print(f"{damage_type:<16} {severity:<10} {cost_range:<20} {cost['method']}")
        print()  # blank line between damage types

    # Example: estimate total for a car with multiple damages
    print("\n" + "=" * 60)
    print("EXAMPLE: Car with 3 damages")
    print("=" * 60)

    example_damages = [
        {"damage_type": "scratch", "severity": "moderate"},
        {"damage_type": "dent", "severity": "minor"},
        {"damage_type": "lamp_broken", "severity": "severe"},
    ]

    total = estimate_total_cost(example_damages)

    for damage_cost in total["per_damage"]:
        print(f"  {damage_cost['damage_type']} ({damage_cost['severity']}): "
              f"${damage_cost['min_cost']:,} - ${damage_cost['max_cost']:,} "
              f"({damage_cost['method']})")

    print(f"\n  TOTAL: ${total['total_min']:,} - ${total['total_max']:,}")
    print(f"  Best estimate: ${total['total_average']:,.0f}")
