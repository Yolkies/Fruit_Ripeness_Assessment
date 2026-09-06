"""Module 2.3 - Colour-Based Fruit Ripeness Assessment."""

from dataclasses import dataclass


from fruit_ripeness.apple_ripeness.apple_ripeness_yolo import (
    assess_apple_ripeness,
)

from fruit_ripeness.banana_ripeness.banana_ripeness_yolo import (
    assess_banana_ripeness,
)

from fruit_ripeness.mango_ripeness.mango_ripeness_yolo import (
    assess_mango_ripeness,
)

from fruit_ripeness.orange_ripeness.orange_ripeness_yolo import (
    assess_orange_ripeness,
)

from fruit_ripeness.strawberry_ripeness.strawberry_ripeness_yolo import (
    assess_strawberry_ripeness,
)

from fruit_ripeness.tomato_ripeness.tomato_ripeness_yolo import (
    assess_tomato_ripeness,
)

from fruit_ripeness.apple_ripeness import apple_ripeness_yolo
from fruit_ripeness.banana_ripeness import banana_ripeness_yolo
from fruit_ripeness.mango_ripeness import mango_ripeness_yolo
from fruit_ripeness.orange_ripeness import orange_ripeness_yolo
from fruit_ripeness.strawberry_ripeness import strawberry_ripeness_yolo
from fruit_ripeness.tomato_ripeness import tomato_ripeness_yolo


# Per-fruit threshold constants live in these modules and are read at call
# time, so the GUI tunes ripeness by writing to them.
FRUIT_MODULES = {
    "apple": apple_ripeness_yolo,
    "banana": banana_ripeness_yolo,
    "mango": mango_ripeness_yolo,
    "orange": orange_ripeness_yolo,
    "strawberry": strawberry_ripeness_yolo,
    "tomato": tomato_ripeness_yolo,
}


@dataclass
class RipenessParams:
    strawberry_supplementary_index1_rule: bool = True


PARAMS = RipenessParams()


def assess_single_fruit(fruit_type, original_crop, fruit_mask):
    """
    Send a detected fruit to its existing ripeness algorithm.
    """
    fruit_type = fruit_type.lower()

    if fruit_type == "apple":
        result, diagnostics = assess_apple_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "banana":
        result, diagnostics = assess_banana_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "mango":
        result, diagnostics = assess_mango_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "orange":
        result, diagnostics = assess_orange_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "strawberry":
        result, diagnostics = assess_strawberry_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
            use_supplementary_index1_rule=PARAMS.strawberry_supplementary_index1_rule,
        )

    elif fruit_type == "tomato":
        result, diagnostics = assess_tomato_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    else:
        raise ValueError(f"Unsupported fruit type: {fruit_type}")

    return result, diagnostics


def normalise_result(fruit_type, result):
    """
    Convert the six different result dataclasses into one
    common format for the UI.
    """
    fruit_type = fruit_type.lower()

    if fruit_type == "apple":
        return {
            "fruit_type": "Apple",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "red_percentage": result.red_pct,
                "yellow_percentage": result.yellow_pct,
                "other_percentage": result.other_pct,
            },
        }

    elif fruit_type == "banana":
        return {
            "fruit_type": "Banana",
            "ripeness_index": result.ripeness_index,
            "ripeness_label": result.ripeness_label,
            "rule_basis": result.rule_basis,
            "details": {
                "yellow_percentage": result.yellow_pct,
                "brown_percentage": result.brown_pct,
                "developed_peel_percentage": result.developed_peel_pct,
                "other_percentage": result.other_pct,
            },
        }

    elif fruit_type == "mango":
        return {
            "fruit_type": "Mango",
            "ripeness_index": result.maturity_stage,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "yellow_percentage": result.yellow_pct,
                "other_percentage": result.other_pct,
            },
        }

    elif fruit_type == "orange":
        return {
            "fruit_type": "Orange",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "yellow_orange_percentage": result.yellow_orange_pct,
                "non_yellow_orange_percentage": result.non_yellow_orange_pct,
                "mean_cci": result.mean_cci,
                "median_cci": result.median_cci,
            },
        }

    elif fruit_type == "strawberry":
        return {
            "fruit_type": "Strawberry",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "red_percentage": result.red_pct,
                "normal_red_percentage": result.normal_red_pct,
                "dark_red_percentage": result.dark_red_pct,
                "other_percentage": result.other_pct,
            },
        }

    elif fruit_type == "tomato":
        return {
            "fruit_type": "Tomato",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "green_percentage": result.green_pct,
                "turning_percentage": result.turning_pct,
                "pink_percentage": result.pink_pct,
                "mature_red_percentage": result.mature_red_pct,
                "colour_break_percentage": result.colour_break_pct,
                "red_orange_percentage": result.red_orange_pct,
            },
        }

    raise ValueError(f"Unsupported fruit type: {fruit_type}")


def assess_detected_fruit(detected_fruit, selected_fruit_type=None):
    """
    Run ripeness assessment after the user confirms
    or overrides YOLO's fruit prediction.
    """
    yolo_fruit_type = detected_fruit["yolo_fruit_type"]

    if selected_fruit_type is None:
        fruit_type = yolo_fruit_type
    else:
        fruit_type = selected_fruit_type.lower().strip()

    original_crop = detected_fruit["crop"]
    fruit_mask = detected_fruit["fruit_mask"]

    result, diagnostics = assess_single_fruit(
        fruit_type,
        original_crop,
        fruit_mask,
    )

    normalised = normalise_result(fruit_type, result)

    normalised["number"] = detected_fruit["number"]
    normalised["confidence"] = detected_fruit["confidence"]
    normalised["yolo_fruit_type"] = yolo_fruit_type
    normalised["selected_fruit_type"] = fruit_type
    normalised["crop"] = original_crop
    normalised["masked_crop"] = detected_fruit["masked_crop"]
    normalised["diagnostics"] = diagnostics

    return normalised


