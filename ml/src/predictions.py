from typing import List, Tuple


def print_predictions(predictions: List[str], tokens: List[Tuple[str, str, str, str]]):
    inside = False
    entity_form = None
    entity_start = None
    for k in range(0, len(predictions)):
        y = predictions[k]
        (sid, form, offS, offE) = tokens[k]

        if y[0] == "B":
            entity_form = form
            entity_start = offS
            entity_end = offE
            entity_type = y[2:]
            inside = True
        elif y[0] == "I" and inside:
            entity_form += " " + form  # noqa
            entity_end = offE
        elif y[0] == "O" and inside:
            print(sid, entity_start + "-" + entity_end, entity_form, entity_type, sep="|")  # noqa
            inside = False

    if inside:
        print(sid, entity_start + "-" + entity_end, entity_form, entity_type, sep="|")  # noqa
