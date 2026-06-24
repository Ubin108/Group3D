OV3DET_CLASSES = [
    "toilet", "bed", "chair", "sofa", "dresser",
    "table", "cabinet", "bookshelf", "pillow", "sink",
    "bathtub", "refrigerator", "desk", "nightstand", "counter",
    "door", "curtain", "box", "lamp", "bag",
]

NAME_TO_ID_20 = {name: i for i, name in enumerate(OV3DET_CLASSES)}

NYU40_TO_OV3DET = {
    33: NAME_TO_ID_20["toilet"],
    4:  NAME_TO_ID_20["bed"],
    5:  NAME_TO_ID_20["chair"],
    6:  NAME_TO_ID_20["sofa"],
    17: NAME_TO_ID_20["dresser"],
    7:  NAME_TO_ID_20["table"],
    3:  NAME_TO_ID_20["cabinet"],
    10: NAME_TO_ID_20["bookshelf"],
    18: NAME_TO_ID_20["pillow"],
    34: NAME_TO_ID_20["sink"],
    36: NAME_TO_ID_20["bathtub"],
    24: NAME_TO_ID_20["refrigerator"],
    14: NAME_TO_ID_20["desk"],
    32: NAME_TO_ID_20["nightstand"],
    12: NAME_TO_ID_20["counter"],
    8:  NAME_TO_ID_20["door"],
    16: NAME_TO_ID_20["curtain"],
    29: NAME_TO_ID_20["box"],
    35: NAME_TO_ID_20["lamp"],
    37: NAME_TO_ID_20["bag"],
}
