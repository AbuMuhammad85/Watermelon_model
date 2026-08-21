# Class labels and translations

DISEASE_LABELS = [
    "watermelon___anthracnose",
    "watermelon___downy_mildew",
    "watermelon___healthy",
    "watermelon___mosaic_virus"
]

TRANSLATIONS = {
    # Diseases
    "watermelon___anthracnose": {
        "en": "Anthracnose",
        "ha": "Ciwon Anthracnose (Tabo a Ganye)"
    },
    "watermelon___downy_mildew": {
        "en": "Downy Mildew",
        "ha": "Ciwon Downy Mildew (Dorawa/Yellow Spots)"
    },
    "watermelon___healthy": {
        "en": "Healthy Watermelon Leaf",
        "ha": "Lafiyayyen Ganye (Babu Ciwo)"
    },
    "watermelon___mosaic_virus": {
        "en": "Mosaic Virus",
        "ha": "Ciwon Mosaic Virus (Kuraje/Karkace Ganye)"
    },
    
    # Statuses
    "confident": {
        "en": "Confident Diagnosis",
        "ha": "Tabbataccen Bincike"
    },
    "uncertain": {
        "en": "Uncertain / Ambiguous",
        "ha": "Babu Tabbas (A Sake Daukar Hoto)"
    },
    "not_watermelon": {
        "en": "Rejected (Not Watermelon)",
        "ha": "An Yi Watsi (Ba Ganyen Kankana Ba)"
    }
}

def get_label_info(label_key, lang="en"):
    """
    Returns localized class name for a given class label.
    """
    if label_key in TRANSLATIONS:
        return TRANSLATIONS[label_key].get(lang, label_key)
    return label_key
