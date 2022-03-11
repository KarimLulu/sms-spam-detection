import re
from pathlib import Path

repo_dir = Path(__file__).resolve().parents[1]
data_dir = repo_dir / "data"
models_dir = data_dir / "models"
DATAFILE = 'sms-uk-total.xlsx'

model_extension = ".model"
model_id = "current_model"


# Modeling parameters
THRESHOLD = 0.5
SPAM_LABEL = "spam"
HAM_LABEL = "ham"
ROUND = 1
CURRENCY_PATT = u"[$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6]"
PATTERNS = [(r"[\(\d][\d\s\(\)-]{8,15}\d", {"name": "phone",
                                            "is_len": 0}),
            (r"%|taxi|скид(?:к|очн)|ц[іе]н|знижк", {"name": "custom",
                                                    "is_len": 0,
                                                    "flags": re.I | re.U}),
            (r"[.]", {"name": "dot", "is_len": 0}),
            (CURRENCY_PATT, {"name": "currency", "is_len": 0, "flags": re.U}),
            (r":\)|:\(|-_-|:p|:v|:\*|:o|B-\)|:’\(", {"name": "emoji", "is_len": 0, "flags": re.U}),
            (r"[0-9]{2,4}[.-/][0-9]{2,4}[.-/][0-9]{2,4}", {"name": "date", "is_len": 0})
            ]
NAMES = ["logit"]
TF_PARAMS = {"lowercase": True,
             "analyzer": "char_wb",
             "stop_words": None,
             "ngram_range": (4, 4),
             "min_df": 0.0,
             "max_df": 1.0,
             "preprocessor": None,
             "max_features": 4000,
             "norm": "l2" * 0,
             "use_idf": 1
             }
TOKEN_FEATURES = ["is_upper", "is_lower"]