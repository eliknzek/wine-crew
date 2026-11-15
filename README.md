# wine-crew

מערכת CrewAI שחוזה את ניקוד היין (`points`) לפי מחיר, מקור, זן ובציר, ומאתרת יינות שקיבלו ניקוד גבוה ממה שהמחיר שלהם צפוי לתת. Flow אחד מריץ שני crews בשרשרת, Data Analyst ואחריו Data Scientist, ויש בדיקות דטרמיניסטיות בין השלבים. האפיון המלא נמצא ב-[`docs/spec.md`](docs/spec.md), וחוזה הנתונים ב-[`docs/dataset_contract.json`](docs/dataset_contract.json).

> הפרויקט בשלב 1 (הקמת ה-repo). המודולים ב-`src/` הם עדיין שלד.

## התקנה

נדרש Python 3.13, כי חלק מהתלויות של crewai עוד לא תומכות ב-3.14.

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
cp .env.example .env   # ולמלא OPENAI_API_KEY
```

## נתונים

הקובץ הגולמי (Wine Reviews מ-Kaggle, רישיון CC BY-NC-SA 4.0, כ-50MB) לא נשמר ב-repo. כדי להוריד אותו:

```bash
python scripts/download_data.py          # מדלג אם הקובץ כבר קיים
python scripts/download_data.py --force  # מוריד מחדש
```

הסקריפט שומר את הקובץ ב-`data/raw/` ומדפיס את ה-sha256 שלו. פרטי ההתחברות ל-Kaggle לא נלקחים מ-`.env`. אפשר להשתמש ב-`kaggle auth login`, במשתנה `KAGGLE_API_TOKEN`, ב-`~/.kaggle/access_token` או ב-`~/.kaggle/kaggle.json`.

## הרצה ובדיקות

```bash
python -m wine_crew.main   # מריץ את ה-Flow (ימומש בשלב 5)
pytest
```

## מבנה

| נתיב | תוכן |
| --- | --- |
| `config.yaml` | seed (42), נתיבים, ספים, מודל LLM |
| `src/wine_crew/tools/` | כלים דטרמיניסטיים: ניקוי, EDA, חוזה, מודלים |
| `src/wine_crew/crews/` | `analyst` (Crew 1) ו-`scientist` (Crew 2) |
| `src/wine_crew/flow.py`, `checks.py` | ה-Flow ושערי הבדיקה |
| `artifacts/` | תוצרי הריצה (נכנסים ל-Git) |
| `logs/` | לוג לכל ריצה |
| `app/` | אפליקציית Streamlit |
| `tests/` | בדיקות pytest |

## עבודה ב-Git

כל שינוי נעשה ב-branch נפרד ונכנס ל-`main` רק דרך Pull Request לפי התבנית ב-`.github/pull_request_template.md`.
