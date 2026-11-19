# wine-crew

אפליקציה חיה: https://wine-crew.streamlit.app/

מערכת CrewAI שחוזה את ניקוד היין (`points`) לפי מחיר, מקור, זן ובציר, ומאתרת יינות שקיבלו ניקוד גבוה ממה שהמחיר שלהם צפוי לתת. Flow אחד מריץ שני crews בשרשרת, Data Analyst ואחריו Data Scientist, ויש בדיקות דטרמיניסטיות בין השלבים. האפיון המלא נמצא ב-[`docs/spec.md`](docs/spec.md), וחוזה הנתונים ב-[`docs/dataset_contract.json`](docs/dataset_contract.json).

> שלבים 1–5 ו-7 הושלמו: הכלים, שני ה-crews, ה-Flow עם השערים, ואפליקציית Streamlit.

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
python -m wine_crew.main   # מריץ את כל ה-Flow; exit code 1 אם שער עצר את הריצה
pytest
```

כל ריצה כותבת לוג ל-`logs/run_<timestamp>.log` (מזהה ריצה, זמנים, שורות, תוצאת כל שער, טוקנים ועלות) וסיכום ל-`artifacts/run_summary.json`.

## הרצת האפליקציה

מהשורש של ה-repo:

```bash
streamlit run app/streamlit_app.py
```

אפליקציה בעברית עם חמישה מסכים: סקירה, ניתוח, מודל, מציאת שווים למחיר וחיזוי. היא רק קוראת את התוצרים ב-`artifacts/` ואת `model.pkl`. היא לא מריצה את ה-Flow ולא צריכה מפתח API. כדי להריץ רק את האפליקציה, בלי crewai, מספיק `pip install -r app/requirements.txt`.

## פריסה

האפליקציה פרוסה מ-`main` ב-Streamlit Community Cloud, וכל merge ל-`main` מעדכן את האתר.

- **קובץ הכניסה:** `app/streamlit_app.py`.
- **תלויות:** `app/requirements.txt`, עם גרסאות זהות ל-`requirements.txt` הראשי ובלי crewai. Cloud קורא אותו כי הוא נמצא ליד ה-entrypoint.
- **Python:** 3.13. בוחרים אותו ב-Advanced settings, כי ברירת המחדל שם היא 3.12.
- **חבילת `wine_crew`:** לא מותקנת. האפליקציה מוסיפה את `src/` ל-`sys.path` בעצמה.
- **סודות:** אין צורך בסודות.
- **עיצוב:** הפונט (Rubik) מוגדר ב-`.streamlit/config.toml`.
- **אחרי merge:** האפליקציה טוענת מחדש בעצמה את הקוד ב-`src/wine_crew` (דרך `app/code_reload.py`) ואת התוצרים. המטמון שלהם תלוי בזמן השינוי ובגודל של הקבצים. אם בכל זאת מוצגת שגיאה או גרסה ישנה, עושים Reboot app ב-Streamlit Cloud.

קישור לאפליקציה: https://wine-crew.streamlit.app/

## ריצה בטוחה והדגמת כשל

כל ריצה כותבת קודם לתיקייה זמנית, `build/runs/<run_id>/`. התוצרים עוברים ל-`artifacts/` רק אם כל השערים עברו.

כשריצה נכשלת:
- `artifacts/` נשאר בדיוק כמו שהיה, והאפליקציה ממשיכה להציג את הריצה המוצלחת האחרונה.
- הסיכום הכושל נשמר ב-`logs/run_<ts>_summary.json`.
- התיקייה הזמנית נשארת לבדיקה.
- exit code 1.

הדגמת כשל מבוקר, בלי לגעת ב-`artifacts/`:

```bash
python -m wine_crew.main --raw /tmp/missing.csv     # בדיקה מקדימה נכשלת מיד, בלי LLM
python -m wine_crew.main --demo-fail contract       # Crew 1 רץ, שער 1 נכשל (עמודת price הוסרה מהקובץ הזמני)
python -m wine_crew.main --demo-fail features       # שני ה-crews רצים, שער 2 נכשל (עמודת דליפה taster_name)
python -m wine_crew.main --demo-fail model          # שני ה-crews רצים, שער 3 נכשל (MAE מעל ה-baseline)
```

הדגל `--demo-fail` משבש רק את הקובץ בתיקייה הזמנית, ונרשם בלוג כ-`DEMO`. אפשר גם להריץ על קובץ גולמי פגום, למשל 1,000 שורות ראשונות בתיקייה זמנית, עם `--raw`. במקרה כזה שער 1 נכשל על מספר השורות.

## עלות ריצה

כל ריצה מתעדת לכל crew ולכל משימה כמה בקשות נשלחו לשירות, וכמה טוקני קלט, טוקני קלט מהמטמון וטוקני פלט הן צרכו. הספירה לא לוקחת את ההערכה של CrewAI אלא את הנתונים שהשירות של OpenAI מחזיר בכל קריאה. העלות מחושבת בקוד מהטוקנים, לפי המחירון ב-`config.yaml`. אם מספר הבקשות ש-CrewAI מדווח שונה מהספירה, נכתבת אזהרה בלוג, והריצה ממשיכה.

**תיקון:** מ-PR #5 (שלב 3ב) ועד PR #12, כל ארבעת הסוכנים ב-crew השתמשו באותו אובייקט LLM, ו-CrewAI ספר את השימוש בו פעם אחת לכל סוכן. לכן הטוקנים, מספר הבקשות והעלות שדווחו בלוגים, בסיכומים ובתיאורי ה-PR היו גבוהים פי 4. למשל, בריצה של PR #11 דווחו $0.0166, אבל ב-Usage של OpenAI נרשמו לה 19,471 טוקני קלט (מהם 4,352 מהמטמון) ו-2,586 טוקני פלט, כלומר כ-$0.004. זה גם בערך המחיר האמיתי של ריצה מלאה. עכשיו לכל סוכן יש אובייקט LLM משלו.

## תוצרים ב-Git

התוצרים ב-`artifacts/` נכנסים ל-Git רק בריצות מלאות מכוונות, יחד עם הלוג של אותה ריצה, ולא אחרי כל ריצה. את `clean_data.csv` אפשר לשחזר בדיוק מהקובץ הגולמי עם `python -m wine_crew.main`. ה-sha256 של הקובץ הגולמי נרשם ב-`artifacts/analyst/dataset_contract.json` ובלוג, כך שרואים על איזה קובץ רצו.

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
