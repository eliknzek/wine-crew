# דוח הערכת מודלים

משתנה המטרה: `points`. חלוקה 80/20 (89,229 אימון, 22,308 בדיקה), 5-fold CV על קבוצת האימון, seed 42. המודל נבחר לפי MAE ממוצע ב-CV.

## מדדים

| מודל | CV MAE | CV MAE std | MAE (test) | RMSE (test) | R² (test) |
| --- | --- | --- | --- | --- | --- |
| baseline | 2.538 | 0.015 | 2.538 | 3.101 | -0.000 |
| v1 | 1.828 | 0.010 | 1.822 | 2.296 | 0.452 |
| **v2** (נבחר) | 1.699 | 0.011 | 1.690 | 2.152 | 0.518 |

## 50 היינות המובילים בדירוג השווים למחירם

value_score = ניקוד בפועל פחות הניקוד שחזה v2, עם תחזית out-of-fold (cross_val_predict), כך שאף יין לא נחזה ע"י מודל שראה אותו.

| # | title | country | variety | vintage | price | points | predicted | value_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Donkey & Goat 2010 Fenaughty Vineyard Syrah (El Dorado) | US | Syrah | 2010 | $35 | 97 | 87.92 | 9.08 |
| 2 | Failla 2010 Estate Vineyard Chardonnay (Sonoma Coast) | US | Chardonnay | 2010 | $44 | 99 | 90.12 | 8.88 |
| 3 | Mt. Brave 2008 Merlot (Mount Veeder) | US | Merlot | 2008 | $60 | 98 | 89.30 | 8.70 |
| 4 | EOS 2006 Tears of Dew Late Harvest Moscato (Paso Robles) | US | Moscato | 2006 | $22 | 95 | 86.36 | 8.64 |
| 5 | Au Bon Climat 2007 Los Alamos Vineyard Chardonnay (Santa Barbara County) | US | Chardonnay | 2007 | $25 | 95 | 86.67 | 8.33 |
| 6 | Quady NV Batch 88 Starboard Red (California) | US | Red Blend |  | $24 | 94 | 85.74 | 8.26 |
| 7 | Alma Rosa 2006 Chardonnay (Santa Barbara County) | US | Chardonnay | 2006 | $20 | 94 | 85.78 | 8.22 |
| 8 | Tamarack Cellars 2008 DuBrul Vineyard Reserve Red Red (Rattlesnake Hills) | US | Red Blend | 2008 | $50 | 96 | 87.87 | 8.13 |
| 9 | Rulo 2007 Syrah (Columbia Valley (WA)) | US | Syrah | 2007 | $20 | 96 | 87.99 | 8.01 |
| 10 | Trefethen 2005 Estate Cabernet Sauvignon (Oak Knoll District) | US | Cabernet Sauvignon | 2005 | $50 | 97 | 89.05 | 7.95 |
| 11 | Vina Robles 2006 Jardine Petite Sirah (Paso Robles) | US | Petite Sirah | 2006 | $26 | 93 | 85.08 | 7.92 |
| 12 | Newton 2007 Cabernet Sauvignon (Napa County) | US | Cabernet Sauvignon | 2007 | $28 | 94 | 86.13 | 7.87 |
| 13 | Williams Selyem 2010 Hirsch Vineyard Pinot Noir (Sonoma Coast) | US | Pinot Noir | 2010 | $75 | 99 | 91.25 | 7.75 |
| 14 | Joseph Carr 2009 Dijon Clone Chardonnay (Sonoma Coast) | US | Chardonnay | 2009 | $18 | 94 | 86.25 | 7.75 |
| 15 | Treana 2006 Mer Soleil Vineyard White (Central Coast) | US | Rhône-style White Blend | 2006 | $25 | 94 | 86.28 | 7.72 |
| 16 | Taylor Fladgate NV 325 Anniversary  (Port) | Portugal | Port |  | $40 | 97 | 89.29 | 7.71 |
| 17 | Vina Robles 2007 Jardine Petite Sirah (Paso Robles) | US | Petite Sirah | 2007 | $26 | 93 | 85.42 | 7.58 |
| 18 | Quady 2006 Starboard Dessert Wine Port (Amador County) | US | Port | 2006 | $34 | 95 | 87.44 | 7.56 |
| 19 | Scott Paul 2009 Dix Pinot Noir (Dundee Hills) | US | Pinot Noir | 2009 | $40 | 96 | 88.48 | 7.52 |
| 20 | Marquis Philips 2000 Sarah's Blend Red (South Eastern Australia) | Australia | Red Blend | 2000 | $15 | 93 | 85.53 | 7.47 |
| 21 | Uvaggio 2009 Vermentino (Lodi) | US | Vermentino | 2009 | $14 | 93 | 85.53 | 7.47 |
| 22 | Goldschmidt 2005 Game Ranch Single Vineyard Selection Cabernet Sauvignon (Oakville) | US | Cabernet Sauvignon | 2005 | $65 | 96 | 88.60 | 7.40 |
| 23 | Sbragia 2006 Monte Rosso Vineyard Cabernet Sauvignon (Sonoma Valley) | US | Cabernet Sauvignon | 2006 | $50 | 95 | 87.66 | 7.34 |
| 24 | Miraflores NV Muscat Canelli (El Dorado) | US | Muscat Canelli |  | $50 | 95 | 87.68 | 7.32 |
| 25 | Novy 2006 Oley Dessert Wine Viognier (Russian River Valley) | US | Viognier | 2006 | $14 | 93 | 85.70 | 7.30 |
| 26 | Carlisle 2013 The Derivative White (Sonoma County) | US | White Blend | 2013 | $28 | 95 | 87.70 | 7.30 |
| 27 | Arrowood 2009 Saralee's Vineyard Viognier (Russian River Valley) | US | Viognier | 2009 | $30 | 95 | 87.71 | 7.29 |
| 28 | Isole e Olena 2010  Chianti Classico | Italy | Red Blend | 2010 | $27 | 96 | 88.75 | 7.25 |
| 29 | Chalk Hill 2008 Sauvignon Gris (Chalk Hill) | US | Sauvignon Gris | 2008 | $40 | 94 | 86.75 | 7.25 |
| 30 | Au Bon Climat 2007 Santa Barbara Historic Vineyards Collection Sanford & Benedict Vineyard Chardonnay (Santa Ynez Valley) | US | Chardonnay | 2007 | $30 | 95 | 87.75 | 7.25 |
| 31 | Gloria Ferrer 2007 Pinot Noir (Carneros) | US | Pinot Noir | 2007 | $22 | 94 | 86.77 | 7.23 |
| 32 | Trefethen 2005 Reserve Cabernet Sauvignon (Oak Knoll District) | US | Cabernet Sauvignon | 2005 | $100 | 99 | 91.82 | 7.18 |
| 33 | Uvaggio 2010 Secco Moscato Giallo (Lodi) | US | Moscato Giallo | 2010 | $16 | 94 | 86.86 | 7.14 |
| 34 | Bodegas Gutiérrez de la Vega 1999 Casta Diva Fondillón Sweet Monastrell (Alicante) | Spain | Monastrell | 1999 | $88 | 97 | 89.87 | 7.13 |
| 35 | Stolpman 2007 La Croce Red (Santa Ynez Valley) | US | Red Blend | 2007 | $66 | 96 | 88.88 | 7.12 |
| 36 | Mercy 2008 Cedar Lane Vineyard Pinot Noir (Arroyo Seco) | US | Pinot Noir | 2008 | $35 | 94 | 86.92 | 7.08 |
| 37 | Redmon 2006 Cabernet Sauvignon (St. Helena) | US | Cabernet Sauvignon | 2006 | $65 | 97 | 89.92 | 7.08 |
| 38 | Wild Hogge 2011 Pinot Noir (Paso Robles) | US | Pinot Noir | 2011 | $34 | 94 | 86.92 | 7.08 |
| 39 | Coquelicot 2008 Sixer Red (Happy Canyon of Santa Barbara) | US | Rhône-style Red Blend | 2008 | $35 | 94 | 86.93 | 7.07 |
| 40 | Domaine Marcel Deiss 2011 White (Alsace) | France | White Blend | 2011 | $20 | 94 | 86.96 | 7.04 |
| 41 | Dutton Estate 2006 Dutton Ranch Kylie's Cuvée Sauvignon Blanc (Russian River Valley) | US | Sauvignon Blanc | 2006 | $17 | 93 | 85.97 | 7.03 |
| 42 | Petroni 2006 Rosso di Sonoma Estate Red (Sonoma Valley) | US | Red Blend | 2006 | $30 | 92 | 85.00 | 7.00 |
| 43 | Melville 2010 Verna's Estate Pinot Noir (Santa Barbara County) | US | Pinot Noir | 2010 | $26 | 93 | 86.01 | 6.99 |
| 44 | Il Colle 2011  Brunello di Montalcino | Italy | Sangiovese | 2011 | $45 | 96 | 89.02 | 6.98 |
| 45 | Pirouette 2008 Red Wine Red (Columbia Valley (WA)) | US | Bordeaux-style Red Blend | 2008 | $50 | 98 | 91.03 | 6.97 |
| 46 | Garnet 2012 Pinot Noir (Monterey County) | US | Pinot Noir | 2012 | $15 | 92 | 85.03 | 6.97 |
| 47 | Boekenoogen 2007 Chardonnay (Santa Lucia Highlands) | US | Chardonnay | 2007 | $34 | 96 | 89.03 | 6.97 |
| 48 | Sean Thackrey NV Pleiades XIX Old Vines Red (California) | US | Red Blend |  | $24 | 93 | 86.05 | 6.95 |
| 49 | Volker Eisele Family Estate 2006 Gemini White Wine White (Napa Valley) | US | Bordeaux-style White Blend | 2006 | $25 | 93 | 86.06 | 6.94 |
| 50 | Krug 2002 Brut  (Champagne) | France | Champagne Blend | 2002 | $259 | 100 | 93.07 | 6.93 |
