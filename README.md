# Prediction-of-music-popularity
## Motivacia
Cieľom tejto analýzy je pochopiť, aké faktory ovplyvňujú popularitu skladieb a využiť tieto poznatky na vytváranie predikčných modelov, ktoré dokážu presne odhadnúť úspešnosť skladieb.

# Datasety
Pre analýzu sme zvolili dataset zo stránky Kaggle ([Spotify Audio Features Dataset](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)), ktorý obsahuje 17 premenných umožňujúcich analyzovať faktory ovplyvňujúce popularitu skladieb. Dataset je dostatočne rozsiahly, obsahuje **130 325 záznamov**, čo zvyšuje presnosť a spoľahlivosť našej analýzy. Popisy jednotlivých premenných sme doplnili informáciami z oficiálnej stránky Spotify. 
Okrem základného datasetu sme použili aj **druhý zdroj** ([Colab Notebook](https://colab.research.google.com/drive/1yrrDze9huXwDTgrX3TyUxZDU7yDhIwLW#scrollTo=TTPA9CxJf1dz&line=5&uniqifier=1)), ktorý hodnotí popularitu interpretov na Spotify. Tento zdroj sme vybrali s predpokladom, že popularita skladby úzko súvisí s popularitou interpreta. 

### Spracovanie interpretov
Aby sme interpretov reprezentovali kvantitatívnou premennou namiesto kategórie (mena interpreta), priradili sme im hodnotu popularity podľa počtu streamov:
- Dataset obsahuje **3 000 najpopulárnejších interpretov**, zoradených podľa celkového počtu prehratí.
- Interpretom sme priradili pozíciu v rebríčku (zoradenú vzostupne). Najpopulárnejší interpret získal hodnotu **3 000**, menej populárni interpreti nižšie hodnoty.
Spojili sme oba datasety a obmedzili analýzu len na skladby interpretov z **top 3 000**. Po odstránení duplicít v stĺpci `track_id` (identifikátor pesničky) sme pracovali s **29 281 záznamami**. 

### Problémy v dátach
Premenná **popularity** nie je rovnomerne rozdelená. Väčšina skladieb má nižšiu popularitu, čo môže viesť k tomu, že model sa viac prispôsobí skladbám s nižšou popularitou a bude mať problémy s identifikáciou skladieb s vyššou popularitou. Z tohto dôvodu sme pridali nový stĺpec klasifikujúci skladby ako **populárne** alebo **nepopulárne**.
Aby sme skladbu mohli označiť ako populárnu alebo nepopulárnu, rozhodli sme sa pracovať s hodnotou popularity ako **binárnou premennou**. Prahová hodnota bola stanovená na úrovni **75% kvantilu popularity (hodnota 53)**: Skladby s hodnotou popularity ≥ 53 sme označili za **populárne** (top 25%). Týmto sme pripravili základ pre klasifikačné metódy.
 ![Alt text](images/popularity_hustota.png)



### Korelačná analýza
Pre analýzu korelácií sme zvolili **Spearmanovu koreláciu**, ktorá lepšie zachytáva nelineárne vzťahy. Výrazné korelácie sme pozorovali pri:
- **Pozitívna korelácia**: `danceability`, `energy`, `loudness`, `artist_rank`.
- **Negatívna korelácia**: `acousticness`, `instrumentalness`.
  ![Alt text](images/korelacia.png)


---

# Predikčné Modely

Dáta sme spracovali pomocou štyroch rôznych metód a porovnali ich výsledky na testovacom súbore.

### Regresné Metódy
Cieľom bolo predikovať hodnotu popularity. Aby sme ich mohli porovnať s klasifikačnými metódami, odhad sme považovali za správny, ak bol v tolerancii **±5** od skutočnej hodnoty.
- **Lineárna regresia**
- **KNN regresia**

### Klasifikačné Metódy
Cieľom bolo predikovať popularitu skladby ako **binárnu premennú** (`popularity_bin`), kde skladby označujeme za populárne, ak patria do top 25 %.
- **Logistická regresia**
- **KNN klasifikácia**
- **Rozhodovací strom**

## Lineárna Regresia

Najskôr sme sa zamerali na analýzu popularity pomocou lineárnej regresie. Tento model pracuje so stĺpcom **popularity** (nie s binárnym stĺpcom `popularity_bin`), pretože lineárna regresia nie je vhodná na binárne klasifikácie – na tento účel sa používa logistická regresia (viď nižšie). Implementáciu lineárnej regresie sme postupne zlepšovali pomocou cross-validácie. Keďže je veľmi náročné predpovedať presnú hodnotu popularity na úroveň percenta, zaviedli sme toleranciu ±5, ktorú považujeme za prijateľný odhad (tolerancia sa uplatňuje pri výpočte presnosti).

### Kroky

### Príprava Dát
1. **Odstránenie stĺpcov**: Odstránili sme stĺpce ako `duration_ms`, `key` a `tempo`. Na základe korelačnej matice sme zistili, že tieto premenné nemajú významný vplyv na model a ich zahrnutie by ho zbytočne komplikovalo.

#### Optimalizácia Modelu
1. **Cross-validácia**: 
   - Optimalizovali sme počet foldov v rámci cross-validácie. Najlepšie výsledky sme pozorovali pri **40 foldoch**, kde sa **Mean Squared Error (MSE)** stabilizovalo.

2. **Regularizácia (Lasso regresia)**:
   - Pri optimalizácii parametra `alpha` pre Lasso regresiu, ktorý znižuje riziko overfittingu, sme zistili, že najlepšie výsledky dosahuje najnižšia hodnota `alpha`. To naznačuje, že regularizácia neprináša výrazný prínos. Preto sme sa rozhodli Lasso regresiu nepoužiť.

3. **Polynomiálna transformácia**:
   - Na základe grafu sme zistili, že dáta nie sú lineárne, a preto sme aplikovali polynomiálnu transformáciu až do tretieho stupňa. Všetky tieto optimalizácie sme implementovali a aplikovali na testovaciu množinu.

#### Výsledky
Podarilo sa nám znížiť hodnotu MSE na testovacej sade z pôvodných **298.8** na **256.1**, čo predstavuje zlepšenie o takmer **15 %**. Toto zlepšenie hodnotíme pozitívne. Napriek tomu je predpoveď stále ďaleko od použiteľnej presnosti. Model nám poskytol zaujímavé poznatky o vzťahoch medzi premennými a potvrdil, že lineárna regresia nie je správnym modelom pre tieto dáta. 

#### Poznatky o vplyve premenných
- **Pozitívny vplyv na popularitu**: `danceability`
- **Negatívny vplyv na popularitu**: `acousticness`, `energy`, `instrumentalness`, `speechiness`

#### Hodnotenie presnosti
Aby sme mohli porovnať náš model s inými prístupmi, rozhodli sme sa použiť metódu presnosti (**accuracy**). Lineárna regresia je v tomto prípade v nevýhode, pretože na rozdiel od iných modelov nepracuje s binárnou premennou `popularity_bin`, ale s reálnou hodnotou popularity. To si vyžaduje veľmi presné predpovede.

Zaviedli sme toleranciu ±5, v rámci ktorej považujeme predikciu za správnu. Aj napriek tejto tolerancii sme dosiahli len **25,7 % presnosti**, čo nie je uspokojivý výsledok. Preto sme sa rozhodli zmeniť prístup a zamerať sa na klasifikáciu.

##Logisticka Regresia

Cieľom je vytvoriť model, ktorý dokáže spoľahlivo rozlíšiť medzi populárnymi a menej populárnymi piesňami. Pieseň definujeme ako populárnu, ak jej kvantilová hodnota dosahuje alebo prekračuje 75 %. Túto informáciu reprezentujeme binárnou premennou `popularity_bin`, kde:
- **1** označuje populárnu pieseň
- **0** označuje menej populárnu pieseň.

Na základe širokého spektra hudobných atribútov, ako sú tanečnosť, energia a akustickosť, sme zvolili logistickú regresiu ako vhodný nástroj pre túto binárnu klasifikáciu. Logistická regresia umožňuje odhadnúť pravdepodobnosť, že daná pieseň patrí do kategórie populárnych.

---

### Úprava dát
Pôvodný dataset bol značne nevyvážený v prospech triedy menej populárnych piesní. Aby sme zabránili zaujatosti modelu, použili sme techniku **SMOTE (Synthetic Minority Oversampling Technique)**. Táto metóda synteticky generuje nové príklady pre menšinovú triedu na základe existujúcich dát, čím sme zabezpečili rovnomerné zastúpenie oboch tried.

---

### Výsledky modelu

| Metrika       | Hodnota  |
|---------------|----------|
| **Presnosť**  | 73 %     |
| **Precision** | 55 %     |
| **Recall**    | 75 %     |
| **F1-score**  | 63 %     |

Model dosiahol prijateľnú celkovú presnosť (73 %), avšak detailnejšia analýza odhalila rozdiely vo výkone medzi jednotlivými triedami.

### Výkonnosť podľa tried

#### Trieda 0: Menej populárne piesne
- **Precision**: Vysoká, čo znamená, že väčšina piesní označených ako „menej populárne“ je správna.
- **Recall**: 72 %, čo ukazuje, že model správne identifikuje väčšinu menej populárnych piesní.

#### Trieda 1: Populárne piesne
- **Precision**: 55 %, teda niečo vyše polovice piesní označených ako „populárne“ je skutočne populárnych.
- **Recall**: 75 %, čo znamená, že model dokáže identifikovať väčšinu populárnych piesní.

---

## Matica zámien (Confusion Matrix)

|                 | Predikované menej populárne | Predikované populárne |
|-----------------|-----------------------------|-----------------------|
| **Menej populárne** | 2907 (True Negatives)     | 1113 (False Positives) |
| **Populárne**      | 467 (False Negatives)      | 1370 (True Positives)  |

---

### ROC krivka a AUC
**ROC krivka** graficky zobrazuje vzťah medzi True Positive Rate (Recall) a False Positive Rate pri rôznych prahových hodnotách klasifikácie.

- **AUC (plocha pod krivkou):** 0.81  
  Táto hodnota naznačuje veľmi dobrý výkon modelu pri rozlišovaní medzi populárnymi a menej populárnymi skladbami.

---

### Precision-Recall krivka
**Precision-Recall krivka** ukazuje vzťah medzi presnosťou (precision) a úplnosťou (recall) pri rôznych prahoch klasifikácie.

- Na začiatku krivky, keď je recall nízky, model dosahuje veľmi vysokú presnosť. Znamená to, že model je v tomto režime opatrný a označuje za populárne iba piesne, o ktorých je veľmi presvedčený.
- So zvyšovaním recall (snaha identifikovať viac populárnych piesní) dochádza k poklesu presnosti, pretože model začne nesprávne označovať niektoré piesne ako populárne.
- **Priemerná presnosť (AP):** 0.60, čo naznačuje uspokojivý, ale nie výnimočný výkon modelu.

---

Výsledky modelu logistickej regresie s **ROC-AUC 0.81** a presnosťou **73 %** naznačujú solídny výkon, najmä pri klasifikácii menej populárnych piesní. Model dokáže identifikovať väčšinu populárnych piesní (vysoký recall), avšak nižšia presnosť naznačuje určité problémy pri správnom určení populárnych skladieb.

