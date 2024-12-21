# Prediction-of-music-popularity

# Lineárna Regresia

Najskôr sme sa zamerali na analýzu popularity pomocou lineárnej regresie. Tento model pracuje so stĺpcom **popularity** (nie s binárnym stĺpcom `popularity_bin`), pretože lineárna regresia nie je vhodná na binárne klasifikácie – na tento účel sa používa logistická regresia (viď nižšie). Implementáciu lineárnej regresie sme postupne zlepšovali pomocou cross-validácie. Keďže je veľmi náročné predpovedať presnú hodnotu popularity na úroveň percenta, zaviedli sme toleranciu ±5, ktorú považujeme za prijateľný odhad (tolerancia sa uplatňuje pri výpočte presnosti).

## Kroky

### Príprava Dát
1. **Odstránenie stĺpcov**: Odstránili sme stĺpce ako `duration_ms`, `key` a `tempo`. Na základe korelačnej matice sme zistili, že tieto premenné nemajú významný vplyv na model a ich zahrnutie by ho zbytočne komplikovalo.

### Optimalizácia Modelu
1. **Cross-validácia**: 
   - Optimalizovali sme počet foldov v rámci cross-validácie. Najlepšie výsledky sme pozorovali pri **40 foldoch**, kde sa **Mean Squared Error (MSE)** stabilizovalo.

2. **Regularizácia (Lasso regresia)**:
   - Pri optimalizácii parametra `alpha` pre Lasso regresiu, ktorý znižuje riziko overfittingu, sme zistili, že najlepšie výsledky dosahuje najnižšia hodnota `alpha`. To naznačuje, že regularizácia neprináša výrazný prínos. Preto sme sa rozhodli Lasso regresiu nepoužiť.

3. **Polynomiálna transformácia**:
   - Na základe grafu sme zistili, že dáta nie sú lineárne, a preto sme aplikovali polynomiálnu transformáciu až do tretieho stupňa. Všetky tieto optimalizácie sme implementovali a aplikovali na testovaciu množinu.

### Výsledky
Podarilo sa nám znížiť hodnotu MSE na testovacej sade z pôvodných **298.8** na **256.1**, čo predstavuje zlepšenie o takmer **15 %**. Toto zlepšenie hodnotíme pozitívne. Napriek tomu je predpoveď stále ďaleko od použiteľnej presnosti. Model nám poskytol zaujímavé poznatky o vzťahoch medzi premennými a potvrdil, že lineárna regresia nie je správnym modelom pre tieto dáta. 

### Poznatky o vplyve premenných
- **Pozitívny vplyv na popularitu**: `danceability`
- **Negatívny vplyv na popularitu**: `acousticness`, `energy`, `instrumentalness`, `speechiness`

### Hodnotenie presnosti
Aby sme mohli porovnať náš model s inými prístupmi, rozhodli sme sa použiť metódu presnosti (**accuracy**). Lineárna regresia je v tomto prípade v nevýhode, pretože na rozdiel od iných modelov nepracuje s binárnou premennou `popularity_bin`, ale s reálnou hodnotou popularity. To si vyžaduje veľmi presné predpovede.

Zaviedli sme toleranciu ±5, v rámci ktorej považujeme predikciu za správnu. Aj napriek tejto tolerancii sme dosiahli len **25,7 % presnosti**, čo nie je uspokojivý výsledok. Preto sme sa rozhodli zmeniť prístup a zamerať sa na klasifikáciu.

