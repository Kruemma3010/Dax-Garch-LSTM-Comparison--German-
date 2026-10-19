# DAX-Renditeschwankungen und Value at Risk: GARCH vs. LSTM

**Python · Zeitreihenanalyse · Deep Learning · Risikomessung**  
Autor: **Marc Krümpelmann**

Liefert ein neuronales Netz einen erkennbaren Zusatznutzen gegenüber einem sparsamen ökonometrischen Modell für tägliche Risikoprognosen? Dieses Projekt vergleicht GARCH mit Student-t-Innovationen und ein LSTM für absolute DAX-Tagesrenditen mit anschließender t-Kalibrierung.

Das Notebook führt von der Datenprüfung über chronologische Modellauswahl und Tagesprognosen bis zum VaR-Backtesting. Es ist eine überarbeitete Portfolio-Fassung eines Universitätsprojekts.

## Ergebnis auf einen Blick

Auf **550 Bewertungstagen** wird bei keinem der beiden VaR-Modelle die Nullhypothese des Kupiec-, Unabhängigkeits- oder Conditional-Coverage-Tests auf dem 5-%-Niveau verworfen.

| Kennzahl | GARCH(1,1) | LSTM mit t-Kalibrierung |
|---|---:|---:|
| Erwartete Überschreitungsquote | 5,00 % | 5,00 % |
| Überschreitungen | 32 | 25 |
| Beobachtete Überschreitungsquote | 5,82 % | 4,55 % |
| Kupiec-p-Wert | 0,3903 | 0,6196 |
| Unabhängigkeits-p-Wert | 0,4127 | 0,1217 |
| Conditional-Coverage-p-Wert | 0,4943 | 0,2668 |

**Einordnung:** Beide Verfahren zeigen im betrachteten Zeitraum unauffällige Backtests. Die LSTM-Quote liegt näher an 5 %, doch daraus folgt keine statistisch belegte Überlegenheit. Ein höherer p-Wert ist kein Qualitätsranking. Das einfachere GARCH-Modell liefert hier ebenfalls eine plausible VaR-Abdeckung.

Die Tabelle dokumentiert einen lokal ausgeführten Lauf. Bei erneuter Ausführung können insbesondere die LSTM-Ergebnisse abweichen; die Tabelle wird nicht automatisch aktualisiert. Aktuelle MAE- und RMSE-Werte sowie Grafiken stehen in den Notebookausgaben.

## Gezeigte Fähigkeiten

- Datenqualität prüfen, Duplikate entfernen und finanzielle Zeitreihen aufbereiten.
- Modellauswahl und Bewertung zeitlich trennen; Vorverarbeitung ohne Zugriff auf spätere Beobachtungen anpassen.
- Ökonometrische und neuronale Modelle implementieren und diagnostisch untersuchen.
- Renditeprognosen in Verlustschwellen übersetzen und statistisch backtesten.
- Ergebnisse verständlich erklären und Grenzen der Aussagekraft benennen.

## Daten und Bereinigung

**Quelle:** DAX-Performanceindex (`.GDAXI`), Schlussstände aus LSEG Workspace / Refinitiv.  
**Zeitraum:** 02.01.2015–31.10.2025. Die CSV wird lokal mitgeliefert; ein API-Zugang ist für die Ausführung nicht notwendig.

| Verarbeitungsschritt | Umfang |
|---|---:|
| Export | 2.826 Zeilen |
| Entfernte exakte Duplikate | 75 |
| Bereinigte Kursbeobachtungen | 2.751 |
| Log-Renditen nach erster Differenz | 2.750 |

Renditen werden als `100 * log(P_t / P_(t-1))` berechnet. Geprüft werden außerdem Datum, fehlende Werte, positive Indexstände und widersprüchliche Einträge desselben Tages. Die ursprüngliche CSV bleibt unverändert; die Bereinigung erfolgt im Notebook.

## Untersuchungsdesign

### Chronologische Aufteilung: 70/10/20

| Abschnitt | Renditen | Zeitraum | Aufgabe |
|---|---:|---|---|
| Training (70 %) | 1.925 | 05.01.2015–09.08.2022 | GARCH-Kandidaten schätzen; LSTM trainieren |
| Validierung (10 %) | 275 | 10.08.2022–04.09.2023 | GARCH-Ordnung und LSTM-Epochenzahl auswählen |
| Abschließende Bewertung (20 %) | 550 | 05.09.2023–31.10.2025 | Prognosefehler und VaR-Backtests berechnen |

**Warum das Refit auf 80 %?** Nach der Auswahl stehen GARCH-Ordnung und LSTM-Epochenzahl fest. Training und Validierung werden zusammengeführt, um die Parameter anhand aller vor Beginn der Bewertung verfügbaren Daten neu zu schätzen. Die letzten 20 % werden dafür nicht verwendet. Der Validierungsabschnitt ist nach diesem Refit keine unabhängige Validierung mehr.

### Modelle und Prognosen

- **GARCH:** Neun Kombinationen mit `p,q = 1,2,3`; Auswahl über den niedrigsten Validierungs-RMSE gegenüber absoluten Renditen. AIC/BIC dienen als ergänzende Trainingsdiagnosen. Im dokumentierten Lauf wird GARCH(1,1) gewählt. Während der Bewertung erfolgt vor jeder Tagesprognose eine Neuschätzung auf der wachsenden Historie (Expanding Window); die Modellordnung bleibt fest.
- **LSTM:** 120 vergangene absolute Renditen als Eingabe, 25 LSTM-Einheiten, lineare Ausgabe, Adam und MSE. Early Stopping bestimmt die Epochenzahl anhand der 10 % Validierung. Ein neues Netz wird anschließend mit fester Epochenzahl auf 80 % trainiert. Während der Bewertung wandert das Eingabefenster weiter; die Gewichte bleiben unverändert.
- **Vorverarbeitung:** Skalierer werden ausschließlich auf den jeweiligen Trainingsdaten angepasst. Für Validierung und Bewertung werden vergangene Eingangswerte über Abschnittsgrenzen übernommen, sodass keine Zieltage durch den 120-Tage-Vorlauf entfallen.

### Value at Risk

Der Ein-Tages-VaR bei 95 % wird als unteres 5-%-Quantil der Log-Renditeverteilung dargestellt. Eine Überschreitung liegt vor, wenn die tatsächliche Rendite diese Schwelle unterschreitet. Die dargestellte Schwelle ist typischerweise negativ und in Prozent angegeben, kein positiver Euroverlust.

- **GARCH:** Mittelwert, Standardabweichung und Freiheitsgrade der jeweiligen Tagesprognose; korrekt auf Varianz eins standardisiertes Student-t-Quantil.
- **LSTM:** Positive Prognose der absoluten Rendite als Skala; zusätzliche Student-t-Kalibrierung mit Lage-, Skalen- und Freiheitsgradparametern anhand der ersten 80 %.
- **Backtests:** Kupiec, Christoffersen-Unabhängigkeit und Conditional Coverage; Log-Likelihood-Berechnung mit expliziter Behandlung von Randfällen.

## Grenzen

- Absolute Renditen sind ein verrauschter Proxy für Schwankungsstärke. Das LSTM lernt diese Größe direkt, während GARCH eine bedingte Standardabweichung liefert. Kleinere MAE-/RMSE-Werte gegenüber absoluten Renditen beweisen deshalb allein keine überlegene Volatilitätsprognose.
- Die Datenbasis und die Zahl der Marktphasen sind begrenzt. Überlappende LSTM-Fenster sind keine unabhängigen Beobachtungen. Die Lernkurve allein beweist weder Datenmangel noch Underfitting.
- GARCH und LSTM verwenden unterschiedliche Aktualisierungsstrategien. Beim LSTM erfolgt die zusätzliche t-Kalibrierung anhand von In-Sample-Prognosen.
- Der Bewertungszeitraum wurde bereits in der ursprünglichen Seminarfassung betrachtet. Er ist im überarbeiteten Code von der Auswahl getrennt, aber kein vollständig neuer Holdout.
- Nichtverwerfen bestätigt keine Modellkorrektheit. Die verwendeten Tests sind keine direkten Überlegenheitstests und bewerten Verluste jenseits der Schwelle nicht umfassend.

## Projektdateien

| Datei | Inhalt |
|---|---|
| [dax_garch_lstm.ipynb](dax_garch_lstm.ipynb) | Analyse, Code, Grafiken und Interpretation |
| [data/DAX_Data.csv](data/DAX_Data.csv) | Ursprünglicher lokaler Datenexport |
| [requirements.txt](requirements.txt) | Direkte Laufzeitabhängigkeiten mit Versionsangaben |
| [README.md](README.md) | Projektüberblick und Einrichtung |

Das Notebook kann `DAX_Data.csv` alternativ direkt neben der Notebookdatei lesen. Groß-/Kleinschreibung des Dateinamens beachten. Bei einem anderen Notebooknamen den Link in dieser README entsprechend anpassen.

## Einrichtung in VS Code

Das Projekt wird mit **Python 3.9** ausgeführt; der ursprüngliche Seminarstand verwendete Python 3.9.6. Diese Version wurde ursprünglich gewählt, weil die damalige Refinitiv-Anbindung in der Projektumgebung nicht mit Python 3.11 funktionierte. Für die aktuelle CSV-basierte Analyse wird das Refinitiv-Paket nicht mehr benötigt. Die Einrichtung unten bleibt bei Python 3.9. Der vollständige Ablauf mit der bereinigten Paketliste muss vor Veröffentlichung noch in einer frischen Umgebung geprüft werden. Eine GPU ist nicht erforderlich.


### 1. Virtuelle Umgebung erstellen

Im Projektordner unter macOS/Linux:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Unter Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 2. Notebook ausführen

1. Projektordner in VS Code öffnen; Python- und Jupyter-Erweiterungen verwenden.
2. Das Notebook öffnen und unter **Select Kernel** die `.venv` auswählen.
3. Prüfen, dass die CSV unter dem oben beschriebenen Pfad liegt.
4. Kernel neu starten und **Run All** ausführen.
5. Notebook mit den aktuellen Ausgaben speichern und die Ergebnistabelle dieser README abgleichen.

Die GARCH-Auswahl umfasst bis zu neun Kandidaten mit je 275 Neuschätzungen plus 550 Neuschätzungen für die abschließende Bewertung. Das LSTM trainiert zunächst zur Epochenwahl und anschließend erneut für das finale Refit. Die Laufzeit hängt von der Hardware ab.

### Plattform und Reproduzierbarkeit

`requirements.txt` verwendet TensorFlow 2.20.0. Für Intel-macOS wählt ein Plattformmarker stattdessen 2.16.2, da für 2.20.0 keine Intel-macOS-Wheels vorliegen. Das ist ein Kompatibilitätspfad, keine Garantie identischer Resultate. Der Intel-Mac-Pfad wurde hier nicht vollständig ausgeführt. Quellen: [TensorFlow 2.20.0](https://pypi.org/project/tensorflow/2.20.0/) und [TensorFlow 2.16.2](https://pypi.org/project/tensorflow/2.16.2/).

Seeds sind im Notebook gesetzt. Ergebnisse können dennoch mit Hardware, TensorFlow-/Keras-Version oder numerischen Bibliotheken variieren. Die Paketliste fixiert direkte Abhängigkeiten; sie ist kein vollständiger Lockfile aller indirekten Pakete.

## Methodische Grundlagen

Literatur und methodische Erläuterungen stehen im Notebook. Ergänzende technische Dokumentation:

- [arch: Volatility und VaR Forecasting](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html)
- [Keras: EarlyStopping](https://keras.io/api/callbacks/early_stopping/)
- [SciPy: xlogy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.xlogy.html)
