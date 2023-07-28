
# **Textklassifikation deutscher philosophischer Texte zur Zeit des deutschen Idealismus**

# Einleitung

Dieses Projekt zielt darauf ab, deutsche philosophische Texte aus dem 18. und 19. Jahrhundert als dem Deutschen Idealismus zugehörig oder nicht zugehörig zu klassifizieren. Das Modell ist eine Kombination aus einem bidirektionalen LSTM und einer Convolutional1D-Schicht mit Spatial Dropout und EarlyStopping Callbacks. Zusätzlich wurden Worteinbettungen (word2vec) auf dem Datensatz trainiert, um die Modellleistung zu verbessern.

# Beschreibung

Die Philosophie des 18. und 19. Jahrhunderts in Deutschland wird als eine der wichtigsten und einflussreichsten Perioden in der Geschichte der Philosophie bezeichnet (Dudley, 2014, p.1). Bedeutende Werke der deutschen Philosophiegeschichte wie Kants “Kritik der reinen Vernunft” prägen den deutschen Idealismus. Gleichzeitig ist der deutsche Idealismus berüchtigt für seine Komplexität.

Ziel dieses Projekt ist die Entwicklung eines Maschine-Learning-Modells, das erkennen soll, ob ein Text der Bewegung des deutschen Idealismus zuzuordnen ist oder nicht. Mögliche Anwendung findet diese Arbeit als Ressource für die Analyse und Kategorisierung philosophischer Textkorpora oder auch bei der Erstellung von Datenbanken.

## Datenvorverarbeitung

Die Daten wurden teilweise mit einem Webscraper extrahiert oder als ganze Textdatei heruntergeladen. Alle Werke sind gemeinfrei. Zur Vorverarbeitung werden sie in ein Pandas-Datenframe konvertiert. Der Text wird vorverarbeitet, indem HTML-Tags, Sonderzeichen und einzelne Zeichen entfernt werden. Stoppwörter werden entfernt, und deutsche Umlaute werden durch den entsprechenden Vokal ersetzt. Der Text wird außerdem mit dem SnowballStemmer aus NLTK gestemmt und dann in kleinere Abschnitte von maximal 200 Token Länge aufgeteilt.

Der finale Datensatz besteht sus 8080 Textsnippets, die in Trainings-, Validations- und Testsatz aufgeteilt wurden.

## Model

Das Modell besteht aus einem Bidirectional Long Short Term Memory (LSTM) in Kombination mit einer Convolutional1D-Schicht, Spatial Dropout und EarlyStopping-Callback. Zusätzlich wurden Word Embeddings (word2vec) auf den Datensatz trainiert, um die Leistung des Modells weiter zu verbessern. 
## Baseline

Die Baseline für dieses Projekt ist ein Naïve  Bayes-Modell. Der Code ist in der Datei "baseline.py" zu finden.

## Ordnerstruktur

Die Ordnerstruktur für dieses Projekt ist wie folgt:

```
|- data/
|  |- i/
|  |- ni/
|- baseline.py
|- model.py
|- scraper.py
```

- Das Verzeichnis "data" enthält zwei Unterverzeichnisse:
    - "i": enthält 136 deutsche philosophische Texte aus der Bewegung des Deutschen Idealismus.
    - "ni": enthält 136 deutsche philosophische wie politische Texte aus der gleichen Zeit
- "baseline.py" enthält den Code für das Naïve-Bayes-Baselinemodell.
- "model.py" enthält den Code für das endgültige Modell, das mit Word Embeddings trainiert wurde und aus einem bidirektionalen LSTM- und einer Convolutional1D-Schicht mit Spatial Dropout- und EarlyStopping-Callbacks besteht.
- "scraper.py" enthält den Code für den Webscraper. 

## **Dependencies**

- TensorFlow
- Keras
- NumPy
- Pandas
- NLTK
- Gensim

## **Training und Evaluation**

Das Modell wird mit Binary Cross-Entropy loss und Adam-Optimierer trainiert. Die Leistung wurde anhand der Metriken Precision, Recall, F1-Score und Accuracy bewertet. Das vorgeschlagene Modell erreichte eine Accuracy von 95%.

# Sources

- Dudley, W. (2014). *Understanding German idealism*. Routledge.