struktura projektu wyglądała następująco:
- src
  - data
    - test (folder z zbiorem testowym)
    - train (folder z zbiorem treningowym)
    - valid (folder z zbiorem walidacyjnym)
    - cards.csv
  - models
    - test_run.keras(moja początkowa sieć)
    - testrun (folder do tego procesu neuroewolucji)
      - gen1
        - test_model_0.keras
        - test_model_1.keras
        - ...
      - gen2
        - test_model_1_0.keras
        - test_model_1_1.keras
        - ...
      - ...
  - src
      - tests
        - klasa do testowania klasy Generation
      - main.py główne uruchomienie programu
      - misc.py (kod do losowych operacji, testowałem GPU)
      - train.py (kod do trenowania na tych konretnych danych)
      - validate.py (kod do walidacji z konkretnego zbioru danych)
      - plot_model (skrypt do obrazowania konkretnego modelu)
      - teacher.py (warianty uczenia)
      - test_and_plot.py (skrypt do dalszego uczenia dwóch wybranych sieci i porównania ich
      - generation.py (główna klasa projektu, niezależna specyfikacji)
      - innitial.py (skrypt do tworzenia pierwszego modelu)
      - find_layer_edits (skrypt do analizy nauki, automatycznie sprawdza czy dodawane były jakiekolwiek warstwy w trakcie uczenia)