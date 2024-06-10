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
        - test_run_0.keras
        - test_run_1.keras
        - ...
      - gen2
        - test_run_1_0.keras
        - test_run_1_1.keras
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
      - test_and_plot.py (skrypt do dalszego uczenia dwóch wybranych sieci i porównania ich)
      - generation.py (główna klasa projektu, niezależna od specyfikacji)
      - innitial.py (skrypt do tworzenia pierwszego modelu)
      - find_layer_edits (skrypt do analizy nauki, automatycznie sprawdza czy dodawane były jakiekolwiek warstwy w trakcie uczenia)

Opis uczenia:
1. skrypt teacher.py przyjmuje na wejściu parametry:
  - ilość generacji (generations)
  - intensywność generacji (mutability)
  - ilość mutacji na generację (mutations)
  - ilość modeli przechodządych do dalszej generacji (candidates)
  - ścieżka modelu początkowego
  - ilość epok nauki na generację (epochs)
  - ścieżkę do folderu w którym odbywać się będzie nauka (folder główny)
2. tworzony jest folder generacji 1 w folderze głównym
3. kopjowany jest model początkowy do folderu generacji 1
4. tworzony jest obiekt klasy generacji z parametrami:
  - folder roboczy (folder generacji 1)
  - intensywność mutacji
  - ilość mutacji
  - ilość epok
  - numer generacji
Wykonać tyle razy ile ma być generacji:
5. mutowane są modele w folderze roboczym za pomocą metody mutate() przez teacher.py
metoda generation.mutate() wykorzystuje metody:
  - load_models_from_directory() - do załadowania modeli w jej folderze roboczym
  - replicate_model() - do stworzenia nowego modelu identycznego jak ten kopjowany
  - get_random_editable_layer_index() - zwraca indeks warsty której można zmienić ilość neuronów
  - change_nodes() - do edycji ilości neuronów w warstwie, mutacje mają 50% szansy na dodanie i 30% na odjęcie neuronów w ilości losowej od 1 do intensywności mutacji (mutability). Odejmowanie jest 2 zawsze połową wylosowanej wartości.
  - add_random_layer() - dodaje losowo warstwę CONV2D lub DENSE, w indeksie otrzymanym przez get_random_editable_layer_index() za pomocą metod:
    - add_dense_layer() - dodaje watstwę DENSE za pomocą metody add_layer()
    - add_conv_layer() - dodaje watstwę CONV2D i MAXPOOLING za pomocą metody add_layer()
6. trenowane są wszystkie modele w folderze roboczym za pomocą metody train_models() wykorzystującej skrypt train.py
7. wywoływana jest metoda create_successor() przez teacher.py
8. metoda create_successor() wywołuje metodę best_models()
  9. metoda best models wywołuje skrypt validate.py na każdym modelu w folderze roboczym
  10. validate.py zwraca wynik danej sieci: celność / log(ilość parametrów)
  11. metoda best models zwraca nejleprze modele w ilości równej parametrowi "candidates"
12. metoda create_successor() tworzy folder roboczy dla kolejnej generacji równoległy z obecnym
13. metoda create_successor() kopjuje najleprze modele do kolejnego folderu
14. metoda create_successor() tworzy nowy obiekt generacji i go zwraca.

