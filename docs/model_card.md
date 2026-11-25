# Model Card

> Minimalna Model Card ze Sprintu 3; aktualizacja w Sprincie 6.
> Decyzja o wyborze modelu i pełna tabela porównawcza:
> [ADR-2](adr/0002-wybor-modelu-produkcyjnego.md).

## Model

- **Typ modelu:** AutoGluon Tabular 1.6.3, konfiguracja `good_quality_120s`
  (preset `good_quality`, limit czasu 120 s, `eval_metric=f1`). Model użyty do predykcji:
  `LightGBM_BAG_L1_FULL` — jeden LightGBM douczony na całym zbiorze treningowym po baggingu.
  Czas treningu konfiguracji: ok. 124 s.
- **Artefakt:** W&B Artifact `churn-model:v4`, alias `production`. Pakiet: katalog `model/`
  (`TabularPredictor` po `clone_for_deployment`) + `metadata.json`.
  Odczyt wyłącznie przez `asi_project.model_package.load_package` i `predict`.
- **Utworzenie:** Sprint 3 (25.11.2026), pipeline'y `autogluon` → `model_selection`,
  pakiet: `scripts/package_model.py --selection … --alias production`.

## Przeznaczenie

- **Problem:** binary classification — czy klient operatora telekomunikacyjnego odejdzie
  (`Churn`). **Klasa pozytywna: `1`** (klient odszedł).
- **Moment predykcji:** koniec 9. miesiąca, na podstawie agregatów z miesięcy 1–9;
  etykieta opisuje stan klienta na koniec 12. miesiąca.
- **Użytkownik i decyzja:** dział utrzymania klienta — wybór klientów do działań
  retencyjnych (kontakt, oferta). Model wskazuje kandydatów; decyzję podejmuje człowiek.
- **Wyjście:** `prediction` (0/1) i `probability` (prawdopodobieństwo klasy `1`).
- **Nie jest przeznaczony do:** decyzji automatycznych bez udziału człowieka (np. zmiany
  warunków umowy), oceny zdolności kredytowej, klientów innych operatorów ani danych
  spoza zakresu zbioru treningowego.

## Dane

- **Dataset:** `iranian_churn` (UCI, CC BY 4.0, DOI 10.24432/C5JW3Z),
  plik `Customer Churn.csv`, 3 150 rekordów, klasa `1`: 15,7%.
- **Fingerprint (SHA-256):** `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.
- **Cechy (12, ALLOWED FEATURES):** `Call Failure`, `Subscription Length`, `Charge Amount`,
  `Seconds of Use`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers`, `Age`,
  `Complains`, `Age Group`, `Tariff Plan`, `Status`. `Customer Value` jest wykluczona
  zgodnie z kontraktem.
- **Split:** `stratified_random`, `test_size=0.2`, `random_state=42`;
  n_train = 2 280, n_test = 570 (89 rekordów klasy `1`).

## Metryki

Zbiór testowy (570 rekordów), próg decyzyjny domyślny AutoGluona.

| Model | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| **`production` (AG `LightGBM_BAG_L1_FULL`)** | **0,888** | 0,888 | 0,888 | 0,965 |
| Baseline `LogisticRegression` | 0,600 | 0,824 | 0,472 | 0,902 |
| `DummyClassifier` (najczęstsza klasa) | 0,000 | 0,000 | 0,000 | 0,844 |

- **Macierz pomyłek (production):** tn = 471, fp = 10, fn = 10, tp = 79.
- **Właściwości wdrożeniowe** (świeży proces, `asi_project.benchmark`, run W&B
  `model-selection-full`): p50/p95 predykcji jednego wiersza 11,6 / 14,9 ms, pakiet
  0,61 MB, pamięć po wczytaniu ok. 37 MB (z importem AutoGluona), czas wczytania 1,7 s.

## Ograniczenia i ryzyka

- **Mały zbiór testowy:** 89 przypadków pozytywnych — F1 ma dużą wariancję; różnice
  rzędu 0,02 między modelami to kilka rekordów.
- **Błędy:** model pomija ok. 11% odchodzących klientów (10 z 89) i oznacza ok. 2%
  zostających jako odchodzących (10 z 481). Przy kosztownej akcji retencyjnej warto w S4+
  rozważyć próg dobrany do kosztu błędów (obecnie domyślny).
- **Dane z jednego operatora i jednego okresu:** brak gwarancji przenoszenia na inny rynek
  lub późniejsze lata; potrzebny monitoring dryfu.
- **Cechy zakodowane liczbami** (`Charge Amount`, `Age Group`, `Tariff Plan`, `Status`):
  API musi przyjmować te same kody co w zbiorze, inaczej predykcja jest bez sensu.
- **Silna cecha `Complains`:** wśród klientów z reklamacją odchodzi 83%, bez reklamacji 10%
  (dane po przetworzeniu). Ponad połowa odchodzących (57%) nie składała reklamacji — to
  trudniejsza grupa, na której model może się mylić częściej.
- **Niedeterminizm treningu:** AutoGluon z limitem czasu może przy ponownym treningu
  wybrać inny model; dlatego wdrażamy konkretny artefakt (`production`), a nie „wynik
  treningu”.
- Dane zawierają wiek klienta; nie analizowaliśmy jakości modelu w podziale na grupy
  wiekowe (do zrobienia w S6).

## Wersje

- **Biblioteki (z `metadata.json`):** Python 3.11, numpy 2.2.6, pandas 2.3.3,
  scikit-learn 1.7.2, joblib 1.6.0, autogluon.tabular 1.6.3.
- **Git SHA treningu:** zapisany w `metadata.json` (`git_sha`) i w konfiguracji runów W&B;
  runy do ADR-2 wykonano na `b0c27c7`.
- **Runy W&B:** `baseline-full`, `autogluon-good_quality_120s-full`, `model-selection-full`
  (projekt `MIJA-asi-iranian-churn`).
