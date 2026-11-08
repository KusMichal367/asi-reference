# ADR-1: Gdzie żyje przetwarzanie danych

- **Status:** proponowany
- **Sprint:** S2
- **Autor:** Igor Bukowski
- **Recenzent:** TODO (inna osoba z zespołu)

## Kontekst

W Sprincie 2 przenosimy trening z notebooka do pipeline'u Kedro (`data_processing` →
`data_science`). Dane to Iranian Churn (binary classification, target `Churn`, split
`stratified_random`). W Sprincie 4 ten sam model ma obsługiwać pojedyncze rekordy przez API.

Mamy trzy rodzaje przetwarzania, które działają w różnych momentach:

- **input normalization** - ujednolicenie nazw kolumn i białych znaków w tekście; da się je
  wykonać na jednym nowym rekordzie;
- **training-only filtering** - usunięcie rekordów bez targetu i dokładnych duplikatów; ma sens
  tylko dla zbioru treningowego;
- **model preprocessing** - imputacja, skalowanie, one-hot encoding; wymaga `fit()` na danych
  treningowych.

Problem: każdy z tych kroków trzeba umieścić tak, żeby trening był reprodukowalny jedną komendą
(`kedro run`), żeby nie było wycieku danych między train i test, i żeby API w S4 mogło użyć tego
samego przetwarzania co trening, bez kopiowania kodu.

## Rozważone opcje

1. **Podział według rodzaju przetwarzania** - input normalization i training-only filtering
   jako funkcje w `data.py` wywoływane przez nody `data_processing`; model preprocessing
   wewnątrz sklearn `Pipeline` w `model.py`.
2. **Wszystko jako nody Kedro** - każdy krok, łącznie z imputacją i skalowaniem, jako osobny
   node przed splitem.
3. **Wszystko w sklearn `Pipeline`** - normalizacja, filtrowanie i preprocessing jako
   transformery jednego Pipeline.

## Decyzja

Wybieramy **opcję 1**:

| Rodzaj | Gdzie żyje | Node | Kiedy działa |
|---|---|---|---|
| Input normalization | `data.py`: `normalize_column_names`, `normalize_input` | `validate_data`, `normalize_data` | trening i predykcja (API w S4) |
| Training-only filtering | `data.py`: `filter_training_rows` | `filter_data` | tylko trening, przed splitem |
| Model preprocessing | `model.py`: `build_baseline_pipeline` (`ColumnTransformer` + `LogisticRegression`) | wewnątrz `train_models` | `fit` na X_train, potem razem z modelem |

Uzasadnienie:

- Input normalization jest jedną funkcją używaną w obu ścieżkach, więc rekord z API przejdzie
  przez dokładnie tę samą normalizację co dane treningowe.
- Training-only filtering działa przed splitem. Usuwa dokładne duplikaty, które inaczej
  mogłyby trafić jednocześnie do train i test i zawyżyć metryki.
- Model preprocessing jest dopasowywany tylko na X_train (brak wycieku statystyk z testu)
  i zapisuje się razem z modelem jako jeden artefakt (`data/06_models/trained_model.pkl`).

Liczby z naszego systemu:

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| Wiersze w surowych danych | 3150 | `data/01_raw/iranian_churn/Customer Churn.csv` |
| Wiersze usunięte przez training-only filtering | 300 (≈9,5%), wszystkie to dokładne duplikaty | node `filter_data` |
| Wiersze po filtrowaniu (train / test) | 2850 (2280 / 570) | `metrics.json`, split 0.2, `random_state=42` |
| F1 LogisticRegression vs Dummy | 0.60 vs 0.00 | `metrics.json` |

## Odrzucone alternatywy

- **Opcja 2 (model preprocessing jako nody Kedro przed splitem).** Imputer i scaler
  dopasowane na całych danych widziałyby statystyki zbioru testowego (data leakage). API
  musiałoby dodatkowo zapisywać i odtwarzać każdy transformer osobno, zamiast wczytać jeden
  model.
- **Opcja 3 (filtrowanie w sklearn `Pipeline`).** Transformery sklearn nie usuwają wierszy
  razem z `y`, a filtrowanie zostałoby wykonane także przy predykcji, gdzie nie ma targetu ani
  duplikatów do usunięcia. Filtrowanie po splicie zostawiłoby 300 duplikatów rozłożonych
  między train i test.
- **Osobna kopia normalizacji w API.** Dwie implementacje tej samej logiki z czasem się
  rozjadą; trening i predykcja dostawałyby różne dane.

## Konsekwencje

- Ułatwia: jeden artefakt modelu z preprocessingiem; brak wycieku statystyk; pipeline można
  testować node po nodzie (`tests/pipelines/`).
- Utrudnia: zmiana reguł filtrowania wymaga ponownego uruchomienia całego pipeline'u, a
  zmiana normalizacji wpływa jednocześnie na trening i API, więc wymaga ponownego treningu.
- S3: pakiet modelu zawiera cały sklearn `Pipeline`, a nie sam klasyfikator.
- S4: API importuje `normalize_input` z `data.py` i wczytuje cały Pipeline; nie wywołuje
  `filter_training_rows`.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-08 | utworzenie |
