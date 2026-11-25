# ADR-2: Wybór modelu produkcyjnego i jego postać jako artefaktu

- **Status:** proponowany
- **Sprint:** S3
- **Autor:** Igor Bukowski
- **Recenzent:** _do uzupełnienia (inna osoba niż autor)_

## Kontekst

W Sprincie 3 mamy trzech kandydatów do dalszej części systemu: baseline ze Sprintu 1–2
(sklearn Pipeline z `LogisticRegression`) oraz dwie konfiguracje AutoGluona uruchamiane
w pipeline'ie `autogluon` (`medium_quality` z limitem 60 s i `good_quality` z limitem 120 s).
`DummyClassifier` jest tylko punktem odniesienia.

W Sprincie 4 model ma obsługiwać pojedyncze rekordy przez API, a w Sprincie 6 działać
w chmurze na małej instancji. Dlatego oprócz F1 liczą się: czas predykcji jednego wiersza,
rozmiar artefaktu i pamięć po wczytaniu.

Ograniczenia:

- dane Iranian Churn: 3 150 rekordów, zbiór testowy 570 rekordów, w tym tylko 89 z klasą
  pozytywną `1` — różnica F1 rzędu 0,02 to kilka poprawnie wykrytych klientów, czyli wynik
  na granicy szumu;
- AutoGluon z limitem czasu nie jest w pełni deterministyczny, więc wybór musi wynikać
  z jawnej reguły, którą da się powtórzyć, a nie z jednorazowego porównania „na oko”;
- pomiary z pipeline'u `autogluon` dotyczą katalogu treningowego (wszystkie modele
  i `tracemalloc`, który nie widzi pamięci natywnej LightGBM/CatBoost). Do decyzji
  potrzebny jest pomiar tego, co faktycznie trafi do wdrożenia.

## Rozważone opcje

1. **Baseline `logistic_regression`** — najmniejszy i najszybszy, ale niski recall.
2. **AutoGluon `medium_quality_60s`** — najlepszy model to `WeightedEnsemble_L2`
   (ensemble kilku modeli L1).
3. **AutoGluon `good_quality_120s`** — najlepszy model to `LightGBM_BAG_L1_FULL`
   (jeden LightGBM douczony na całym zbiorze treningowym po baggingu).

Postać artefaktu:

- **A.** katalog treningowy AutoGluona w całości;
- **B.** pakiet modelu z `model_package.py`: `model/` (dla AutoGluona
  `clone_for_deployment` — tylko modele potrzebne do predykcji) + `metadata.json`,
  wysłany do W&B jako Artifact `churn-model` z aliasem `production`.

## Jak mierzyliśmy

Pipeline `kedro run --pipeline model_selection`:

1. każdy kandydat jest zapisywany funkcją `save_package` (ta sama co dla `production`);
2. pomiar odbywa się w **świeżym procesie** Pythona (`asi_project.benchmark`):
   `load_package` → rozgrzewka → 200 predykcji pojedynczego wiersza (p50/p95) →
   predykcja całego zbioru testowego;
3. pamięć = przyrost RSS procesu po wczytaniu i pierwszej predykcji (obejmuje import
   bibliotek modelu — to realny koszt startu API);
4. pakiet musi po wczytaniu odtworzyć F1 z treningu (`package_reproduces_metrics`);
5. wyniki: `data/08_reporting/model_comparison.csv`, `model_benchmark.json`,
   run W&B `model-selection-full` (tabela `model_comparison`).

Reguła wyboru (`conf/base/parameters.yml`, `model_selection`):

1. odrzucamy kandydatów łamiących limity wdrożeniowe: p95 predykcji 1 wiersza ≤ 100 ms,
   pakiet ≤ 100 MB, pamięć ≤ 1024 MB;
2. shortlista: F1 ≥ najlepsze F1 − 0,02;
3. z shortlisty wygrywa kandydat najtańszy we wdrożeniu: p95 → rozmiar → pamięć.

## Wyniki

Pełne dane (`data_mode: full`), split `stratified_random`, `test_size=0.2`,
`random_state=42`, n_test = 570. Fingerprint danych `90d5fb6b…f7dbe1`. Kod: commit `b0c27c7`.

| Kandydat | Model | F1 | Precision | Recall | p50 1 wiersz [ms] | p95 1 wiersz [ms] | Batch [ms/wiersz] | Pakiet [MB] | Pamięć [MB] | Trening [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| **AG `good_quality_120s`** | `LightGBM_BAG_L1_FULL` | **0,888** | 0,888 | 0,888 | 11,6 | 14,9 | 0,022 | 0,61 | 36,9 | 124,0 |
| AG `medium_quality_60s` | `WeightedEnsemble_L2` | 0,864 | 0,874 | 0,854 | 33,2 | 85,1 | 0,081 | 2,17 | 37,2 | 19,5 |
| `logistic_regression` | `LogisticRegression` | 0,600 | 0,824 | 0,472 | 6,8 | 8,3 | 0,013 | 0,006 | 6,7 | — |
| `dummy` (odniesienie) | `DummyClassifier` | 0,000 | 0,000 | 0,000 | 0,4 | 0,7 | 0,001 | 0,002 | 0,2 | — |

Macierze pomyłek (tn / fp / fn / tp): AG good 471 / 10 / 10 / 79,
AG medium 470 / 11 / 13 / 76, LR 472 / 9 / 47 / 42.

Runy W&B (projekt `MIJA-asi-iranian-churn`, 25.11.2026, Windows 11, Python 3.11):

| Run | Co zawiera |
|---|---|
| [`baseline-full`](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/MIJA-asi-iranian-churn/runs/jfv1d1t4) | metryki `logistic_regression` i `dummy` |
| [`autogluon-medium_quality_60s-full`](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/MIJA-asi-iranian-churn/runs/npfutzi8) | eksperyment AG medium |
| [`autogluon-good_quality_120s-full`](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/MIJA-asi-iranian-churn/runs/7o4h1eo9) | eksperyment AG good |
| [`model-selection-full`](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/MIJA-asi-iranian-churn/runs/lcsmwqut) | tabela `model_comparison`, wybrany model |
| [`vivid-snow-14`](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/MIJA-asi-iranian-churn/runs/9gdqy7p2) | upload pakietu `churn-model:v4`, alias `production` |

Czasy predykcji zależą od maszyny: na drugim komputerze (Linux) p95 wyniosło 12,7 ms
(good) i 32,1 ms (medium). Ranking i decyzja były takie same; F1 identyczne.

## Decyzja

**Model produkcyjny: AutoGluon `good_quality_120s` (`LightGBM_BAG_L1_FULL`),
przekazywany jako pakiet modelu (opcja B) — W&B Artifact `churn-model:v4`, alias
`production`.** Smoke po pobraniu z W&B: `SMOKE OK`.

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| F1 wybranego modelu vs baseline | 0,888 vs 0,600 | runy `autogluon-good_quality_120s-full`, `baseline-full` |
| Recall (wykryci odchodzący klienci) | 79 / 89 vs 42 / 89 | macierz pomyłek, `model_benchmark.json` |
| p95 predykcji 1 wiersza vs AG medium | 14,9 ms vs 85,1 ms | `asi_project.benchmark`, run `model-selection-full` |
| Rozmiar pakietu vs AG medium | 0,61 MB vs 2,17 MB | rozmiar katalogu pakietu |

Uzasadnienie:

- Wybrany model ma najwyższe F1 i **jednocześnie** jest tańszy we wdrożeniu od drugiej
  konfiguracji AutoGluona: ok. 3× szybszy w medianie i ok. 6× w p95 dla pojedynczego
  wiersza, ok. 3,5× mniejszy, przy podobnej pamięci (ok. 37 MB, głównie import
  AutoGluona). To pojedynczy LightGBM, a nie ensemble, więc w API nie ma łańcucha kilku
  modeli do wczytania.
- Decyzja nie zależy od tolerancji F1: przy tolerancji 0 wygrywa najlepsze F1, a przy
  tolerancji obejmującej oba warianty AutoGluona (np. 0,03) wygrywa tańszy we wdrożeniu —
  w obu przypadkach ten sam model. Dlatego różnica F1 0,024 będąca na granicy szumu nie
  przesądza o wyborze.
- Wszystkie trzy modele spełniają limity wdrożeniowe, ale AG medium ma p95 85 ms przy
  limicie 100 ms — mały zapas na obciążonej maszynie. Wybrany model ma ok. 85 ms zapasu.

## Odrzucone alternatywy

- **`logistic_regression`** — najszybszy (p95 8,3 ms) i najmniejszy, ale F1 0,600 i recall
  0,47: przegapia ponad połowę odchodzących klientów (47 z 89). Różnica 0,29 F1 to dużo
  więcej niż tolerancja; zysk ok. 7 ms na predykcji nie ma znaczenia przy limicie 100 ms.
- **AutoGluon `medium_quality_60s`** — niższe F1 (0,864), a przy tym wolniejszy i większy,
  bo to ensemble kilku modeli. Jedyna zaleta to krótszy trening (20 s vs 124 s), który nie
  dotyczy API.
- **Postać A (cały katalog treningowy)** — 62 MB dla `good_quality_120s`
  (`autogluon_results.json`), w tym wszystkie modele z treningu, których API nie używa. `clone_for_deployment` zostawia tylko
  `LightGBM_BAG_L1_FULL`.

## Konsekwencje

- API w Sprincie 4 wczytuje model wyłącznie przez `load_package` i `predict`
  z `model_package.py` oraz pobiera `churn-model:production` z W&B — nie zna numeru
  eksperymentu ani sposobu treningu.
- Obraz API musi zawierać `autogluon.tabular[lightgbm]` w wersji z `metadata.json`
  (`library_versions`), co powiększa obraz względem samego scikit-learn. To świadomy koszt
  za +0,29 F1.
- Wybrany pakiet nie pozwala na dalszy trening (`clone_for_deployment`) — zmiana modelu
  oznacza ponowne uruchomienie `autogluon` → `model_selection` → `package_model.py`.
- Jeżeli po ponownym treningu reguła wskaże inny model, aktualizujemy ten ADR
  (sekcja „Historia”) i przenosimy alias `production`.

## Historia

| Data | Zmiana |
|---|---|
| 25.11.2026 | utworzenie |
