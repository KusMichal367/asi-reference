# Architecture Decision Records

ADR dokumentuje jedną istotną decyzję architektoniczną: kontekst, rozważone opcje, wybór
i jego konsekwencje. W projekcie powstają dokładnie trzy ADR-y.

| ADR | Sprint | Temat |
|---|---|---|
| ADR-1 | S2 | Gdzie żyje przetwarzanie: input normalization, training-only filtering, model preprocessing |
| ADR-2 | S3 | Wybór modelu produkcyjnego i jego postać jako artefaktu |
| ADR-3 | S4 → S5 → S6 | Jak model trafia do usługi; stan i konfiguracja systemu; sekrety, cloud persistence i deployment |

ADR-3 powstaje stopniowo: szkic w Sprincie 4 (jak model trafia do usługi i obrazu),
uzupełnienie w Sprincie 5 (stan, persistence i konfiguracja w lokalnym systemie
wielousługowym), finalizacja w Sprincie 6 (sekrety, cloud persistence, deployment).

## Zasady

- Plik: `docs/adr/NNNN-krotki-tytul.md`, np. `docs/adr/0001-gdzie-zyje-przetwarzanie.md`.
  Szablon: [`template.md`](template.md).
- Autor i recenzent ADR to dwie różne osoby; obie są wpisane w dokumencie.
- ADR zawiera co najmniej dwie liczby z waszego systemu (np. metryki, czasy, rozmiary)
  i co najmniej jedną rozważoną, odrzuconą alternatywę z powodem odrzucenia.
- ADR trafia do repozytorium przez Pull Request jak każda inna zmiana.
- Zmiana decyzji to aktualizacja ADR (sekcja „Historia”) albo nowy ADR, który zastępuje stary.

## Kedro quickstart

Od Sprintu 2 trening uruchamia pipeline Kedro: surowe dane → walidacja → input normalization
→ training-only filtering → split → trening → ewaluacja. Notebook nie jest do tego potrzebny.

1. Środowisko (raz, w aktywnym `asi-ml`):

   ```bash
   conda activate asi-ml
   python -m pip install -r requirements/preflight-s2.txt -c constraints.txt
   python -m pip install --no-deps -e .
   kedro info
   ```

2. Dane (pełne dane nie są w repozytorium):

   ```bash
   python scripts/download_data.py
   ```

   Plik trafia do `data/01_raw/iranian_churn/`.

3. Uruchomienie:

   ```bash
   kedro run             # pełne dane (konfiguracja base + local)
   kedro run --env ci    # próbka data/01_raw/sample.csv (konfiguracja base + ci), tak jak w CI
   ```

4. Wyniki (poza gitem):

   | Plik | Zawartość |
   |---|---|
   | `data/02_intermediate/prepared_data.csv` | dane po pipeline'ie `data_processing` |
   | `data/06_models/trained_model.pkl` | sklearn Pipeline (model preprocessing + LogisticRegression) |
   | `data/06_models/dummy_model.pkl` | baseline DummyClassifier |
   | `data/08_reporting/metrics.json` | metryki obu modeli |

Konfiguracja: `conf/base/catalog.yml` (wejścia i wyjścia), `conf/base/parameters.yml`
(`test_size`, `random_state`). Testy nodów: `pytest tests/pipelines`. Decyzja o podziale
przetwarzania: [`docs/adr/0001-gdzie-zyje-przetwarzanie.md`](docs/adr/0001-gdzie-zyje-przetwarzanie.md).
