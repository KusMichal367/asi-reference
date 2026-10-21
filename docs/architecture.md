# Architektura systemu

Ten dokument opisuje architekturę systemu zespołu. W Sprincie 1 powstaje diagram v0, który
będzie aktualizowany w kolejnych sprintach.

## Architektura docelowa (scaffold)

Tak wygląda system po sześciu sprintach — na poziomie warstw. To mapa odpowiedzialności,
a nie diagram do skopiowania: diagram v0 (niżej) rysujecie sami.

Relacje między warstwami (opis tekstowy, nie diagram):

- training: dane → pipeline Kedro → artefakt modelu
- serving: UI → API → model (artefakt wczytany przez API)
- state: API → persistence
- deployment: API + UI → Cloud Run

| Warstwa | Odpowiedzialność | Sprint | Miejsce w repozytorium |
|---|---|---|---|
| Dane i kontrakt | źródło danych, kontrakt, input normalization, training-only filtering | S1 | `data/01_raw/`, `src/asi_project/datasets.yaml`, `src/asi_project/data.py` |
| Pipeline | odtwarzalne przetwarzanie i trening od surowych danych | S2 | `conf/`, `src/asi_project/pipelines/` |
| Eksperymenty i artefakt | porównanie runów, wybór modelu, pakiet modelu ładowalny poza treningiem | S3 | W&B, pakiet modelu, `docs/model_card.md` |
| API | kontrakt usługi, walidacja na granicy, model wczytywany raz | S4 | `src/asi_project/api/`, `docker/` |
| Persistence | zapis predykcji poza procesem aplikacji (tryby `sql` / `disabled`) | S4 → S5 → S6 | repository w API, Postgres w compose |
| UI | formularz jako klient API | S5 | `ui/`, `docker/`, `docker-compose.yml` |
| Chmura | wdrożenie, sekret, logi, release | S6 | `deploy/`, Cloud Run, Secret Manager, Cloud Logging |

## Diagram v0 (Sprint 1)

Narysujcie, jak — według waszej obecnej wiedzy — będzie wyglądał system na koniec projektu.
Diagram v0 nie musi być poprawny w szczegółach; ma pokazać wasze rozumienie przepływu.
Sama lista warstw z sekcji wyżej nie wystarcza — diagram pokazuje wasz system i wasze decyzje.

Na diagramie zaznaczcie co najmniej:

- skąd pochodzą dane i gdzie żyje kontrakt danych;
- gdzie działa każdy z trzech typów przetwarzania: input normalization, training-only
  filtering, model preprocessing;
- gdzie powstaje model i jak trafia tam, gdzie wykonuje się predykcja;
- kto lub co wysyła żądanie predykcji i co dostaje w odpowiedzi;
- gdzie przechowywany jest stan (np. zapis predykcji);
- które elementy już istnieją po Sprincie 1, a które są planowane.

Diagram umieśćcie w bloku `mermaid` w sekcji poniżej (GitHub renderuje Mermaid w Markdown)
albo jako obraz w `docs/` z linkiem w tej sekcji. Usuńcie znaczniki TODO, gdy diagram będzie
gotowy — sprawdza to `python scripts/check_dod.py`.

<!-- DIAGRAM-V0:START -->
```mermaid
flowchart TD
    subgraph DATA["Dane i kontrakt (S1, gotowe)"]
        SRC["UCI Iranian Churn, licencja CC BY 4.0<br/>scripts/download_data.py -> data/01_raw/iranian_churn/<br/>poza Gitem, SHA-256 w karcie projektu"]
        CONTRACT["Kontrakt danych<br/>datasets.yaml (stały kontrakt kursu) + project.yaml (decyzje zespołu)"]
        SAMPLE["data/01_raw/sample.csv<br/>próbka CI"]
    end

    subgraph PREP["src/asi_project/data.py (S1, gotowe)"]
        NORM["1. input normalization<br/>normalize_input(), inference-safe"]
        FILT["2. training-only filtering<br/>filter_training_rows()"]
    end

    subgraph TRAIN["Trening (S1 notebook 02 gotowe, S2 kedro run planowane)"]
        SPLIT["split train/test<br/>stratified_random"]
        PIPE["3. model preprocessing w sklearn Pipeline<br/>imputacja, skalowanie, one-hot<br/>+ LogisticRegression (model.py)"]
        EVAL["porównanie z DummyClassifier<br/>-> metrics.json"]
    end

    ART["Pakiet modelu + metadata.json<br/>W&B, Model Card (S3, planowane)"]

    subgraph SERVE["Serving (S4 i S5, planowane)"]
        UI["UI Streamlit<br/>użytkownik systemu"]
        API["API z tokenem<br/>normalize_input() -> Pipeline.predict_proba()"]
    end

    DB[("Postgres, zapis predykcji<br/>(S4 i S5, planowane)")]
    CLOUD["Docker, docker compose, Cloud Run,<br/>Secret Manager, logi (S4 do S6, planowane)"]

    SRC --> NORM
    CONTRACT -.-> NORM
    SAMPLE -. "pytest / CI" .-> NORM
    NORM --> FILT --> SPLIT --> PIPE --> EVAL
    PIPE --> ART
    ART -- "model wczytany raz przy starcie" --> API
    UI -- "dane jednego klienta" --> API
    API -- "prawdopodobieństwo odejścia (Churn = 1)" --> UI
    API --> DB
    API -.-> CLOUD
    UI -.-> CLOUD
```

Dane Iranian Churn pobieramy skryptem z UCI, a kontrakt danych (target Churn, klasa pozytywna 1, ALLOWED FEATURES, split stratified_random) jest zapisany w datasets.yaml i project.yaml. Trzy typy przetwarzania są rozdzielone: normalize_input() działa zarówno w treningu, jak i w API dla pojedynczego rekordu, filter_training_rows() działa tylko przy budowie zbioru treningowego, a model preprocessing (wszystko, co wymaga fit) jest wewnątrz sklearn Pipeline, więc zapisuje się razem z modelem. Po Sprincie 1 gotowe są dane, kontrakt, przygotowanie danych i baseline; artefakt modelu, API, UI, persistence i chmura powstaną w kolejnych sprintach. Użytkownik wysyła przez UI dane jednego klienta i dostaje prawdopodobieństwo jego odejścia, a API zapisuje predykcję w Postgresie.
<!-- DIAGRAM-V0:END -->

## Historia zmian architektury

| Sprint | Zmiana | Powiązany ADR |
|---|---|---|
| S1 | diagram v0 | — |
