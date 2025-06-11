# Dokumentacja techniczna systemu CatVTON do wirtualnego przymierzania odzieży

## Instalacja

Utworzenie źrodowiska Conda i instalacja zależności:
```shell
conda create -n catvton python==3.9.0
conda activate catvton
cd CatVTON-main  # or your path to CatVTON project dir
pip install -r requirements.txt
```

## Uruchomienie lokalnie (wymaga zainstalowanych sterowników CUDA 12.1)
```shell
uvicorn serving:app --host 0.0.0.0 --port 5000
```



## 1. Streszczenie
System CatVTON to rozwiązanie do generowania fotorealistycznych wizualizacji przymierzania odzieży w czasie rzeczywistym, wykorzystujące zaawansowane modele głębokiego uczenia. Dokument opisuje architekturę, interfejsy oraz mechanizmy działania systemu.

## 2. Architektura systemu

### 2.1 Diagram komponentów
```
[Klient] → [API Gateway] → [Moduł przetwarzania] → [Model CatVTON]
           ↑               ↑
[GCS] ← [Cache modeli]   [Secret Manager]
```

### 2.2 Specyfikacja techniczna
- **Język programowania**: Python 3.9
- **Frameworki**:
  - FastAPI (interfejs REST)
  - PyTorch 2.0 (przetwarzanie modeli)
  - Detectron2 (segmentacja obrazu)
- **Infrastruktura**:
  - Konteneryzacja: Docker
  - Orchestracja: Vertex AI
  - Przechowywanie danych: Google Cloud Storage

## 3. Interfejs API

### 3.1 Endpoint `/predict`
#### Żądanie:
```json
{
  "instances": [
    {
      "person_image_id": "string",
      "cloth_upper_image_id": "string|null",
      "cloth_lower_image_id": "string|null",
      "cloth_overall_image_id": "string|null",
      "inference_steps": "int=20"
    }
  ]
}
```

#### Wymagania:
- Co najmniej jeden parametr `cloth_*_image_id` musi być określony
- Wzajemnie wykluczające się: `cloth_overall_image_id` i pozostałe typy odzieży

## 4. Mechanizmy przetwarzania

### 4.1 Pipeline przetwarzania obrazu
1. Pobranie i deszyfracja obrazów wejściowych
2. Preprocessing:
   - Normalizacja rozmiaru (768×1024 px)
   - Transformacja przestrzeni barwnej (RGB)
3. Segmentacja:
   - Generowanie masek przez AutoMasker
   - Detekcja pozy ciała (DensePose)
4. Generowanie wyników:
   - Model dyfuzyjny z kontrolą uwagi
   - Postprocessing (normalizacja, wyostrzanie)

### 4.2 Optymalizacje wydajnościowe
- Wykorzystanie mixed precision (bfloat16)
- Memory-efficient attention
- Slicing tensorów VAE
- Dynamiczne zarządzanie pamięcią CUDA

## 5. Bezpieczeństwo danych

### 5.1 Szyfrowanie
- Algorytm: AES-256
- Klucze: przechowywane w Google Secret Manager
- Nonce: 16 bajtów + tag 16 bajtów

### 5.2 Izolacja środowiska
- Konteneryzacja z ograniczeniami dostępu
- Tymczasowe przechowywanie modeli (/tmp)
- Automatyczne czyszczenie pamięci GPU

## 6. Wnioski

System prezentuje efektywne połączenie:
1. Zaawansowanych modeli generatywnych
2. Optymalizacji obliczeniowych
3. Bezpiecznych mechanizmów przetwarzania danych
