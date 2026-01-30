# ML Project

W tym projekcie implementujemy pipeline pozyskujący featury z URL i trenujący model ML wykrywający próby phishing'u.

---

## Features

- **`features.py`** udostępnia listę funkcji `FEATURES`.  
  Każda funkcja przyjmuje `pd.Series` jako argument, a zwraca int.  
- Funkcje które chcemy udostępnić należy oznaczyć dekoratorem @feature 
- **Nie** należy dekorować funkcji pomocniczych.  

---

## Preprocessing

- **`preprocessing.py`** przyjmuje surowe dane i dodaje featury, aplikując funkcje z FEATURES.
- Wynikowy DataFrame jest zapisywany jako **`data_with_features.pkl`**. 

---

## Modeling

- **`model.ipynb`** trenuje i ewaluuje model.
- **Model**:
  - `model_train`: trenuje wybrany model (obecnie regersja logistyczna).  
  - `model_predict`: aplikuje model.  

- **Customowa ewaluacja**:
  - `cost`: liczy customową funkcję kosztu na podstawie macierzy pomyłek. 
    - Chcemy różnie karać FP i FN. 
  - `information_criterium`: Oblicza łączną karę za koszt i złożoność modelu
  - `eval`: trains the model on a specified subset of features and returns its cost.  

- **Wybieranie cech**:
  - `select_features` iteracyjnie usuwa najmniej informacyjną cechę tak długo jak poprawia to kryterium informacyjne.
  - Zwraca listę wybranych cech

---



