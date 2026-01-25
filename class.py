import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных с правильной обработкой
def load_and_preprocess_data(file_path):
    """
    Загрузка и предобработка данных
    """
    df = pd.read_excel(file_path)
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {df.columns.tolist()}")
    print("\nПервая строка первой колонки:")
    print(df.iloc[0, 0])
    
    # Разделяем первую колонку на отдельные колонки
    first_column_name = df.columns[0]
    split_data = df[first_column_name].str.split(',', expand=True)
    
    # Определяем названия колонок для первой части
    # Из примера: id,text,datetime,text_length,word_count,hour,day_of_week,month,date,year
    first_part_columns = ['id', 'text', 'datetime', 'text_length', 'word_count', 
                         'hour', 'day_of_week', 'month', 'date', 'year']
    
    # Если split_data имеет меньше колонок, чем ожидается, берем то что есть
    if split_data.shape[1] >= len(first_part_columns):
        for i, col_name in enumerate(first_part_columns):
            df[col_name] = split_data[i]
    else:
        for i in range(split_data.shape[1]):
            df[f'col_{i}'] = split_data[i]
    
    # Удаляем старую колонку
    df = df.drop(columns=[first_column_name])
    
    # Проверяем, есть ли метки в данных
    # Возможно метки находятся в тексте или нужно их создать
    print("\nОбновленные колонки:")
    print(df.columns.tolist())
    
    return df

# 2. Подготовка данных (Пункт 3)
def prepare_data(df, target_column=None, text_column='text', 
                 test_size=0.2, val_size=0.1, random_state=42):
    """
    Разделение данных и кодирование меток
    """
    print("\nАнализ данных для поиска целевой переменной...")
    
    # Проверяем, есть ли у нас уже целевая переменная
    if target_column is None:
        # Пробуем найти возможные колонки с метками
        possible_label_cols = []
        for col in df.columns:
            unique_values = df[col].nunique()
            if 2 <= unique_values <= 50:  # Предполагаем, что меток от 2 до 50
                possible_label_cols.append((col, unique_values))
        
        print(f"Возможные колонки для целевой переменной:")
        for col, count in possible_label_cols:
            print(f"  {col}: {count} уникальных значений")
        
        # Если нашли подходящие колонки, используем первую
        if possible_label_cols:
            target_column = possible_label_cols[0][0]
            print(f"\nИспользуем '{target_column}' в качестве целевой переменной")
        else:
            # Создаем искусственную метку для демонстрации
            print("\nНе найдено подходящей колонки для целевой переменной")
            print("Создаю искусственную метку на основе длины текста...")
            if text_column in df.columns:
                df['label'] = pd.qcut(df[text_column].str.len(), q=3, labels=[0, 1, 2])
                target_column = 'label'
            else:
                # Используем первую векторную колонку для создания меток
                vector_cols = [col for col in df.columns if col.startswith('vector_')]
                if vector_cols:
                    df['label'] = pd.qcut(df[vector_cols[0]], q=3, labels=[0, 1, 2])
                    target_column = 'label'
    
    # Готовим признаки (X)
    # Если есть текстовая колонка, используем ее
    if text_column in df.columns:
        X = df[text_column].values
        is_vectorized = False
        print(f"Используем текстовую колонку: '{text_column}'")
    else:
        # Используем векторные признаки
        vector_cols = [col for col in df.columns if col.startswith('vector_')]
        if vector_cols:
            X = df[vector_cols].values
            is_vectorized = True
            print(f"Используем векторные признаки: {len(vector_cols)} признаков")
        else:
            # Используем все числовые колонки кроме целевой
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            X = df[numeric_cols].values
            is_vectorized = True
            print(f"Используем числовые признаки: {len(numeric_cols)} признаков")
    
    y = df[target_column].values
    
    # Кодирование меток, если они не числовые
    if not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Кодирование меток: {len(label_encoder.classes_)} классов")
        print(f"Классы: {label_encoder.classes_}")
    else:
        label_encoder = None
        y_encoded = y
        unique_classes = np.unique(y)
        print(f"Числовые метки: {len(unique_classes)} классов")
        print(f"Классы: {unique_classes}")
    
    # Разделение на train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, 
        random_state=random_state, stratify=y_encoded
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\nРазмеры выборок:")
    print(f"Обучающая: {len(X_train)}")
    print(f"Валидационная: {len(X_val)}")
    print(f"Тестовая: {len(X_test)}")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'label_encoder': label_encoder,
        'is_vectorized': is_vectorized,
        'target_column': target_column
    }

# 3. Класс Dataset для BERT (оставляем без изменений)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 4. Архитектура моделей (упрощаем для векторизованных данных)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class ClassicalModels:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        self.label_encoder = None
    
    def train(self, X_train, y_train):
        """Обучение всех классических моделей"""
        print("\nОбучение классических моделей...")
        for name, model in self.models.items():
            print(f"  Обучение {name}...")
            model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Оценка всех моделей"""
        print("\nОценка моделей на тестовой выборке:")
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            print(f"  {name}: Accuracy = {accuracy:.4f}")
        return results

# 5. Основная функция
def main():
    # Укажите путь к вашему файлу
    file_path = 'vectorized_dataset.xlsx'  # Замените на актуальный путь
    
    # Загрузка данных
    print("=" * 50)
    print("1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 50)
    df = load_and_preprocess_data(file_path)
    
    # Посмотрим на первые несколько строк
    print("\nПервые 3 строки данных:")
    print(df.head(3))
    
    # Подготовка данных (Пункт 3)
    print("\n" + "=" * 50)
    print("2. ПОДГОТОВКА ДАННЫХ")
    print("=" * 50)
    
    # Пробуем найти текстовую колонку
    text_column = None
    possible_text_cols = ['text', 'processed_text', 'col_1']
    for col in possible_text_cols:
        if col in df.columns:
            text_column = col
            break
    
    # Подготовка данных
    data_dict = prepare_data(
        df, 
        target_column=None,  # Автоматический поиск
        text_column=text_column,
        test_size=0.15,
        val_size=0.15
    )
    
    # Выбор варианта модели (Пункт 2)
    print("\n" + "=" * 50)
    print("3. ПРОЕКТИРОВАНИЕ АРХИТЕКТУРЫ МОДЕЛИ")
    print("=" * 50)
    
    if data_dict['is_vectorized']:
        print("Данные уже векторизованы!")
        print("Рекомендуемый вариант: Классические модели (Logistic Regression, SVM, Random Forest)")
        
        # Используем классические модели
        classical_models = ClassicalModels()
        classical_models.label_encoder = data_dict['label_encoder']
        
        # Обучение и оценка
        classical_models.train(data_dict['X_train'], data_dict['y_train'])
        results = classical_models.evaluate(data_dict['X_test'], data_dict['y_test'])
        
        # Сохраняем результаты
        print("\nЛучшая модель по точности:")
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"{best_model[0]}: Accuracy = {best_model[1]['accuracy']:.4f}")
        
    else:
        print("Данные содержат тексты!")
        print("Доступные варианты:")
        print("А. Классические модели + TF-IDF")
        print("Б. Нейросетевая модель (LSTM/GRU/CNN)")
        print("В. Тонкая настройка BERT (предпочтительный)")
        
        # Для текстовых данных можно использовать BERT
        num_labels = len(np.unique(data_dict['y_train']))
        
        print(f"\nЗапуск BERT для {num_labels} классов...")
        print("Внимание: Для использования BERT нужны текстовые данные")
        
        # Проверяем, есть ли тексты
        if isinstance(data_dict['X_train'][0], str):
            print("Текстовые данные найдены, можно использовать BERT")
        else:
            print("Текстовые данные не найдены в правильном формате")
    
    # Сохранение подготовленных данных
    print("\n" + "=" * 50)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    # Сохраняем подготовленные данные
    np.savez('prepared_data.npz',
             X_train=data_dict['X_train'],
             X_val=data_dict['X_val'],
             X_test=data_dict['X_test'],
             y_train=data_dict['y_train'],
             y_val=data_dict['y_val'],
             y_test=data_dict['y_test'])
    
    print("Данные сохранены в файл: prepared_data.npz")
    
    return data_dict, df

# Запуск
if __name__ == "__main__":
    data_dict, df = main()
    
    # Дополнительная информация о данных
    print("\n" + "=" * 50)
    print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
    print("=" * 50)
    
    print("\nТипы данных в DataFrame:")
    print(df.dtypes)
    
    print("\nКоличество уникальных значений в каждой колонке:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} уникальных значений")