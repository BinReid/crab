# train_models_text.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import time
import os
import re
import string
from sklearn.preprocessing import StandardScaler

# Импорт моделей
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

def preprocess_text(text):
    """
    Простая предобработка текста
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Удаляем лишние пробелы
    text = ' '.join(text.split())
    
    # Можно добавить дополнительную обработку
    # Например, удаление пунктуации:
    # text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

def create_text_features(X_train_texts, X_test_texts, max_features=5000):
    """
    Создание признаков из текста с помощью TF-IDF
    """
    print("\nСоздание признаков из текстов...")
    print(f"  Обучающих текстов: {len(X_train_texts)}")
    print(f"  Тестовых текстов: {len(X_test_texts)}")
    
    # Предобработка текстов
    X_train_processed = [preprocess_text(text) for text in X_train_texts]
    X_test_processed = [preprocess_text(text) for text in X_test_texts]
    
    # Создаем TF-IDF векторайзер
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,  # слово должно встречаться минимум в 2 документах
        max_df=0.95,  # игнорируем слишком частые слова
        ngram_range=(1, 2),  # учитываем униграммы и биграммы
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b'
    )
    
    # Обучаем на тренировочных данных
    print("  Обучение TF-IDF векторайзера...")
    X_train_vectorized = vectorizer.fit_transform(X_train_processed)
    
    # Преобразуем тестовые данные
    X_test_vectorized = vectorizer.transform(X_test_processed)
    
    print(f"  Размерность признаков: {X_train_vectorized.shape[1]}")
    print(f"  Размер матрицы признаков: {X_train_vectorized.shape}")
    
    # Сохраняем векторайзер
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    return X_train_vectorized, X_test_vectorized, vectorizer

def train_and_evaluate_models(X_train, X_test, y_train, y_test, random_state=42):
    """
    Обучение и оценка различных моделей
    """
    print("\n" + "="*80)
    print("Настройка моделей:")
    print("="*80)
    
    # Для текстовых данных используем sparse матрицы
    use_dense = X_train.shape[1] < 1000  # Если мало признаков, преобразуем в dense
    
    if use_dense and hasattr(X_train, 'toarray'):
        print("Преобразование sparse матриц в dense...")
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    
    # 2. Список моделей для тестирования
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state,
            class_weight='balanced_subsample',
            n_jobs=-1
        ),
    }
    
    results = {}
    
    # 3. Обучение и оценка каждой модели
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Обучение модели: {name}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Оценка
            train_score = model.score(X_train, y_train)
            test_score = accuracy_score(y_test, y_pred)
            
            # Отчет по классификации
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Сохранение результатов
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'classification_report': report,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'training_time': time.time() - start_time,
                'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else 'sparse'
            }
            
            print(f"Оценка на обучении: {train_score:.4f}")
            print(f"Оценка на тесте: {test_score:.4f}")
            print(f"Время обучения: {results[name]['training_time']:.2f} секунд")
            print(f"Использовано признаков: {X_train.shape[1] if hasattr(X_train, 'shape') else 'sparse'}")
            
            # Вывод основных метрик
            print("\nОсновные метрики:")
            if 'accuracy' in report:
                print(f"  Accuracy: {report['accuracy']:.4f}")
            
            # Безопасный доступ к метрикам
            if 'macro avg' in report:
                print(f"  Precision (macro): {report['macro avg']['precision']:.4f}")
                print(f"  Recall (macro): {report['macro avg']['recall']:.4f}")
                print(f"  F1-Score (macro): {report['macro avg']['f1-score']:.4f}")
            
        except Exception as e:
            print(f"Ошибка при обучении {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    return results

def visualize_results(results, y_test, class_names=None):
    """
    Визуализация результатов всех моделей
    """
    # Фильтруем только успешные модели
    successful_models = {k: v for k, v in results.items() if v is not None}
    
    if not successful_models:
        print("Нет успешно обученных моделей для визуализации")
        return None, None
    
    model_names = list(successful_models.keys())
    train_scores = [successful_models[name]['train_score'] for name in model_names]
    test_scores = [successful_models[name]['test_score'] for name in model_names]
    
    # Создаем графики
    n_plots = min(4, len(model_names))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # График 1: Сравнение точности
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
    axes[0].bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
    axes[0].set_xlabel('Модели')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Сравнение точности моделей')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: F1-Score по моделям
    f1_scores = []
    for name in model_names:
        if 'macro avg' in successful_models[name]['classification_report']:
            f1 = successful_models[name]['classification_report']['macro avg']['f1-score']
        else:
            f1 = 0
        f1_scores.append(f1)
    
    axes[1].bar(model_names, f1_scores, alpha=0.8, color='green')
    axes[1].set_xlabel('Модели')
    axes[1].set_ylabel('F1-Score (macro)')
    axes[1].set_title('F1-Score по моделям')
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # График 3: Время обучения
    training_times = [successful_models[name]['training_time'] for name in model_names]
    
    axes[2].bar(model_names, training_times, alpha=0.8, color='orange')
    axes[2].set_xlabel('Модели')
    axes[2].set_ylabel('Время (секунды)')
    axes[2].set_title('Время обучения моделей')
    axes[2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    # График 4: Матрица ошибок для лучшей модели
    # Находим лучшую модель по F1-Score
    if f1_scores:
        best_idx = np.argmax(f1_scores)
        best_model_name = model_names[best_idx]
        
        best_result = successful_models[best_model_name]
        cm = confusion_matrix(y_test, best_result['y_pred'])
        
        # Ограничим размер матрицы для визуализации
        max_classes = 20
        if cm.shape[0] > max_classes:
            cm = cm[:max_classes, :max_classes]
            title_suffix = f" (первые {max_classes} классов)"
        else:
            title_suffix = ""
        
        # Нормализованная матрица ошибок
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Заменяем NaN на 0
        
        im = axes[3].imshow(cm_normalized, cmap='Blues', aspect='auto')
        axes[3].set_xlabel('Предсказанный класс')
        axes[3].set_ylabel('Истинный класс')
        axes[3].set_title(f'Матрица ошибок - {best_model_name}{title_suffix}')
        
        # Добавляем цветовую шкалу
        plt.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    
    # Сохранение таблицы с результатами
    results_df = pd.DataFrame({
        'Model': model_names,
        'Train_Accuracy': train_scores,
        'Test_Accuracy': test_scores,
        'F1_Score': f1_scores,
        'Training_Time': training_times,
        'Features_Used': [successful_models[name]['n_features'] for name in model_names]
    })
    
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    print("\n" + "="*80)
    print("Сводная таблица результатов:")
    print("="*80)
    print(results_df.to_string(index=False))
    
    best_model_name = results_df.iloc[0]['Model'] if len(results_df) > 0 else None
    
    return best_model_name, results_df

def save_best_model(results, best_model_name, vectorizer=None):
    """
    Сохранение лучшей модели и необходимых компонентов
    """
    if not best_model_name or best_model_name not in results or results[best_model_name] is None:
        print("Лучшая модель не найдена или не была успешно обучена")
        return
    
    # Создание директории для моделей
    os.makedirs('models', exist_ok=True)
    
    # Сохранение лучшей модели
    best_model = results[best_model_name]['model']
    
    # Используем joblib для лучшей совместимости
    model_filename = f'models/best_model_{best_model_name.replace(" ", "_").lower()}.joblib'
    joblib.dump(best_model, model_filename)
    
    # Сохраняем векторайзер если есть
    if vectorizer:
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    print(f"\n" + "="*80)
    print(f"Лучшая модель сохранена: {model_filename}")
    
    # Также сохраняем в pickle для совместимости
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Сохранение результатов в CSV
    best_result = results[best_model_name]
    metrics_df = pd.DataFrame({
        'metric': ['best_model', 'test_accuracy', 'f1_score_macro', 'train_accuracy'],
        'value': [
            best_model_name, 
            best_result['test_score'],
            best_result['classification_report']['macro avg']['f1-score'],
            best_result['train_score']
        ]
    })
    metrics_df.to_csv('models/model_performance.csv', index=False)
    
    # Сохраняем полный отчет
    report_df = pd.DataFrame(best_result['classification_report']).transpose()
    report_df.to_csv('models/classification_report.csv')
    
    print("Метрики модели сохранены в models/model_performance.csv")
    print("Полный отчет сохранен в models/classification_report.csv")
    print("="*80)

# Основной скрипт
if __name__ == "__main__":
    try:
        # Загрузка данных
        print("="*80)
        print("Загрузка текстовых данных...")
        print("="*80)
        
        data = np.load('split_data.npz', allow_pickle=True)
        
        # Загружаем разделенные данные
        X_train_texts = data['X_train']
        X_test_texts = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"Размеры загруженных данных:")
        print(f"  Обучающих текстов: {len(X_train_texts)}")
        print(f"  Тестовых текстов: {len(X_test_texts)}")
        print(f"  Меток обучения: {len(y_train)}")
        print(f"  Меток теста: {len(y_test)}")
        
        # Примеры текстов
        print(f"\nПримеры текстов:")
        print(f"  Обучающий текст 1: {X_train_texts[0][:50]}...")
        print(f"  Обучающий текст 2: {X_train_texts[1][:50]}...")
        print(f"  Тестовый текст 1: {X_test_texts[0][:50]}...")
        
        # Проверяем баланс классов
        print(f"\nОбщее количество классов: {len(np.unique(np.concatenate([y_train, y_test])))}")
        print(f"Классы в обучающей выборке: {len(np.unique(y_train))}")
        print(f"Классы в тестовой выборке: {len(np.unique(y_test))}")
        
        # Создаем признаки из текстов
        X_train_vectorized, X_test_vectorized, vectorizer = create_text_features(
            X_train_texts, X_test_texts, max_features=3000
        )
        
        # Обучение и оценка моделей
        print("\n" + "="*80)
        print("Начало обучения моделей")
        print("="*80)
        
        results = train_and_evaluate_models(X_train_vectorized, X_test_vectorized, y_train, y_test)
        
        # Визуализация результатов
        best_model_name, results_df = visualize_results(results, y_test)
        
        if best_model_name:
            # Сохранение лучшей модели
            save_best_model(results, best_model_name, vectorizer)
        else:
            print("Не удалось определить лучшую модель")
            
        # Сохранение полной таблицы результатов
        if results_df is not None:
            results_df.to_csv('models/all_models_results.csv', index=False)
            print("Полные результаты сохранены в models/all_models_results.csv")
        
        # Пример предсказания
        if best_model_name and best_model_name in results:
            print("\n" + "="*80)
            print("Пример предсказания:")
            print("="*80)
            
            test_texts = [
                "не получила код",
                "мне насчет посылок по выдаче",
                "где мой заказ",
                "почему не приходит смс"
            ]
            
            best_model = results[best_model_name]['model']
            
            for text in test_texts:
                processed = preprocess_text(text)
                vectorized = vectorizer.transform([processed])
                prediction = best_model.predict(vectorized)[0]
                print(f"Текст: '{text}' -> Предсказанный класс: {prediction}")
        
    except Exception as e:
        print(f"\nПроизошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()