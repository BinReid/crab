# predict_final_fixed.py
import pickle
import joblib
import numpy as np
import pandas as pd
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class IntentClassifierOptimized:
    def __init__(self, model_path, vectorizer_path, 
                 config_path='model_config_optimized.json',
                 class_mapping=None):
        """
        Инициализация оптимизированного классификатора
        
        Args:
            model_path: путь к модели
            vectorizer_path: путь к векторизатору
            config_path: путь к конфигурации
            class_mapping: словарь маппинга {исходный_класс: новый_класс}
        """
        # Загрузка модели
        self.model = joblib.load(model_path)
        self.original_classes = [str(c) for c in self.model.classes_]
        self.n_original_classes = len(self.original_classes)
        
        print(f"Исходных классов: {self.n_original_classes}")
        
        # Загрузка векторизатора
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Загружен векторизатор: {type(self.vectorizer).__name__}")
        
        # Создание или использование маппинга классов
        if class_mapping:
            self.class_mapping = class_mapping
        else:
            # Создаем маппинг на основе анализа текстов
            self.class_mapping = self._create_intelligent_mapping()
        
        # Создание новой конфигурации
        self._create_optimized_config(config_path)
        
        # Загрузка оптимизированной конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Порог уверенности
        self.confidence_threshold = self.config.get('threshold', 0.7)
        
        print(f"\nОптимизированных классов: {len(self.config['optimized_classes'])}")
        print(f"Классы: {self.config['optimized_classes']}")
        print(f"Порог уверенности: {self.confidence_threshold}")
    
    def _create_intelligent_mapping(self):
        """
        Создание интеллектуального маппинга на основе анализа классов
        """
        print("\nСоздание интеллектуального маппинга...")
        
        # Базовый маппинг на основе номеров классов
        # Можно расширить этот маппинг на основе анализа текстов
        class_mapping = {}
        
        # Группируем классы по категориям
        # 1. Проблемы с получением заказа (14, 12, 11, 17)
        problem_classes = ['14', '12', '11', '17', '10', '13']
        for cls in problem_classes:
            class_mapping[cls] = 'проблема_получения_заказа'
        
        # 2. Поиск/отслеживание заказа (15, 9, 20)
        tracking_classes = ['15', '9', '20', '19', '21']
        for cls in tracking_classes:
            class_mapping[cls] = 'поиск_отслеживание_заказа'
        
        # 3. Связь с оператором (23, 22, 24, 25)
        operator_classes = ['23', '22', '24', '25', '26']
        for cls in operator_classes:
            class_mapping[cls] = 'связь_с_оператором'
        
        # 4. Проблемы с постаматом/ячейкой (16, 18, 27, 28)
        postamat_classes = ['16', '18', '27', '28', '29']
        for cls in postamat_classes:
            class_mapping[cls] = 'проблема_с_постаматом'
        
        # 5. Изменение заказа/доставки (8, 7, 6)
        change_classes = ['8', '7', '6', '5', '4']
        for cls in change_classes:
            class_mapping[cls] = 'изменение_заказа'
        
        # 6. Общие вопросы (остальные классы)
        all_classes = set(self.original_classes)
        mapped_classes = set(class_mapping.keys())
        remaining_classes = all_classes - mapped_classes
        
        for cls in remaining_classes:
            class_mapping[cls] = 'общий_вопрос'
        
        # Дополняем на основе анализа текстов
        # (это можно расширить после анализа датасета)
        text_based_mapping = {
            '14': 'проблема_получения_заказа',  # не могу получить посылку
            '12': 'проблема_получения_заказа',  # не могу получить заказ
            '15': 'поиск_отслеживание_заказа',  # где находится моя посылка
            '23': 'связь_с_оператором',  # соедините с оператором
            '11': 'проблема_получения_заказа',  # не открывается ячейка
            '17': 'проблема_получения_заказа',  # не удается получить посылку
            '10': 'проблема_с_постаматом',  # не работает постамат
            '13': 'изменение_заказа',  # перенести срок доставки
            '16': 'проблема_с_постаматом',  # получить код подтверждения
            '18': 'проблема_с_постаматом',  # продлить срок хранения
        }
        
        # Обновляем маппинг
        for cls, new_cls in text_based_mapping.items():
            if cls in class_mapping:
                class_mapping[cls] = new_cls
        
        print(f"Создано правил маппинга: {len(class_mapping)}")
        
        return class_mapping
    
    def _create_optimized_config(self, config_path):
        """
        Создание оптимизированной конфигурации
        """
        # Уникальные новые классы
        new_classes = sorted(list(set(self.class_mapping.values())))
        
        print(f"\nМаппинг классов:")
        print(f"Исходных: {len(self.original_classes)}")
        print(f"Оптимизированных: {len(new_classes)}")
        
        # Создание конфигурации
        config = {
            'optimized_classes': new_classes,
            'threshold': 0.7,
            'top_n': 3,
            'original_classes': self.original_classes,
            'class_mapping': {str(k): str(v) for k, v in self.class_mapping.items()},
            'model_info': {
                'type': 'optimized_intent_classifier',
                'n_original_classes': self.n_original_classes,
                'n_optimized_classes': len(new_classes),
                'version': '2.0',
                'description': 'Оптимизированная модель классификации интентов'
            }
        }
        
        # Сохранение конфигурации
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\nКонфигурация сохранена: {config_path}")
        
        return config
    
    def _map_class(self, original_class):
        """
        Маппинг исходного класса в оптимизированный
        """
        original_str = str(original_class)
        return self.class_mapping.get(original_str, 'общий_вопрос')
    
    def predict_optimized(self, input_text, return_original=False):
        """
        Предсказание с оптимизированными классами
        """
        # Векторизация
        if isinstance(input_text, str):
            features = self.vectorizer.transform([input_text])
        else:
            features = self.vectorizer.transform(input_text)
        
        # Предсказание исходной моделью
        probabilities = self.model.predict_proba(features)[0]
        
        # Агрегация вероятностей по оптимизированным классам
        optimized_probs = {}
        
        for i, orig_class in enumerate(self.original_classes):
            prob = probabilities[i]
            opt_class = self._map_class(orig_class)
            
            if opt_class not in optimized_probs:
                optimized_probs[opt_class] = 0
            optimized_probs[opt_class] += prob
        
        # Преобразование в список для сортировки
        opt_classes = list(optimized_probs.keys())
        opt_probs = [optimized_probs[cls] for cls in opt_classes]
        
        # Получение топ-N
        top_n = min(self.config.get('top_n', 3), len(opt_classes))
        top_indices = np.argsort(opt_probs)[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = opt_classes[idx]
            confidence = float(opt_probs[idx])
            
            results.append({
                'intent': class_name,
                'confidence': confidence,
                'is_confident': confidence >= self.confidence_threshold
            })
        
        # Подготовка результата
        main_result = results[0]
        
        response = {
            'text': input_text if isinstance(input_text, str) else 'batch_input',
            'main_intent': main_result['intent'],
            'main_confidence': main_result['confidence'],
            'is_confident': main_result['is_confident'],
            'all_predictions': results,
            'optimized_probabilities': optimized_probs
        }
        
        # Если нужно вернуть исходные предсказания
        if return_original:
            # Находим лучший исходный класс
            best_original_idx = np.argmax(probabilities)
            best_original_class = self.original_classes[best_original_idx]
            best_original_prob = probabilities[best_original_idx]
            
            response['original_prediction'] = {
                'class': str(best_original_class),
                'confidence': float(best_original_prob)
            }
            
            # Топ-3 исходных классов
            top_original_indices = np.argsort(probabilities)[-3:][::-1]
            original_top = []
            for idx in top_original_indices:
                original_top.append({
                    'class': str(self.original_classes[idx]),
                    'confidence': float(probabilities[idx])
                })
            response['original_top3'] = original_top
        
        return response
    
    def batch_predict_optimized(self, texts_list):
        """
        Пакетное предсказание с оптимизированными классами
        """
        results = []
        for text in texts_list:
            result = self.predict_optimized(text)
            results.append(result)
        
        return results

def analyze_dataset_with_text_samples(data_path='split_data.npz', n_samples_per_class=3):
    """
    Анализ датасета с примерами текстов для каждого класса
    """
    print("="*60)
    print("АНАЛИЗ ДАТАСЕТА С ПРИМЕРАМИ ТЕКСТОВ")
    print("="*60)
    
    # Загрузка данных
    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    
    print(f"Всего примеров: {len(X_train)}")
    print(f"Всего классов: {len(set(y_train))}")
    
    # Анализ распределения классов
    class_counter = Counter(y_train)
    
    # Группировка текстов по классам
    class_texts = {}
    for text, cls in zip(X_train, y_train):
        cls_str = str(cls)
        if cls_str not in class_texts:
            class_texts[cls_str] = []
        class_texts[cls_str].append(text)
    
    # Анализ топ-N классов
    print(f"\nТоп-20 самых частых классов с примерами текстов:")
    print("-" * 60)
    
    class_analysis = []
    
    for i, (cls, count) in enumerate(class_counter.most_common(20), 1):
        percentage = count / len(X_train) * 100
        cls_str = str(cls)
        
        # Примеры текстов для этого класса
        examples = class_texts.get(cls_str, [])[:n_samples_per_class]
        
        print(f"\n{i:2}. Класс '{cls_str}': {count} примеров ({percentage:.1f}%)")
        for j, text in enumerate(examples, 1):
            print(f"     {j}. {text}")
        
        # Анализ ключевых слов
        all_texts = ' '.join([str(t).lower() for t in class_texts.get(cls_str, [])[:10]])
        
        # Определяем категорию на основе ключевых слов
        category = 'неизвестно'
        keywords = {
            'проблема_получения': ['не могу', 'не получается', 'не удается', 'не открывается'],
            'поиск_заказа': ['где', 'когда', 'отследить', 'находится'],
            'оператор': ['оператор', 'соедин', 'консультант', 'специалист'],
            'постамат': ['постамат', 'ячейк', 'код', 'штрих'],
            'изменение': ['перенести', 'изменить', 'адрес', 'срок'],
            'доставка': ['доставк', 'привез', 'придет', 'курьер'],
            'оплата': ['оплат', 'деньг', 'карт', 'стоимос'],
            'возврат': ['вернут', 'обмен', 'замен'],
        }
        
        for cat, words in keywords.items():
            if any(word in all_texts for word in words):
                category = cat
                break
        
        class_analysis.append({
            'class': cls_str,
            'count': count,
            'percentage': percentage,
            'examples': examples,
            'suggested_category': category
        })
        
        print(f"     Предлагаемая категория: {category}")
    
    # Группировка по предложенным категориям
    print(f"\n" + "="*60)
    print("ГРУППИРОВКА КЛАССОВ ПО КАТЕГОРИЯМ")
    print("="*60)
    
    category_groups = {}
    for item in class_analysis:
        category = item['suggested_category']
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(item)
    
    for category, items in category_groups.items():
        total_count = sum(item['count'] for item in items)
        total_percentage = sum(item['percentage'] for item in items)
        
        print(f"\nКатегория: {category}")
        print(f"Количество классов: {len(items)}")
        print(f"Примеров всего: {total_count} ({total_percentage:.1f}%)")
        print(f"Классы: {[item['class'] for item in items]}")
    
    # Создание рекомендуемого маппинга
    print(f"\n" + "="*60)
    print("РЕКОМЕНДУЕМЫЙ МАППИНГ")
    print("="*60)
    
    recommended_mapping = {}
    
    # Основные категории
    main_categories = {
        'проблема_получения_заказа': 'проблемы с получением',
        'поиск_отслеживание_заказа': 'поиск и отслеживание',
        'связь_с_оператором': 'связь с поддержкой',
        'проблема_с_постаматом': 'проблемы с постаматом',
        'изменение_заказа': 'изменение заказа',
        'уточнение_доставки': 'вопросы по доставке',
        'оплата_возврат': 'оплата и возвраты',
        'общий_вопрос': 'общие вопросы'
    }
    
    # Сопоставление категорий анализа с основными
    category_mapping = {
        'проблема_получения': 'проблема_получения_заказа',
        'поиск_заказа': 'поиск_отслеживание_заказа',
        'оператор': 'связь_с_оператором',
        'постамат': 'проблема_с_постаматом',
        'изменение': 'изменение_заказа',
        'доставка': 'уточнение_доставки',
        'оплата': 'оплата_возврат',
        'возврат': 'оплата_возврат',
        'неизвестно': 'общий_вопрос'
    }
    
    for item in class_analysis:
        analysis_category = item['suggested_category']
        main_category = category_mapping.get(analysis_category, 'общий_вопрос')
        recommended_mapping[item['class']] = main_category
    
    print(f"Создано {len(recommended_mapping)} правил маппинга")
    print(f"В {len(main_categories)} основных категорий")
    
    # Сохранение анализа
    analysis_results = {
        'total_samples': len(X_train),
        'total_classes': len(set(y_train)),
        'class_analysis': class_analysis,
        'category_groups': {k: [item['class'] for item in v] for k, v in category_groups.items()},
        'recommended_mapping': recommended_mapping,
        'main_categories': list(main_categories.keys())
    }
    
    # Преобразование для JSON (все ключи должны быть строками)
    json_safe_results = json.loads(json.dumps(analysis_results, default=str, ensure_ascii=False))
    
    with open('dataset_analysis_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(json_safe_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nАнализ сохранен в dataset_analysis_fixed.json")
    
    return recommended_mapping, main_categories

def create_final_classifier():
    """
    Создание финального классификатора с оптимальной группировкой
    """
    print("="*60)
    print("СОЗДАНИЕ ФИНАЛЬНОГО КЛАССИФИКАТОРА")
    print("="*60)
    
    # 1. Анализ датасета
    print("\n1. Анализ датасета...")
    recommended_mapping, main_categories = analyze_dataset_with_text_samples()
    
    # 2. Создание классификатора
    print("\n2. Создание классификатора...")
    
    classifier = IntentClassifierOptimized(
        model_path='models/best_model_random_forest.joblib',
        vectorizer_path='models/tfidf_vectorizer.joblib',
        config_path='model_config_final.json',
        class_mapping=recommended_mapping
    )
    
    # 3. Тестирование
    print("\n3. Тестирование классификатора...")
    
    test_phrases = [
        # Проблемы с получением
        "не могу получить посылку",
        "не открывается ячейка",
        "не получается забрать заказ",
        
        # Поиск и отслеживание
        "где находится моя посылка",
        "когда приедет заказ",
        "как отследить отправление",
        
        # Связь с поддержкой
        "соедините с оператором",
        "мне нужен консультант",
        "помогите решить проблему",
        
        # Проблемы с постаматом
        "не работает постамат",
        "не пришел код подтверждения",
        "нужно продлить срок хранения",
        
        # Изменение заказа
        "перенести срок доставки",
        "изменить адрес получения",
        "перевести посылку на другой адрес",
        
        # Общие вопросы
        "здравствуйте",
        "спасибо за помощь",
        "что делать если"
    ]
    
    print(f"\nТестируем {len(test_phrases)} фраз:\n")
    
    results_by_category = {}
    
    for phrase in test_phrases:
        result = classifier.predict_optimized(phrase, return_original=False)
        
        category = result['main_intent']
        if category not in results_by_category:
            results_by_category[category] = []
        
        results_by_category[category].append({
            'phrase': phrase,
            'confidence': result['main_confidence']
        })
        
        print(f"Фраза: '{phrase[:40]}...'")
        print(f"  Категория: {category}")
        print(f"  Уверенность: {result['main_confidence']:.3f}")
        
        if len(result['all_predictions']) > 1:
            print(f"  Альтернативы: {[p['intent'] for p in result['all_predictions'][1:3]]}")
        print()
    
    # 4. Статистика по категориям
    print("\n" + "="*60)
    print("СТАТИСТИКА ПО КАТЕГОРИЯМ")
    print("="*60)
    
    for category, phrases in results_by_category.items():
        avg_confidence = np.mean([p['confidence'] for p in phrases])
        print(f"\n{category}:")
        print(f"  Примеров: {len(phrases)}")
        print(f"  Средняя уверенность: {avg_confidence:.3f}")
        print(f"  Примеры фраз:")
        for p in phrases[:2]:
            print(f"    - '{p['phrase']}' ({p['confidence']:.3f})")
    
    # 5. Сохранение примеров использования
    usage_examples = {
        'classifier_info': {
            'original_classes': classifier.n_original_classes,
            'optimized_classes': len(classifier.config['optimized_classes']),
            'categories': classifier.config['optimized_classes']
        },
        'test_results': results_by_category,
        'sample_queries': {
            category: [p['phrase'] for p in phrases[:3]]
            for category, phrases in results_by_category.items()
        }
    }
    
    with open('classifier_usage_examples.json', 'w', encoding='utf-8') as f:
        json.dump(usage_examples, f, ensure_ascii=False, indent=2)
    
    print(f"\nПримеры использования сохранены в classifier_usage_examples.json")
    
    return classifier

def interactive_testing(classifier):
    """
    Интерактивное тестирование классификатора
    """
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ")
    print("="*60)
    print("Вводите фразы для классификации.")
    print("Команды: 'выход' - завершить, 'категории' - список категорий")
    print()
    
    test_history = []
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                print("Завершение работы...")
                break
                
            elif user_input.lower() == 'категории':
                print("\nДоступные категории:")
                for i, category in enumerate(classifier.config['optimized_classes'], 1):
                    print(f"  {i}. {category}")
                print()
                continue
            
            # Классификация
            print(f"\nАнализ фразы: '{user_input}'")
            result = classifier.predict_optimized(user_input, return_original=True)
            
            # Сохраняем в историю
            test_history.append(result)
            
            # Основной результат
            confidence_bar = "█" * int(result['main_confidence'] * 20)
            
            print(f"Основная категория: {result['main_intent']}")
            print(f"Уверенность: {result['main_confidence']:.3f} [{confidence_bar:<20}]")
            
            # Альтернативные варианты
            if len(result['all_predictions']) > 1:
                print(f"Дополнительные варианты:")
                for pred in result['all_predictions'][1:]:
                    print(f"  {pred['intent']}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

# Основной скрипт
if __name__ == "__main__":
    try:
        # Создание финального классификатора
        classifier = create_final_classifier()
        
        # Интерактивное тестирование
        interactive_testing(classifier)
        
        # Дополнительно: оценка на тестовых данных
        print("\n" + "="*60)
        print("ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")
        print("="*60)
        
        try:
            data = np.load('split_data.npz', allow_pickle=True)
            X_test = data['X_test']
            y_test = data['y_test']
            
            # Берем выборку для быстрой оценки
            sample_size = min(50, len(X_test))
            X_sample = X_test[:sample_size]
            y_sample = y_test[:sample_size]
            
            print(f"\nОцениваем на {sample_size} примерах...")
            
            # Предсказания
            predictions = []
            for text in X_sample:
                result = classifier.predict_optimized(text, return_original=False)
                predictions.append(result['main_intent'])
            
            # Маппинг истинных меток
            true_categories = []
            for true_label in y_sample:
                true_categories.append(classifier._map_class(true_label))
            
            # Расчет точности
            correct = sum(1 for p, t in zip(predictions, true_categories) if p == t)
            accuracy = correct / sample_size
            
            print(f"\nРезультаты оценки:")
            print(f"  Точность: {accuracy:.3f} ({correct}/{sample_size})")
            
            # Матрица ошибок (упрощенная)
            print(f"\nРаспределение предсказаний:")
            pred_counts = Counter(predictions)
            for category, count in pred_counts.most_common():
                print(f"  {category}: {count}")
            
        except Exception as e:
            print(f"Ошибка при оценке: {e}")
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

# Утилита для ручного создания маппинга
def create_manual_mapping():
    """
    Ручное создание маппинга классов
    """
    print("="*60)
    print("РУЧНОЕ СОЗДАНИЕ МАППИНГА")
    print("="*60)
    
    # Загружаем данные для анализа
    data = np.load('split_data.npz', allow_pickle=True)
    y_train = data['y_train']
    
    # Получаем список классов
    unique_classes = sorted(set(y_train))
    print(f"Всего классов: {len(unique_classes)}")
    
    # Определяем целевые категории
    print("\nОпределите целевые категории (рекомендуется 8-10):")
    
    categories = []
    while len(categories) < 10:
        category = input(f"Категория {len(categories)+1} (или Enter для завершения): ").strip()
        if not category:
            break
        categories.append(category)
    
    if not categories:
        categories = ['общий_вопрос']
    
    print(f"\nСоздано категорий: {len(categories)}")
    print("Категории:", categories)
    
    # Создаем маппинг
    mapping = {}
    
    for cls in unique_classes:
        cls_str = str(cls)
        print(f"\nКласс: {cls_str}")
        
        # Показываем категории
        for i, cat in enumerate(categories, 1):
            print(f"  {i}. {cat}")
        print(f"  {len(categories)+1}. другая категория")
        
        while True:
            try:
                choice = input(f"Выберите категорию (1-{len(categories)+1}): ").strip()
                if not choice:
                    choice = str(len(categories) + 1)
                
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(categories):
                    mapping[cls_str] = categories[choice_idx]
                    break
                elif choice_idx == len(categories):
                    # Запрос новой категории
                    new_cat = input("Введите название новой категории: ").strip()
                    if new_cat:
                        categories.append(new_cat)
                        mapping[cls_str] = new_cat
                        print(f"Добавлена новая категория: {new_cat}")
                    else:
                        mapping[cls_str] = 'общий_вопрос'
                    break
                else:
                    print("Неверный выбор")
            except:
                print("Введите число")
    
    # Сохранение маппинга
    mapping_data = {
        'categories': categories,
        'mapping': mapping,
        'original_classes_count': len(unique_classes)
    }
    
    with open('manual_mapping_final.json', 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nМаппинг сохранен в manual_mapping_final.json")
    print(f"Сопоставлено {len(mapping)} классов с {len(categories)} категориями")
    
    return mapping