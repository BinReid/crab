from flask import Flask, render_template, request, jsonify, session
import numpy as np
import json
import os
from datetime import datetime
import joblib
import pickle
import sys
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Глобальные переменные для модели
classifier = None
vectorizer = None
class_mapping = None
optimized_classes = None
confidence_threshold = 0.7

def numpy_to_python(obj):
    """Конвертация NumPy типов в Python типы для JSON сериализации"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

def load_model_and_config():
    """Загрузка модели и конфигурации"""
    global classifier, vectorizer, class_mapping, optimized_classes, confidence_threshold
    
    try:
        # 1. Загружаем модель
        print("Загрузка модели...")
        classifier = joblib.load('models/best_model_random_forest.joblib')
        original_classes = [str(c) for c in classifier.classes_]
        print(f"Модель загружена. Классов: {len(original_classes)}")
        
        # 2. Загружаем векторизатор
        print("Загрузка векторизатора...")
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        print("Векторизатор загружен.")
        
        # 3. Загружаем или создаем конфигурацию
        config_path = 'model_config_final.json'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            class_mapping = config.get('class_mapping', {})
            optimized_classes = config.get('optimized_classes', [])
            confidence_threshold = config.get('threshold', 0.7)
            print(f"Конфигурация загружена: {len(optimized_classes)} категорий")
        else:
            # Создаем простой маппинг
            print("Создание простого маппинга...")
            class_mapping = create_simple_mapping(original_classes)
            optimized_classes = sorted(list(set(class_mapping.values())))
            
            # Сохраняем конфигурацию
            config = {
                'optimized_classes': optimized_classes,
                'threshold': 0.7,
                'class_mapping': class_mapping,
                'model_info': {
                    'type': 'random_forest',
                    'n_original_classes': len(original_classes),
                    'n_optimized_classes': len(optimized_classes),
                    'version': '1.0'
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"Конфигурация сохранена: {len(optimized_classes)} категорий")
        
        return True
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_mapping(original_classes):
    """Создание простого маппинга классов"""
    mapping = {}
    
    # Группируем классы по категориям
    for cls in original_classes:
        cls_str = str(cls)
        
        # Маппинг на основе номеров классов
        if cls_str in ['14', '12', '11', '17', '10', '13']:
            mapping[cls_str] = 'проблема_получения_заказа'
        elif cls_str in ['15', '9', '20', '19', '21']:
            mapping[cls_str] = 'поиск_отслеживание_заказа'
        elif cls_str in ['23', '22', '24', '25', '26']:
            mapping[cls_str] = 'связь_с_оператором'
        elif cls_str in ['16', '18', '27', '28', '29']:
            mapping[cls_str] = 'проблема_с_постаматом'
        elif cls_str in ['8', '7', '6', '5', '4']:
            mapping[cls_str] = 'изменение_заказа'
        elif cls_str in ['0', '1', '2', '3', '30', '31']:
            mapping[cls_str] = 'уточнение_доставки'
        elif cls_str in ['32', '33', '34', '35']:
            mapping[cls_str] = 'оплата_возврат'
        else:
            mapping[cls_str] = 'общий_вопрос'
    
    return mapping

def predict_intent(text):
    """Предсказание интента для текста"""
    global classifier, vectorizer, class_mapping, confidence_threshold
    
    if not classifier or not vectorizer:
        return None
    
    try:
        # Векторизация текста
        features = vectorizer.transform([text])
        
        # Предсказание
        probabilities = classifier.predict_proba(features)[0]
        
        # Агрегация по оптимизированным классам
        optimized_probs = {}
        
        for i, orig_class in enumerate(classifier.classes_):
            prob = float(probabilities[i])  # Конвертируем сразу в float
            orig_class_str = str(orig_class)
            opt_class = class_mapping.get(orig_class_str, 'общий_вопрос')
            
            if opt_class not in optimized_probs:
                optimized_probs[opt_class] = 0.0
            optimized_probs[opt_class] += prob
        
        # Получаем лучший результат
        best_class = max(optimized_probs, key=optimized_probs.get)
        best_confidence = float(optimized_probs[best_class])
        
        # Получаем топ-3
        sorted_classes = sorted(optimized_probs.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_classes[:3]
        
        # Подготавливаем результат
        result = {
            'text': text,
            'main_intent': best_class,
            'main_confidence': best_confidence,
            'is_confident': bool(best_confidence >= confidence_threshold),  # Явное преобразование в bool
            'all_predictions': [
                {
                    'intent': intent,
                    'confidence': float(conf),
                    'is_confident': bool(conf >= confidence_threshold)
                }
                for intent, conf in top_3
            ],
            'optimized_probabilities': {k: float(v) for k, v in optimized_probs.items()}
        }
        
        # Добавляем оригинальное предсказание
        best_original_idx = int(np.argmax(probabilities))
        best_original_class = str(classifier.classes_[best_original_idx])
        best_original_prob = float(probabilities[best_original_idx])
        
        result['original_prediction'] = {
            'class': best_original_class,
            'confidence': best_original_prob
        }
        
        # Топ-3 оригинальных классов
        top_original_indices = np.argsort(probabilities)[-3:][::-1]
        original_top = []
        for idx in top_original_indices:
            original_top.append({
                'class': str(classifier.classes_[int(idx)]),  # Явное преобразование в int
                'confidence': float(probabilities[int(idx)])
            })
        result['original_top3'] = original_top
        
        return result
        
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        import traceback
        traceback.print_exc()
        return None

# История запросов
class QueryHistory:
    def __init__(self):
        self.history = []
    
    def add_query(self, text, result):
        # Очищаем результат от numpy типов
        cleaned_result = numpy_to_python(result)
        self.history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text': text,
            'result': cleaned_result
        })
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def get_history(self):
        return self.history
    
    def clear_history(self):
        self.history = []

history = QueryHistory()

# Маршруты Flask
@app.route('/')
def index():
    """Главная страница"""
    categories = optimized_classes if optimized_classes else []
    
    return render_template('index.html',
                         classifier_loaded=classifier is not None,
                         categories=categories)

@app.route('/classify', methods=['POST'])
def classify_text():
    """API для классификации текста"""
    if not classifier:
        return jsonify({
            'success': False,
            'error': 'Классификатор не загружен'
        })
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Текст не предоставлен'
            })
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'Текст пуст'
            })
        
        # Классификация
        result = predict_intent(text)
        if not result:
            return jsonify({
                'success': False,
                'error': 'Ошибка классификации'
            })
        
        # Сохраняем в историю
        history.add_query(text, result)
        
        # Добавляем timestamp
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['success'] = True
        
        # Конвертируем numpy типы
        cleaned_result = numpy_to_python(result)
        
        return jsonify(cleaned_result)
        
    except Exception as e:
        print(f"Ошибка в /classify: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    """API для пакетной классификации"""
    if not classifier:
        return jsonify({
            'success': False,
            'error': 'Классификатор не загружен'
        })
    
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Тексты не предоставлены'
            })
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'Тексты должны быть списком'
            })
        
        if len(texts) > 100:
            texts = texts[:100]
        
        # Классификация каждого текста
        results = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                result = predict_intent(text.strip())
                if result:
                    cleaned_result = {
                        'text': text.strip(),
                        'main_intent': str(result['main_intent']),
                        'main_confidence': float(result['main_confidence']),
                        'is_confident': bool(result['is_confident'])
                    }
                    results.append(cleaned_result)
        
        response = {
            'success': True,
            'total_texts': len(texts),
            'processed_texts': len(results),
            'results': results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Ошибка в /batch_classify: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """API для загрузки файла с текстами"""
    if not classifier:
        return jsonify({
            'success': False,
            'error': 'Классификатор не загружен'
        })
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Файл не предоставлен'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Файл не выбран'
            })
        
        # Проверяем расширение файла
        allowed_extensions = ['.txt', '.csv']
        if not any(file.filename.endswith(ext) for ext in allowed_extensions):
            return jsonify({
                'success': False,
                'error': 'Разрешены только файлы .txt и .csv'
            })
        
        # Читаем файл
        content = file.read().decode('utf-8', errors='ignore')
        
        # Парсим тексты
        texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(texts) > 1000:
            texts = texts[:1000]
        
        # Классификация
        results = []
        for text in texts:
            result = predict_intent(text)
            if result:
                cleaned_result = {
                    'text': text,
                    'main_intent': str(result['main_intent']),
                    'main_confidence': float(result['main_confidence']),
                    'is_confident': bool(result['is_confident'])
                }
                results.append(cleaned_result)
        
        response = {
            'success': True,
            'filename': file.filename,
            'total_texts': len(texts),
            'results': results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Ошибка в /upload_file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history')
def get_history():
    """Получить историю запросов"""
    history_data = history.get_history()
    response = {
        'success': True,
        'history': history_data,
        'total_queries': len(history_data)
    }
    
    return jsonify(response)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Очистить историю запросов"""
    history.clear_history()
    return jsonify({
        'success': True,
        'message': 'История очищена'
    })

@app.route('/classifier_info')
def classifier_info():
    """Информация о классификаторе"""
    if not classifier:
        return jsonify({
            'success': False,
            'error': 'Классификатор не загружен'
        })
    
    try:
        info = {
            'success': True,
            'original_classes': len(classifier.classes_),
            'optimized_classes': len(optimized_classes) if optimized_classes else 0,
            'categories': optimized_classes if optimized_classes else [],
            'confidence_threshold': float(confidence_threshold),
            'model_info': {
                'type': 'random_forest',
                'n_original_classes': len(classifier.classes_),
                'n_optimized_classes': len(optimized_classes) if optimized_classes else 0,
                'version': '1.0'
            }
        }
        
        return jsonify(info)
        
    except Exception as e:
        print(f"Ошибка в /classifier_info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/statistics')
def get_statistics():
    """Статистика использования"""
    history_data = history.get_history()
    
    stats = {
        'success': True,
        'total_queries': len(history_data),
        'queries_today': len([q for q in history_data 
                             if q['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))]),
        'categories_used': {},
        'confidence_stats': {
            'avg_confidence': 0.0,
            'confident_queries': 0
        }
    }
    
    # Собираем статистику по категориям
    for query in history_data:
        intent = query['result'].get('main_intent')
        if intent:
            if intent not in stats['categories_used']:
                stats['categories_used'][intent] = 0
            stats['categories_used'][intent] += 1
        
        # Статистика уверенности
        confidence = float(query['result'].get('main_confidence', 0))
        stats['confidence_stats']['avg_confidence'] += confidence
        if confidence >= confidence_threshold:
            stats['confidence_stats']['confident_queries'] += 1
    
    # Средняя уверенность
    if history_data:
        stats['confidence_stats']['avg_confidence'] = float(stats['confidence_stats']['avg_confidence'] / len(history_data))
    
    return jsonify(stats)

@app.route('/example_queries')
def example_queries():
    """Примеры запросов для тестирования"""
    examples = [
        {"text": "не могу получить посылку", "category": "проблема_получения_заказа"},
        {"text": "где находится моя посылка", "category": "поиск_отслеживание_заказа"},
        {"text": "соедините с оператором", "category": "связь_с_оператором"},
        {"text": "не работает постамат", "category": "проблема_с_постаматом"},
        {"text": "перенести срок доставки", "category": "изменение_заказа"},
        {"text": "здравствуйте", "category": "общий_вопрос"},
        {"text": "не открывается ячейка", "category": "проблема_получения_заказа"},
        {"text": "как отследить отправление", "category": "поиск_отслеживание_заказа"},
        {"text": "нужен консультант", "category": "связь_с_оператором"},
        {"text": "не пришел код подтверждения", "category": "проблема_с_постаматом"}
    ]
    
    return jsonify({
        'success': True,
        'examples': examples,
        'total_examples': len(examples)
    })

@app.route('/test_connection')
def test_connection():
    """Тест подключения к классификатору"""
    test_phrase = "тестовый запрос"
    
    if not classifier:
        return jsonify({
            'success': False,
            'status': 'classifier_not_loaded',
            'message': 'Классификатор не загружен'
        })
    
    try:
        result = predict_intent(test_phrase)
        if result:
            # Очищаем результат
            cleaned_result = numpy_to_python(result)
            response = {
                'success': True,
                'status': 'working',
                'test_phrase': test_phrase,
                'result': cleaned_result,
                'message': 'Классификатор работает корректно'
            }
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'status': 'error',
                'message': 'Ошибка предсказания'
            })
    except Exception as e:
        print(f"Ошибка в /test_connection: {e}")
        return jsonify({
            'success': False,
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    print("="*60)
    print("Flask веб-сервер для классификатора интентов")
    print("="*60)
    
    # Загружаем модель
    model_loaded = load_model_and_config()
    
    if model_loaded:
        print(f"✅ Модель загружена успешно!")
        print(f"   Исходных классов: {len(classifier.classes_)}")
        print(f"   Оптимизированных категорий: {len(optimized_classes)}")
        print(f"   Порог уверенности: {confidence_threshold}")
        print(f"   Категории: {optimized_classes}")
    else:
        print("❌ ВНИМАНИЕ: Не удалось загрузить модель!")
        print("   Убедитесь, что файлы моделей существуют в папке models/")
        print("   Необходимые файлы:")
        print("   - models/best_model_random_forest.joblib")
        print("   - models/tfidf_vectorizer.joblib")
    
    print("\n🌐 Запуск сервера на http://localhost:5000")
    print("   Для остановки нажмите Ctrl+C")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)