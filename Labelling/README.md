В MultiwozMarkup.ipynb файле находится код для разметки датасета Multiwoz по эмоциям, сентименту и аннотациям. Сначала происходит парсинг датасета на отдельные высказывания, скачивание готовых моделей-классификаторов/аннотаторов, затем модели классифицируют эти высказывания и результаты сохраняются в файлы .json. 
Для оценки качества работы моделей построены confusion матрицы (см. прикрепленные картинки).
Выводы: модели, определяющие эмоции, на данном датасете работают плохо. Разметка по сентименту и аннотациям в целом пригодна для использования
