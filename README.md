<p align="center">
  <img src="https://img.shields.io/badge/Дипломный%20проект-RestatEval-ff69b4?style=for-the-badge&logo=github&logoColor=white">
</p>

# 🎓 Дипломный проект – RestatEval  

## 📖 Оглавление  
- [📌 Описание проекта](#описание-проекта)  
- [📂 Структура проекта](#структура-проекта)  
- [📬 Контакты](#контакты)  

## 📌 Описание проекта  

## 📂 Структура проекта  

```
📁 RestatEval
├── 📁 main_folder
│   ├── 📁 data
│   │   ├── 🗄️ ERZ_X_metrics.csv
│   │   └── 🗄️ ERZ_X_values.csv
│   ├── 📁 notebooks              
│   │   ├── 📓 0.1 (WIN) template_data_analysis.ipynb  
│   │   ├── 🔍 1.1 (WIN) parsing_&_setting_erz_rg.ipynb
│   │   ├── 💼 2.1 (WIN) setting_reestr_&_union_erz.ipynb
│   │   └── 📈 2.2 (WIN) setting_reestr_&_errors_graphs_&_no_spb_lo.ipynb
│   └── 📁 source  
│       └── 📋 parameters_erz_parsing.py  
└── 📁 raw_data               
    ├── 🛢️ csv         
    ├── 🛢️ excel         
    └── 🛢️ json        
```

🔗 Ссылки на папки и файлы:

- 📂 [main_folder](main_folder) — Основная папка с кодом и скриптами  

  * 📁 [data](main_folder/data) — Обработанные данные  

    - 🗄️ [ERZ_X_metrics.csv](main_folder/data/ERZ_X_metrics.csv) — Обработанные данные из ЕРЗ - стандартизированные (с применением эмбеддинга)  
    - 🗄️ [ERZ_X_values.csv](main_folder/data/ERZ_X_values.csv) — Обработанные данные из ЕРЗ - только числовой формат  

  * 📁 [notebooks](main_folder/notebooks) — Ноутбуки с анализом и кодом  

    - 📓 [0.1 (WIN) template_data_analysis.ipynb](main_folder/notebooks/0.1_(WIN)_template_data_analysis.ipynb) — Шаблон для новых ноутбуков  
    - 🔍 [1.1 (WIN) parsing_&_setting_erz_rg.ipynb](main_folder/notebooks/1.1_(WIN)_parsing_&_setting_erz_rg.ipynb) — Скрипт по парсингу сайта ЕРЗ
    - 💼 [2.1 (WIN) setting_reestr_&_union_erz.ipynb](main_folder/notebooks/2.1_(WIN)_setting_reestr_&_union_erz.ipynb) — Скрипт по обработке данных ЕГРН и их объединения с ЕРЗ
    - 📈 [2.2 (WIN) setting_reestr_&-errors_graphs-&_no_spb_lo.ipynb](main_folder/notebooks/2.2_(WIN)_setting_reestr_&_errors_graphs_&_no_spb_lo.ipynb) — Скрипт по обработке данных ЕГРН, подсчёте обнаруженных ошибках и выведении их распределения в графиках

  * 📁 [source](source) — Исходный код и вспомогательные скрипты
 
    - 📋 [parameters_erz_parsing.py](main_folder/source/parameters_erz_parsing.py) — Скрипт с параметрами страницы ЖК на сайте ЕРЗ, с которой парсятся данные

- 📂 [raw_data](raw_data) — Исходные (сырые) данные  

  * 🛢️ [csv](raw_data/csv) — Данные в формате .csv  
  * 🛢️ [excel](raw_data/excel) — Данные в формате .xlsx    
  * 🛢️ [json](raw_data/json) — Данные в формате .json  

## 📬 Контакты
| Роль | Имя | GitHub |
| -----| --- | ------ |
| 🛠️ Инженер данных и машинного обучения | Леонид | [barabrossk1n](https://github.com/barbarossk1n) |
| 💻 Инженер машинного обучения, Frontend/Backend-разработчик | Алёна | [ссылка на GitHub]() |
| 📈 Продакт-менеджер, маркетолог и Frontend-разработчик | Екатерина | [PodmogilnayaES](https://github.com/PodmogilnayaES) |
