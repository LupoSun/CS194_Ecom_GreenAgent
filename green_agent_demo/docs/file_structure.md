# Project File Structure

This document outlines the file and directory structure of the project.

```
.
├── .gitignore
├── green_agent_demo/
│   ├── .DS_Store
│   ├── .env
│   ├── ecom_green_agent.toml
│   ├── main_A2A.py
│   ├── quick_test.py
│   ├── requirements.txt
│   ├── test_a2a_green.py
│   ├── ab_src/
│   │   ├── __init__.py
│   │   ├── launcher.py
│   │   ├── green_agent/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   └── tau_green_agent.toml
│   │   ├── my_util/
│   │   │   ├── __init__.py
│   │   │   └── my_a2a.py
│   │   └── white_agent/
│   │       ├── __init__.py
│   │       └── agent.py
│   ├── archive/
│   │   ├── green_agent_card.toml
│   │   ├── main_A2A_AfterMeeting.py
│   │   ├── main_A2A_BeforeMeeting.py
│   │   └── main_FastAPI.py
│   ├── dataset/
│   │   ├── ic_products.csv
│   │   ├── super_shortened_orders_products_combined.csv
│   │   └── tasks.json
│   └── docs/
│       ├── a2a_demo_guide.md
│       ├── fastAPI_demo_guide.md
│       └── progress_summary.md
