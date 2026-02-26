# 🏥 insurance-data-forge

> Simulated insurance datasets built with **Python**, **DuckDB**, and **Faker** —
> designed for analytics, machine learning, and actuarial practice.

---

## 📌 What Is This?

This repository generates **realistic but fully synthetic** insurance databases.
No real people. No real claims. Just clean, structured data you can use to:

- Practice SQL and data analysis
- Build and test machine learning models
- Learn actuarial and insurance data concepts
- Experiment with DuckDB as an analytical database

---

## 🗂️ Insurance Types

| Module | Status | Description |
|---|---|---|
| 🏥 `health/` | ✅ Ready | Health insurance claims, members, providers |
| 🔁 `reinsurance/` | 🚧 Coming soon | Treaty and facultative reinsurance data |
| 🧬 `life/` | 🚧 Coming soon | Life insurance policies and mortality data |
| 🚗 `auto/` | 🚧 Coming soon | Motor insurance claims and vehicles |

---

## 🏥 Health Insurance Database

### Schema — 8 Normalized Tables

```
dim_families ───┐
dim_employers ──┤
                ├──► dim_members ──► dim_policies ──► fact_claims ──► fact_claim_diagnoses
dim_providers ──┘                                          │
                                                     fact_payments
```

| Table | Rows | Description |
|---|---|---|
| `dim_families` | 2,000 | Household groups |
| `dim_employers` | 50 | Companies for group insurance |
| `dim_members` | 8,000 | Insured individuals |
| `dim_providers` | 300 | Hospitals, clinics, pharmacies |
| `dim_policies` | 3,000 | Insurance plan details |
| `fact_claims` | 50,000 | Core claims transactions |
| `fact_claim_diagnoses` | ~75,000 | ICD-10 diagnoses per claim |
| `fact_payments` | ~32,500 | Payment records |

### Key Features
- **Realistic distributions** — claim amounts follow a lognormal distribution
- **ICD-10 diagnosis codes** — real medical coding standard
- **CPT procedure codes** — real medical procedure standard
- **3% fraud rate** — built-in fraud flag for detection practice
- **25% chronic condition prevalence** — realistic population health
- **Fully normalized 3NF schema** — proper relational design
- **Reproducible** — seeded random generation (`seed=42`)

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install duckdb faker numpy pandas
```

### 2. Generate the health insurance database
```bash
cd health
python health_insurance_db_normalized.py
```

This creates `health_insurance_normalized.duckdb` in your folder.

### 3. Run practice queries
```bash
python practice_queries.py
```

### 4. Connect and explore manually
```python
import duckdb

con = duckdb.connect("health_insurance_normalized.duckdb")

# See all tables
con.execute("SHOW TABLES").df()

# Example query
con.execute("""
    SELECT
        p.provider_type,
        COUNT(*)                    AS total_claims,
        ROUND(AVG(c.claim_amount), 2) AS avg_claim
    FROM fact_claims   c
    JOIN dim_providers p ON c.provider_id = p.provider_id
    GROUP BY p.provider_type
    ORDER BY avg_claim DESC
""").df()
```

---

## 📚 Practice Query Levels

The `practice_queries.py` file contains **20 queries** across 4 difficulty levels:

| Level | Topics |
|---|---|
| ⭐ Beginner | SELECT, COUNT, GROUP BY, basic aggregations |
| ⭐⭐ Intermediate | JOINs, date filters, multi-table queries |
| ⭐⭐⭐ Advanced | CTEs, window functions, subqueries |
| ⭐⭐⭐⭐ Analytics | Fraud detection, risk scoring, cohort analysis |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [DuckDB](https://duckdb.org/) | Analytical in-process SQL database |
| [Faker](https://faker.readthedocs.io/) | Realistic fake data generation |
| [NumPy](https://numpy.org/) | Statistical distributions for simulation |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| Python 3.8+ | Core language |

---

## 📁 Repository Structure

```
insurance-data-forge/
│
├── README.md
├── .gitignore
│
├── health/
│   ├── health_insurance_db_normalized.py   # database builder
│   └── practice_queries.py                 # 20 practice SQL queries
│
├── reinsurance/                            # coming soon
├── life/                                   # coming soon
└── docs/
    └── schema_diagram.md                   # detailed schema docs
```

---

## ⚠️ Important Notes

- **No real data** — everything is synthetically generated
- **Database files are gitignored** — `.duckdb` files are never pushed to GitHub
- Run the Python script locally to generate your own database

---

## 🤝 Contributing

Contributions are welcome! If you want to add a new insurance type (reinsurance, life, auto), feel free to open a pull request.

---

## 📄 License

MIT License — free to use for learning, research, and personal projects.