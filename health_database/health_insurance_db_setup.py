"""
Health Insurance DuckDB Database — NORMALIZED VERSION
======================================================
Fully normalized 3NF relational schema.
Each piece of information is stored in exactly ONE place.
Use JOINs to combine tables for analysis.

Schema / Relationships:
  dim_families   <──  dim_members  ──>  dim_employers
                           │
                      dim_policies
                           │
  dim_providers  <──  fact_claims
                           │
                 fact_claim_diagnoses   (one claim → many diagnoses)
                           │
                      fact_payments

Tables:
  1. dim_families            — household groups
  2. dim_employers           — companies (group insurance)
  3. dim_members             — insured individuals (NO repeated data)
  4. dim_providers           — hospitals, clinics, pharmacies
  5. dim_policies            — insurance plan details
  6. fact_claims             — claim transactions (FK only, no copied fields)
  7. fact_claim_diagnoses    — diagnosis + procedure per claim (separate table)
  8. fact_payments           — payment records per approved claim

Usage:
  pip install duckdb faker numpy pandas
  python health_insurance_db_normalized.py
"""
#%%
import duckdb
import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta
import random
import os
#%%
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH     = "health_insurance_normalized.duckdb"
N_FAMILIES  = 2_000
N_EMPLOYERS = 50
N_MEMBERS   = 8_000
N_PROVIDERS = 300
N_POLICIES  = 3_000
N_CLAIMS    = 50_000

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ─────────────────────────────────────────────
# LOOKUP LISTS
# ─────────────────────────────────────────────
def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))

ICD10_CODES = [
    "I10","E11","J45","M54","K21","F32","I25","E78",
    "J06","N39","G43","M79","L30","F41","E66","Z00"
]
PROCEDURE_CODES = [
    "99213","99214","93000","85025","80053","71046",
    "99232","36415","90834","97110","43239","66984"
]
PLAN_TYPES      = ["HMO","PPO","EPO","HDHP","POS"]
PROVIDER_TYPES  = ["Hospital","Clinic","Pharmacy","Physician","Specialist","Lab"]
CLAIM_TYPES     = ["Inpatient","Outpatient","Pharmacy","Emergency","Preventive"]
CHANNELS        = ["Online","Mobile App","Paper","Agent","Hospital Portal"]
INSURANCE_CATS  = ["Individual","Family","Group"]
CLAIM_STATUSES  = ["Approved","Rejected","Pending","Under Review"]
GENDERS         = ["Male","Female","Non-binary"]
EMPLOYMENT_CATS = ["Employed","Self-Employed","Unemployed","Retired","Student"]
#%%
# ═══════════════════════════════════════════════════════
# TABLE BUILDERS
# ═══════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. DIM_FAMILIES
#    Primary key: family_id
#    No foreign keys
# ─────────────────────────────────────────────
def build_families(n: int) -> pd.DataFrame:
    print(f"  Building dim_families        ({n:>6,} rows)...")
    return pd.DataFrame([{
        "family_id":       f"FAM{i+1:06d}",
        "family_size":     random.randint(1, 7),
        "residential_zip": fake.zipcode(),
        "state":           fake.state_abbr(),
        "income_category": random.choice(["Low", "Middle", "High"]),
    } for i in range(n)])

#%%
build_families(8)
#%%
# ─────────────────────────────────────────────
# 2. DIM_EMPLOYERS
#    Primary key: employer_id
#    No foreign keys
# ─────────────────────────────────────────────
def build_employers(n: int) -> pd.DataFrame:
    print(f"  Building dim_employers       ({n:>6,} rows)...")
    return pd.DataFrame([{
        "employer_id":    f"EMP{i+1:05d}",
        "employer_name":  fake.company(),
        "industry":       random.choice(["Healthcare","Finance","Tech",
                                         "Education","Retail","Manufacturing"]),
        "employee_count": random.randint(50, 10_000),
        "state":          fake.state_abbr(),
    } for i in range(n)])

#%%
build_employers(5)
#%%
# ─────────────────────────────────────────────
# 3. DIM_MEMBERS
#    Primary key : member_id
#    Foreign keys: family_id  → dim_families
#                  employer_id → dim_employers
#
#    NORMALIZED RULES APPLIED:
#    ✓  gender, date_of_birth, zip_code stored HERE only
#    ✓  age removed  (calculated from date_of_birth — 3NF rule)
#    ✓  No claim data here
# ─────────────────────────────────────────────
def build_members(n: int, families: pd.DataFrame,
                  employers: pd.DataFrame) -> pd.DataFrame:
    print(f"  Building dim_members         ({n:>6,} rows)...")
    family_ids   = families["family_id"].tolist()
    employer_ids = employers["employer_id"].tolist()
    rows = []
    for i in range(n):
        ins_cat = random.choice(INSURANCE_CATS)
        rows.append({
            "member_id":           f"MEM{i+1:07d}",
            # Foreign keys (NULL when not applicable)
            "family_id":           random.choice(family_ids)   if ins_cat == "Family" else None,
            "employer_id":         random.choice(employer_ids) if ins_cat == "Group"  else None,
            # Demographics — stored ONCE here, never repeated elsewhere
            "first_name":          fake.first_name(),
            "last_name":           fake.last_name(),
            "date_of_birth":       rand_date(date(1940,1,1), date(2005,12,31)),
            "gender":              random.choice(GENDERS),
            "marital_status":      random.choice(["Single","Married","Divorced","Widowed"]),
            "num_dependents":      random.randint(0, 5),
            "zip_code":            fake.zipcode(),
            "state":               fake.state_abbr(),
            "employment_category": random.choice(EMPLOYMENT_CATS),
            "insurance_category":  ins_cat,
            # Health info — belongs to the member, not the claim
            "chronic_condition_flag": random.random() < 0.25,
            "health_risk_score":      round(np.random.beta(2, 5) * 100, 2),
            "smoking_status":         random.choice(["Never","Former","Current"]),
            "bmi":                    round(random.gauss(27, 5), 1),
            # Coverage info
            "coverage_start_date": rand_date(date(2018,1,1), date(2024,1,1)),
            "coverage_status":     random.choices(
                                       ["Active","Suspended","Terminated"],
                                       weights=[80,10,10])[0],
            "premium_amount":      round(random.uniform(150, 900), 2),
            "benefit_limit":       random.choice([50_000,100_000,200_000,500_000]),
            "msa_balance":         round(random.uniform(0, 5_000), 2),
        })
    return pd.DataFrame(rows)

#%%
build_members(8, build_families(2), build_employers(2))
#%%
# ─────────────────────────────────────────────
# 4. DIM_PROVIDERS
#    Primary key: provider_id
#    No foreign keys
#
#    NORMALIZED RULES APPLIED:
#    ✓  provider_type stored HERE only — never copied to fact_claims
# ─────────────────────────────────────────────
def build_providers(n: int) -> pd.DataFrame:
    print(f"  Building dim_providers       ({n:>6,} rows)...")
    return pd.DataFrame([{
        "provider_id":         f"PRV{i+1:06d}",
        "provider_name":       fake.company() + " Medical",
        "provider_type":       random.choice(PROVIDER_TYPES),
        "state":               fake.state_abbr(),
        "zip_code":            fake.zipcode(),
        "network_flag":        random.random() < 0.75,
        "accreditation_level": random.choice(["Level1","Level2","Level3"]),
        "established_year":    random.randint(1970, 2020),
    } for i in range(n)])

#%%
build_providers(4)
#%%
# ─────────────────────────────────────────────
# 5. DIM_POLICIES
#    Primary key : policy_id
#    Foreign key : member_id → dim_members
# ─────────────────────────────────────────────
def build_policies(n: int, members: pd.DataFrame) -> pd.DataFrame:
    print(f"  Building dim_policies        ({n:>6,} rows)...")
    member_ids = members["member_id"].tolist()
    rows = []
    for i in range(n):
        start = rand_date(date(2018,1,1), date(2024,1,1))
        rows.append({
            "policy_id":          f"POL{i+1:07d}",
            "member_id":          random.choice(member_ids),   # FK → dim_members
            "plan_type":          random.choice(PLAN_TYPES),
            "insurance_category": random.choice(INSURANCE_CATS),
            "start_date":         start,
            "end_date":           start + timedelta(days=365),
            "deductible":         random.choice([500,1000,2000,3000,5000]),
            "co_payment_rate":    round(random.uniform(0.05, 0.30), 2),
            "out_of_pocket_max":  random.choice([3000,5000,7000,10000]),
            "premium_amount":     round(random.uniform(150, 900), 2),
            "status":             random.choices(
                                      ["Active","Expired","Cancelled"],
                                      weights=[70,20,10])[0],
        })
    return pd.DataFrame(rows)

#%%
build_policies(8, build_members(8, build_families(2), build_employers(2)))
#%%
# ─────────────────────────────────────────────
# 6. FACT_CLAIMS  ← NORMALIZED CORE TABLE
#    Primary key : claim_id
#    Foreign keys: policy_id   → dim_policies
#                  member_id   → dim_members
#                  provider_id → dim_providers
#
#    NORMALIZED RULES APPLIED:
#    ✗  member_gender     REMOVED  (lives in dim_members)
#    ✗  member_birth_date REMOVED  (lives in dim_members)
#    ✗  member_zip_code   REMOVED  (lives in dim_members)
#    ✗  provider_type     REMOVED  (lives in dim_providers)
#    ✗  diagnosis_code    MOVED    (→ fact_claim_diagnoses)
#    ✗  procedure_code    MOVED    (→ fact_claim_diagnoses)
#    ✓  Only claim-specific transactional data remains here
# ─────────────────────────────────────────────
def build_claims(n: int, members: pd.DataFrame, policies: pd.DataFrame,
                 providers: pd.DataFrame) -> pd.DataFrame:
    print(f"  Building fact_claims         ({n:>6,} rows)  [Normalized]...")
    member_ids   = members["member_id"].tolist()
    policy_ids   = policies["policy_id"].tolist()
    provider_ids = providers["provider_id"].tolist()
    rows = []
    for i in range(n):
        claim_date   = rand_date(date(2019,1,1), date(2024,12,31))
        is_inpatient = random.random() < 0.20
        los          = random.randint(1, 14) if is_inpatient else 0
        discharge    = claim_date + timedelta(days=los) if is_inpatient else None
        claim_amt    = round(np.random.lognormal(7, 1.2), 2)
        rows.append({
            # ── Identifiers & Foreign Keys only
            "claim_id":              f"CLM{i+1:08d}",
            "policy_id":             random.choice(policy_ids),   # FK → dim_policies
            "member_id":             random.choice(member_ids),   # FK → dim_members
            "provider_id":           random.choice(provider_ids), # FK → dim_providers
            # ── Claim-specific data (belongs here and nowhere else)
            "claim_date":            claim_date,
            "claim_type":            random.choice(CLAIM_TYPES),
            "claim_status":          random.choices(CLAIM_STATUSES, weights=[65,10,15,10])[0],
            "claim_amount":          claim_amt,
            "paid_amount":           round(claim_amt * random.uniform(0.5, 0.95), 2),
            "admission_date":        claim_date if is_inpatient else None,
            "discharge_date":        discharge,
            "length_of_stay":        los,
            "submission_channel":    random.choice(CHANNELS),
            "network_provider_flag": random.random() < 0.75,
            "co_payment_amount":     round(claim_amt * random.uniform(0.05, 0.20), 2),
            "is_fraud_flagged":      random.random() < 0.03,
        })
    return pd.DataFrame(rows)

#%%
build_claims(8, build_members(8, build_families(2), build_employers(2)),
             build_policies(8, build_members(8, build_families(2), build_employers(2))),
             build_providers(5))
#%%
# ─────────────────────────────────────────────
# 7. FACT_CLAIM_DIAGNOSES  ← NEW NORMALIZED TABLE
#    Primary key : diagnosis_id
#    Foreign key : claim_id → fact_claims
#
#    WHY A SEPARATE TABLE?
#    A single claim can have MULTIPLE diagnoses and procedures.
#    Storing them in fact_claims would require either:
#      (a) multiple columns  diagnosis_1, diagnosis_2... (bad)
#      (b) comma-separated values in one cell            (violates 1NF)
#    A separate table is the correct 3NF solution.
# ─────────────────────────────────────────────
def build_claim_diagnoses(claims: pd.DataFrame) -> pd.DataFrame:
    claim_ids = claims["claim_id"].tolist()
    # Each claim gets 1-3 diagnosis/procedure pairs
    rows = []
    diag_counter = 1
    for cid in claim_ids:
        n_diags = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        for rank in range(1, n_diags + 1):
            rows.append({
                "diagnosis_id":    f"DGN{diag_counter:09d}",
                "claim_id":        cid,               # FK → fact_claims
                "diagnosis_rank":  rank,              # 1=primary, 2=secondary...
                "diagnosis_code":  random.choice(ICD10_CODES),
                "procedure_code":  random.choice(PROCEDURE_CODES),
            })
            diag_counter += 1
    print(f"  Building fact_claim_diagnoses ({len(rows):>6,} rows)...")
    return pd.DataFrame(rows)
#%%
build_claim_diagnoses(build_claims(8, build_members(8, build_families(2), build_employers(2)),
                                  build_policies(8, build_members(8, build_families(2), build_employers(2))),
                                  build_providers(5)))
#%%

# ─────────────────────────────────────────────
# 8. FACT_PAYMENTS
#    Primary key : payment_id
#    Foreign keys: claim_id  → fact_claims
#                  member_id → dim_members
# ─────────────────────────────────────────────
def build_payments(claims: pd.DataFrame) -> pd.DataFrame:
    approved = claims[claims["claim_status"] == "Approved"].copy()
    print(f"  Building fact_payments       ({len(approved):>6,} rows)...")
    rows = []
    for i, (_, row) in enumerate(approved.iterrows()):
        rows.append({
            "payment_id":        f"PAY{i+1:08d}",
            "claim_id":          row["claim_id"],     # FK → fact_claims
            "member_id":         row["member_id"],    # FK → dim_members
            "payment_date":      row["claim_date"] + timedelta(days=random.randint(3, 45)),
            "paid_amount":       row["paid_amount"],
            "co_payment_amount": row["co_payment_amount"],
            "payment_method":    random.choice(["Bank Transfer","Check",
                                                "Direct Deposit","Digital Wallet"]),
            "payment_status":    random.choices(
                                     ["Completed","Failed","Reversed"],
                                     weights=[92, 5, 3])[0],
        })
    return pd.DataFrame(rows)

#%%
build_payments(build_claims(8, build_members(8, build_families(2), build_employers(2)),
                            build_policies(8, build_members(8, build_families(2), build_employers(2))),
                            build_providers(5)))
#%%
# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Health Insurance DuckDB — NORMALIZED DATABASE BUILDER")
    print("=" * 60)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Removed existing '{DB_PATH}'")

    # ── Generate all tables
    print("\n[1/2] Generating simulated data...")
    families   = build_families(N_FAMILIES)
    employers  = build_employers(N_EMPLOYERS)
    members    = build_members(N_MEMBERS, families, employers)
    providers  = build_providers(N_PROVIDERS)
    policies   = build_policies(N_POLICIES, members)
    claims     = build_claims(N_CLAIMS, members, policies, providers)
    diagnoses  = build_claim_diagnoses(claims)
    payments   = build_payments(claims)

    # ── Load into DuckDB
    print("\n[2/2] Loading into DuckDB...")
    con = duckdb.connect(DB_PATH)

    table_map = {
        "dim_families":         families,
        "dim_employers":        employers,
        "dim_members":          members,
        "dim_providers":        providers,
        "dim_policies":         policies,
        "fact_claims":          claims,
        "fact_claim_diagnoses": diagnoses,
        "fact_payments":        payments,
    }

    for name, df in table_map.items():
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  ✓  {name:<28} {count:>8,} rows")

    # ── Indexes
    print("\n  Adding indexes...")
    indexes = [
        ("idx_members_family",     "dim_members",          "family_id"),
        ("idx_members_employer",   "dim_members",          "employer_id"),
        ("idx_policies_member",    "dim_policies",         "member_id"),
        ("idx_claims_member",      "fact_claims",          "member_id"),
        ("idx_claims_policy",      "fact_claims",          "policy_id"),
        ("idx_claims_provider",    "fact_claims",          "provider_id"),
        ("idx_claims_date",        "fact_claims",          "claim_date"),
        ("idx_diagnoses_claim",    "fact_claim_diagnoses", "claim_id"),
        ("idx_payments_claim",     "fact_payments",        "claim_id"),
        ("idx_payments_member",    "fact_payments",        "member_id"),
    ]
    for idx_name, tbl, col in indexes:
        con.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl}({col})")
    print(f"  ✓  {len(indexes)} indexes created")

    # ── Sample JOIN query to show normalization in action
    print("\n  Sample normalized query — claims with full member + provider info:")
    result = con.execute("""
        SELECT
            c.claim_id,
            c.claim_date,
            c.claim_amount,
            c.claim_status,
            -- from dim_members (via JOIN)
            m.gender          AS member_gender,
            m.date_of_birth   AS member_birth_date,
            m.zip_code        AS member_zip_code,
            -- from dim_providers (via JOIN)
            p.provider_type,
            p.provider_name,
            -- from fact_claim_diagnoses (via JOIN)
            d.diagnosis_code,
            d.procedure_code
        FROM fact_claims           c
        JOIN dim_members           m ON c.member_id   = m.member_id
        JOIN dim_providers         p ON c.provider_id = p.provider_id
        LEFT JOIN fact_claim_diagnoses d
               ON c.claim_id = d.claim_id AND d.diagnosis_rank = 1
        LIMIT 5
    """).df()
    print(result.to_string(index=False))

    con.close()

    size_mb = os.path.getsize(DB_PATH) / (1024**2)
    print(f"\n✅  Database saved to '{DB_PATH}'  ({size_mb:.1f} MB)")
    print("=" * 60)
    print("  Tables created:")
    for name in table_map:
        print(f"    - {name}")
    print("\n  Connect with:")
    print("    import duckdb")
    print(f"    con = duckdb.connect('{DB_PATH}')")
    print("    con.execute('SHOW TABLES').df()")
    print("=" * 60)


if __name__ == "__main__":
    main()
# %%
con = duckdb.connect("health_insurance_normalized.duckdb")

con.execute("SHOW TABLES").df()
# %%
