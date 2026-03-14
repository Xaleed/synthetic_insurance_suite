"""
Health Insurance DuckDB Database — NORMALIZED VERSION v3
=========================================================
KEY IMPROVEMENTS OVER v2:
  ✓  NEW dim_contracts table — separates WHO buys from WHAT they buy
  ✓  Group contracts support mixed coverage:
       some employees choose Individual, some choose Family
  ✓  All members of a family ALWAYS share the same single policy
       (no mixing within a family — one family = one policy)
  ✓  Every member has exactly one role in exactly one policy
  ✓  member_id is the universal identifier for every human being
       regardless of contract type

SCHEMA v3:
  dim_families   <──  dim_members  ──>  (linked via policy_members)
  dim_employers  <──  dim_contracts
                           │
                      dim_policies
                           │
                    policy_members   (bridge: policy → members)
                           │
  dim_providers  <──  fact_claims
                           │
                 fact_claim_diagnoses
                           │
                      fact_payments

CONTRACT vs POLICY — why two separate tables?
  dim_contracts = the BUYING AGREEMENT between the insurer and the buyer
                  answers: WHO is buying? (person / family / company)
                  one contract per buyer

  dim_policies  = the SPECIFIC PLAN chosen within that contract
                  answers: WHAT plan? (HMO/PPO, deductible, premium etc.)
                  Individual contract → 1 policy
                  Family contract     → 1 policy (covers whole family)
                  Group contract      → 1 policy per employee
                                        (employee chooses Individual or Family level)

FAMILY RULE (enforced in v3):
  All members belonging to the same family_id
  ALWAYS share the same single policy.
  No family member can have a different policy from the rest.

Usage:
  pip install duckdb faker numpy pandas
  python health_insurance_normalized_v3.py
"""

import duckdb
import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta
import random
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH = "health_insurance_normalized_v3.duckdb"
# The file path where the entire database will be saved on disk.
# DuckDB is a file-based database — meaning the ENTIRE database
# (all tables, indexes, data) lives inside this single file.
# This is similar to SQLite but DuckDB is optimized for analytics.
#
# Practical notes:
#   - If the file already exists, the script DELETES and recreates it
#   - The file will be roughly 50–150 MB depending on data volume
#   - You connect to it with: duckdb.connect("health_insurance_normalized_v3.duckdb")
#   - You can share the entire database by sharing just this one file


N_FAMILIES = 2_000
# The number of FAMILY HOUSEHOLD records in dim_families.
# (The underscore in 2_000 is just Python's visual separator — same as 2000)
#
# A family in this database means a household group where ONE contract
# covers ALL members together under a Family-type contract.
# Important rules:
#   - family_size is always >= 2 (a single person buys Individual, not Family)
#   - ALL members belonging to the same family share EXACTLY ONE policy
#   - No family member can have a different policy from the rest (v3 rule)
#
# HOW IT CONNECTS TO THE REST:
#   dim_contracts.family_id → points here (one Family contract per household)
#   dim_contracts → dim_policies → policy_members → dim_members
#   So a family of 4 people looks like this:
#
#     FAM000001
#         └── CON_FAM_001  (one Family contract)
#                 └── POL_001  (one shared policy for the whole family)
#                         ├── MEM001  role = "Policyholder"  (the buyer)
#                         ├── MEM002  role = "Dependent"     (spouse)
#                         ├── MEM003  role = "Dependent"     (child)
#                         └── MEM004  role = "Dependent"     (child)
#



N_EMPLOYERS = 50
# The number of EMPLOYER COMPANY records in dim_employers.
#
# An employer in this database is a company that purchases ONE Group contract
# from the insurer to cover all of its employees. This is called
# employer-sponsored insurance — the most common type of coverage in the US.
#
# HOW GROUP COVERAGE WORKS IN v3:
#   - The company signs ONE group contract with the insurer
#   - Each individual employee then chooses their own coverage LEVEL:
#       40% of employees → Individual policy (covers just themselves)
#       60% of employees → Family policy (covers themselves + dependents)
#   - All employees share the SAME group contract_id
#   - But each employee gets their OWN policy_id with their chosen level
#   - Within each employee's family → all share ONE policy 
#
#   Example with EMP00001:
#     EMP00001
#         └── CON_GRP_001  (one Group contract for the whole company)
#                 ├── POL_101  coverage_level = "Individual"  → just Employee A
#                 ├── POL_102  coverage_level = "Family"      → Employee B + family
#                 └── POL_103  coverage_level = "Individual"  → just Employee C
#
# WHY ONLY 50 EMPLOYERS for 8,000 MEMBERS?
# Roughly 1/3 of 8,000 members = ~2,667 are covered via Group contracts.
# Each employer gets between 10–80 employees assigned randomly.
# Average = ~2,667 / 50 ≈ 53 employees per employer.
# This is realistic for small-to-medium sized companies.
# In real life, one large employer can cover thousands of employees
# under a single group contract.
#
# Real world employer insurance facts:
#   - Employer typically pays 70–80% of the monthly premium
#   - Employee pays remaining 20–30% via paycheck deduction
#   - Losing your job = losing your insurance (this is why COBRA exists)


N_MEMBERS = 8_000
# The total number of INSURED INDIVIDUALS in dim_members.
# This is the most important dimension table — every single human being
# covered by any insurance in this database gets EXACTLY ONE row here.
#
# KEY DESIGN PRINCIPLE IN v3:
#   member_id is the UNIVERSAL identifier for every person
#   regardless of what type of contract covers them.
#   A person covered under an Individual contract,
#   a person who is a dependent on a Family contract,
#   and an employee covered under a Group contract
#   ALL have their own member_id — they are all equal citizens in dim_members.
#
# WHAT dim_members stores in v3:
#   Pure demographics only — name, date_of_birth, gender, zip_code, state
#   Health info           — chronic_condition_flag, bmi, smoking_status
#   Employment info       — employment_category
#   Financial info        — msa_balance (Medical Savings Account)
#
# WHAT dim_members does NOT store in v3 (moved to better places):
#   ✗  insurance_category  → belongs to dim_contracts (the contract, not the person)
#   ✗  family_id           → belongs to dim_contracts
#   ✗  employer_id         → belongs to dim_contracts
#   ✗  premium_amount      → belongs to dim_policies (the plan, not the person)
#   ✗  coverage_status     → belongs to dim_policies
#
# HOW 8,000 MEMBERS ARE DISTRIBUTED (approximately):
#   ~2,667 covered under Individual contracts  → 1 person per policy
#   ~2,667 covered under Family contracts      → grouped into 2,000 households
#   ~2,667 covered under Group contracts       → spread across 50 employers
#                                                some as employees, some as dependents


N_PROVIDERS = 300
# The number of HEALTHCARE PROVIDER records in dim_providers.
# A provider is any entity that DELIVERS medical care to a patient.
#
# Provider types in this database:
#   Hospital    → inpatient stays, surgeries, emergency care
#   Clinic      → routine outpatient visits
#   Pharmacy    → dispenses prescription drugs
#   Physician   → individual primary care doctor
#   Specialist  → cardiologist, dermatologist, orthopedist etc.
#   Lab         → blood tests, pathology, imaging (X-ray, MRI)
#
# HOW PROVIDERS CONNECT TO CLAIMS:
# Every claim in fact_claims has a provider_id foreign key.
# The provider is the ORIGIN of the claim — not a submission channel.
# (submission channel is a separate field: Online / Paper / Mobile App etc.)
#
#   Member gets sick
#       → visits a Provider
#           → Provider delivers care
#               → Provider submits a Claim to the insurer
#                   → Insurer reviews and pays the Provider
#
# THE NETWORK FLAG — most important provider attribute:
#   network_flag = True  (~75% of providers)
#       → Provider has a pre-negotiated CONTRACT with the insurer
#       → Member pays LESS out of pocket
#       → Insurer reimburses at full agreed rate
#
#   network_flag = False (~25% of providers)
#       → No contract exists with the insurer
#       → Member pays MUCH MORE out of pocket
#       → For HMO plans: out-of-network is NOT covered at all
#       → For PPO plans: out-of-network covered but very expensive
#
# WHY 300 PROVIDERS for 8,000 MEMBERS?
#   8,000 / 300 ≈ 27 members per provider on average.
#   Realistic — a single hospital or clinic serves hundreds of patients.
#   With 50,000 claims across 300 providers → ~167 claims per provider.


N_CLAIMS = 50_000
# The number of CLAIM TRANSACTION records in fact_claims.
# This is the LARGEST and most central table — nearly all analytics,
# reporting, and dashboards will start from here.
#
# WHAT IS A CLAIM?
# A claim is a formal request for reimbursement submitted to the insurer
# after a medical event occurs:
#
#   Member visits provider (doctor / hospital / pharmacy)
#       → receives care
#           → provider submits a claim:
#             "I treated member X under policy Y,
#              performed procedure Z for diagnosis D,
#              the bill is $1,500 — please pay me"
#               → insurer reviews:
#                   Approved     (65%) → payment created in fact_payments
#                   Rejected     (10%) → denied, no payment
#                   Pending      (15%) → not yet processed
#                   Under Review (10%) → flagged for investigation
#
# WHAT EACH CLAIM RECORDS:
#   Who submitted   → member_id   (which person received care)
#   Where           → provider_id (which hospital/clinic/doctor)
#   Under           → policy_id   (which insurance plan covers it)
#   When            → claim_date
#   What type       → claim_type  (Inpatient/Outpatient/Pharmacy/Emergency/Preventive)
#   How submitted   → submission_channel (Online/Paper/Mobile App etc.)
#   Money           → claim_amount, paid_amount, co_payment_amount
#   Outcome         → claim_status (Approved/Rejected/Pending/Under Review)
#   Flags           → is_fraud_flagged (~3% of claims), network_provider_flag
#
# KEY INTEGRITY RULE ENFORCED IN v3:
#   The policy_id on each claim MUST be a policy that actually covers
#   that member — verified via the policy_members bridge table.
#   This means a claim can never be filed under a policy
#   that does not cover the member who received the care.
#
# WHY 50,000 CLAIMS for 8,000 MEMBERS?
#   50,000 / 8,000 = ~6.25 claims per member
#   Data spans 2019–2024 = ~6 years
#   → roughly 1 claim per member per year
#   Very realistic — the average person makes 1–3 insurance claims per year.
#   Members with chronic_condition_flag = True would realistically
#   generate more claims than healthy members.
#
# FRAUD FLAG:
#   is_fraud_flagged = True for ~3% of claims (~1,500 claims)
#   These represent suspected overbilling, phantom services,
#   duplicate submissions, or identity theft cases.

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ─────────────────────────────────────────────
# LOOKUP LISTS
# ─────────────────────────────────────────────
def rand_date(start: date, end: date) -> date:
    # A helper function that generates a random date between two dates.
    #
    # PARAMETERS:
    #   start : date  → the earliest possible date (inclusive)
    #   end   : date  → the latest possible date   (inclusive)
    #
    # HOW IT WORKS — step by step:
    #   (end - start).days
    #       → subtracts two date objects to get a timedelta
    #       → .days converts that timedelta to a plain integer
    #       → Example: date(2024,12,31) - date(2019,1,1) = 2191 days
    #
    #   random.randint(0, 2191)
    #       → picks a random integer between 0 and 2191 inclusive
    #       → 0 means "stay on the start date"
    #       → 2191 means "land on the end date"
    #
    #   start + timedelta(days=random_number)
    #       → adds that many days to the start date
    #       → produces a valid date somewhere in the range
    #
    # EXAMPLE:
    #   rand_date(date(2019,1,1), date(2024,12,31))
    #   → might return date(2022, 7, 14)  — a random date in that 6-year window
    #
    # WHERE IT IS USED IN THIS DATABASE:
    #   - Member coverage_start_date  → rand_date(2018, 2024)
    #   - Contract and policy dates   → rand_date(2018, 2024)
    #   - Claim dates                 → rand_date(2019, 2024)
    #   - Payment dates               → claim_date + random 3–45 days
    return start + timedelta(days=random.randint(0, (end - start).days))


ICD10_CODES = [
    "I10","E11","J45","M54","K21","F32","I25","E78",
    "J06","N39","G43","M79","L30","F41","E66","Z00"
]
# ICD-10 = International Classification of Diseases, 10th Revision.
# This is the OFFICIAL global standard for classifying medical diagnoses.
# Published by the World Health Organization (WHO) and used by every
# insurance company, hospital, and government health system worldwide.
#
# Every claim MUST have at least one ICD-10 code to justify WHY
# the patient needed care. Without a diagnosis code, the insurer
# has no basis to approve or calculate reimbursement.
#
# Each code in our list represents a real common condition:
#
#   "I10"  → Essential Hypertension (high blood pressure)
#              Most common chronic condition in the US — ~45% of adults
#
#   "E11"  → Type 2 Diabetes Mellitus
#              ~11% of US adults, generates frequent claims for monitoring
#
#   "J45"  → Asthma
#              Common respiratory condition requiring ongoing medication
#
#   "M54"  → Dorsalgia (back pain)
#              One of the top reasons people visit a doctor
#
#   "K21"  → Gastro-esophageal Reflux Disease (GERD / acid reflux)
#              Very common, often requires prescription medication
#
#   "F32"  → Major Depressive Episode (depression)
#              Leading cause of mental health claims
#
#   "I25"  → Chronic Ischemic Heart Disease (coronary artery disease)
#              Serious cardiac condition, generates high-cost claims
#
#   "E78"  → Disorders of Lipoprotein Metabolism (high cholesterol)
#              Extremely common, often treated with statins
#
#   "J06"  → Acute Upper Respiratory Infection (common cold / flu)
#              One of the highest VOLUME diagnosis codes — very frequent
#
#   "N39"  → Urinary Tract Infection (UTI)
#              Very common, especially in women
#
#   "G43"  → Migraine
#              Common neurological condition with recurring claims
#
#   "M79"  → Other Soft Tissue Disorders (fibromyalgia / muscle pain)
#              Broad category covering various pain conditions
#
#   "L30"  → Other Dermatitis (eczema / skin inflammation)
#              Common skin condition requiring dermatology visits
#
#   "F41"  → Other Anxiety Disorders
#              Second most common mental health diagnosis after depression
#
#   "E66"  → Obesity
#              Increasingly common, drives claims for related conditions
#
#   "Z00"  → General Examination / Health Check-up (preventive visit)
#              Annual wellness visit — usually $0 copay under ACA rules
#              Highest volume code in preventive claim_type
#
# In fact_claim_diagnoses, each claim gets 1–3 of these codes assigned.
# The first one (diagnosis_rank = 1) is the PRIMARY diagnosis —
# the main reason for the visit.
# Ranks 2 and 3 are SECONDARY diagnoses — additional conditions
# noted during the same visit.


PROCEDURE_CODES = [
    "99213","99214","93000","85025","80053","71046",
    "99232","36415","90834","97110","43239","66984"
]
# CPT = Current Procedural Terminology codes.
# Published by the American Medical Association (AMA).
# These describe WHAT SERVICE was performed on the patient —
# as opposed to ICD-10 which describes WHY (the diagnosis).
#
# Together: ICD-10 (why) + CPT (what) = the complete picture of a claim.
# The insurer uses BOTH to determine how much to reimburse.
#
# Each code in our list:
#
#   "99213"  → Office Visit, Established Patient, Low-Moderate Complexity
#               The most commonly billed code in outpatient medicine.
#               A routine follow-up visit for a known patient (~15 minutes).
#
#   "99214"  → Office Visit, Established Patient, Moderate-High Complexity
#               A more detailed visit requiring more decision-making (~25 minutes).
#               Higher reimbursement than 99213.
#
#   "93000"  → Electrocardiogram (ECG / EKG) with Interpretation
#               Records electrical activity of the heart.
#               Commonly ordered for chest pain, hypertension, pre-surgery.
#
#   "85025"  → Complete Blood Count (CBC) with Differential
#               Standard blood test measuring red cells, white cells, platelets.
#               Ordered for almost every hospital admission and annual checkup.
#
#   "80053"  → Comprehensive Metabolic Panel (CMP)
#               14-test blood chemistry panel covering kidney, liver,
#               glucose, electrolytes. Extremely common lab order.
#
#   "71046"  → Chest X-Ray, 2 Views
#               Standard imaging for chest pain, cough, pneumonia screening.
#               One of the most frequently ordered radiology procedures.
#
#   "99232"  → Subsequent Hospital Inpatient Visit, Moderate Complexity
#               Billed each day a doctor visits a hospitalized patient.
#               Only appears on Inpatient claim types.
#
#   "36415"  → Routine Venipuncture (blood draw)
#               The actual act of drawing blood from a vein for lab tests.
#               One of the highest VOLUME procedure codes in any database.
#
#   "90834"  → Psychotherapy, 45 Minutes
#               Individual therapy session with a licensed therapist.
#               Paired with F32 (depression) or F41 (anxiety) diagnoses.
#
#   "97110"  → Therapeutic Exercise (Physical Therapy)
#               Strength and range-of-motion exercises supervised by a PT.
#               Paired with M54 (back pain) or M79 (soft tissue) diagnoses.
#
#   "43239"  → Upper GI Endoscopy with Biopsy
#               Camera down the throat to examine stomach and esophagus.
#               Paired with K21 (GERD) diagnoses. Higher cost procedure.
#
#   "66984"  → Extracapsular Cataract Removal with Lens Implant
#               One of the most common surgical procedures in the US.
#               Very high reimbursement rate. Paired with elderly members.


PLAN_TYPES = ["HMO","PPO","EPO","HDHP","POS"]
# The TYPE of insurance plan recorded on each policy in dim_policies.
# Plan type determines the RULES around which providers a member can use
# and how much they pay out of pocket.
#
#   "HMO"  → Health Maintenance Organization
#               MOST RESTRICTIVE but LOWEST PREMIUM.
#               Member MUST use only in-network providers.
#               MUST have a Primary Care Physician (PCP) who acts as gatekeeper.
#               MUST get a referral from PCP before seeing any specialist.
#               Going out-of-network = 100% out of pocket (not covered at all).
#               Best for: people who want low monthly costs and don't mind restrictions.
#
#   "PPO"  → Preferred Provider Organization
#               MOST FLEXIBLE but HIGHEST PREMIUM.
#               Can see ANY doctor, in-network OR out-of-network.
#               No need for referrals to see specialists.
#               In-network = lower cost. Out-of-network = higher cost but still covered.
#               Best for: people who travel, have existing specialist relationships,
#               or want maximum freedom of choice.
#
#   "EPO"  → Exclusive Provider Organization
#               HYBRID — like HMO for network rules, like PPO for referrals.
#               Must use in-network providers only (like HMO).
#               BUT no referral needed to see specialists (like PPO).
#               Going out-of-network = not covered except emergencies.
#               Best for: people who want flexibility within a network.
#
#   "HDHP" → High Deductible Health Plan
#               Very HIGH deductible (minimum $1,600/individual in 2024).
#               Very LOW monthly premium.
#               Member pays 100% of all costs until deductible is met.
#               After deductible → insurance kicks in normally.
#               KEY BENEFIT: Can be paired with an HSA (Health Savings Account)
#               allowing tax-free savings for medical expenses.
#               Best for: young healthy people who rarely need care,
#               or people who want to maximize HSA tax benefits.
#
#   "POS"  → Point of Service
#               HYBRID of HMO and PPO.
#               Has a PCP/gatekeeper like HMO.
#               Requires referrals for specialists like HMO.
#               BUT allows out-of-network care at higher cost like PPO.
#               Best for: people who want some flexibility but lower premiums than PPO.


PROVIDER_TYPES = ["Hospital","Clinic","Pharmacy","Physician","Specialist","Lab"]
# The category of healthcare provider recorded in dim_providers.
# Determines what kind of care they deliver and what claim types they generate.
#
#   "Hospital"    → Full inpatient and emergency facility.
#                   Generates the HIGHEST COST claims in the database.
#                   Source of all Inpatient and Emergency claim types.
#                   Has operating rooms, ICU, maternity ward etc.
#
#   "Clinic"      → Outpatient facility for routine and same-day care.
#                   No overnight stays. Source of most Outpatient claims.
#                   Examples: urgent care center, community health clinic.
#
#   "Pharmacy"    → Dispenses prescription medications.
#                   Source of all Pharmacy claim types.
#                   Claims are typically lower cost but very HIGH VOLUME.
#                   Every prescription refill = a separate claim.
#
#   "Physician"   → Individual primary care doctor (PCP).
#                   First point of contact for most members.
#                   Generates Outpatient and Preventive claims.
#                   For HMO members: this is the required gatekeeper.
#
#   "Specialist"  → Doctor focused on one medical area.
#                   Examples: cardiologist, dermatologist, orthopedist,
#                   psychiatrist, neurologist, oncologist.
#                   Generally higher cost than Physician visits.
#                   HMO members need a referral to see a Specialist.
#
#   "Lab"         → Diagnostic laboratory and imaging center.
#                   Performs blood tests, pathology, X-rays, MRI, CT scans.
#                   Does NOT treat patients — only runs tests.
#                   Very HIGH VOLUME of low-cost claims.


CLAIM_TYPES = ["Inpatient","Outpatient","Pharmacy","Emergency","Preventive"]
# The setting and context in which a claim was generated.
# Recorded on every row in fact_claims.
# Critical for cost analysis — different claim types have very different
# average costs and reimbursement rules.
#
#   "Inpatient"   → Patient was ADMITTED to a hospital and stayed overnight.
#                   HIGHEST COST claim type — can be tens of thousands of dollars.
#                   Has admission_date, discharge_date, and length_of_stay fields.
#                   ~20% of claims in this database are inpatient.
#                   Typical ICD-10 pairs: I25 (heart disease), I10 (hypertension crisis)
#
#   "Outpatient"  → Patient visited a provider and LEFT THE SAME DAY.
#                   Most common claim type by volume.
#                   Covers office visits, same-day surgery, outpatient procedures.
#                   Moderate cost range.
#                   Typical ICD-10 pairs: M54 (back pain), J06 (cold/flu)
#
#   "Pharmacy"    → Prescription drug dispensed by a pharmacy.
#                   HIGHEST VOLUME claim type — every refill is a separate claim.
#                   Usually lowest individual cost but adds up significantly.
#                   Typical ICD-10 pairs: E11 (diabetes), I10 (hypertension),
#                   F32 (depression) — all require ongoing medication.
#
#   "Emergency"   → Unplanned ER visit — patient could not wait for appointment.
#                   HIGH COST and HIGH COPAY even for in-network visits.
#                   For HMO plans: out-of-network ER IS covered (federal law).
#                   Typical ICD-10 pairs: J06 (severe infection), G43 (migraine attack)
#
#   "Preventive"  → Wellness visit, annual checkup, vaccine, or screening.
#                   SPECIAL RULE under the ACA (Affordable Care Act):
#                   Preventive care must be covered at $0 copay for in-network.
#                   Member pays NOTHING for a preventive visit.
#                   Typical ICD-10 pair: Z00 (general health examination)


CHANNELS = ["Online","Mobile App","Paper","Agent","Hospital Portal"]
# HOW the claim was SUBMITTED to the insurance company.
# This is about the technical submission METHOD, not where care was received.
# Recorded on every row in fact_claims as submission_channel.
#
#   "Online"          → Member or provider filed via the insurer's website.
#                       Fast, trackable, increasingly common.
#
#   "Mobile App"      → Filed through the insurer's smartphone application.
#                       Growing rapidly — most insurers now have apps.
#                       Members can photograph receipts and submit instantly.
#
#   "Paper"           → Old-school physical claim form mailed to the insurer.
#                       Slowest method — takes weeks to process.
#                       Still used by older providers and members.
#                       Higher administrative cost for the insurer.
#
#   "Agent"           → Submitted through a licensed insurance broker or agent.
#                       Common for complex claims or group contract submissions.
#                       The agent acts as an intermediary.
#
#   "Hospital Portal" → Filed directly through the hospital's own billing system
#                       which connects electronically to the insurer.
#                       Most common for large hospitals — automated and fast.
#                       Hospitals have entire billing departments for this.


CLAIM_STATUSES = ["Approved","Rejected","Pending","Under Review"]
# The current processing STATE of a claim in fact_claims.
# Assigned with weights: Approved 65% / Pending 15% / Under Review 10% / Rejected 10%
#
#   "Approved"      → Insurer accepted the claim and will pay.
#                     Triggers creation of a record in fact_payments.
#                     65% of claims — the expected normal outcome.
#                     Payment follows within 3–45 days (modeled in fact_payments).
#
#   "Rejected"      → Insurer DENIED the claim — no payment will be made.
#                     10% of claims. Common rejection reasons:
#                       - Service not covered under the policy
#                       - Out-of-network provider on an HMO plan
#                       - Wrong or missing diagnosis/procedure codes
#                       - Claim submitted after the deadline
#                       - Duplicate claim already processed
#                       - Pre-authorization was required but not obtained
#
#   "Pending"       → Claim has been received but not yet processed.
#                     15% of claims — sitting in the processing queue.
#                     Normal processing time: 30 days in most US states.
#                     No payment yet — waiting for review.
#
#   "Under Review"  → Claim has been flagged for MANUAL investigation.
#                     10% of claims. Triggered by:
#                       - is_fraud_flagged = True
#                       - Unusually high claim amount
#                       - Pattern of suspicious billing by a provider
#                       - Mismatch between diagnosis and procedure codes
#                     Payment is HELD until investigation completes.


GENDERS = ["Male","Female","Non-binary"]
# The gender options available for members in dim_members.
# Stored ONCE on the member record — never repeated on claims or policies
# (that would violate 3NF normalization rules).
#
#   "Male"        → biological male or male gender identity
#   "Female"      → biological female or female gender identity
#   "Non-binary"  → gender identity outside the male/female binary
#                   Included for inclusivity and to reflect modern
#                   insurance data collection standards.
#                   Small percentage of the overall member population.
#
# WHY GENDER MATTERS IN INSURANCE DATA:
#   - Risk scoring and premium calculation (historically)
#   - Clinical relevance — certain conditions more common by gender
#     (e.g. N39 UTIs more common in females, I25 heart disease more in males)
#   - Regulatory reporting requirements
#   - Note: Under the ACA, insurers CANNOT charge different premiums
#     based on gender for individual/family plans (but group plans vary)


EMPLOYMENT_CATS = ["Employed","Self-Employed","Unemployed","Retired","Student"]
# The employment situation of the member stored in dim_members.
# This is important context for understanding HOW a member gets their insurance
# and what their financial situation likely is.
#
#   "Employed"       → Works for a company.
#                      Most likely covered under a GROUP contract
#                      through their employer.
#                      Employer pays majority of premium.
#
#   "Self-Employed"  → Runs their own business or freelances.
#                      Cannot get employer-sponsored insurance.
#                      Must buy their own INDIVIDUAL or FAMILY contract
#                      through the marketplace or directly from insurer.
#                      Pays 100% of premium themselves (no employer contribution).
#                      Can deduct premiums as a business expense (tax benefit).
#
#   "Unemployed"     → Currently without a job.
#                      May be on COBRA — continuing their previous
#                      employer's group plan temporarily (very expensive).
#                      May qualify for Medicaid (government insurance).
#                      May buy a marketplace plan with subsidies.
#                      Higher financial risk for the insurer.
#
#   "Retired"        → No longer working.
#                      If 65+: likely on Medicare (federal government insurance).
#                      May ALSO have a supplemental private policy (Medigap)
#                      to cover what Medicare does not — this is why some
#                      retired members have dual coverage in our database.
#                      If under 65: must buy private insurance until Medicare eligible.
#
#   "Student"        → Enrolled in school/university.
#                      Under 26: can stay on parent's insurance plan (ACA rule).
#                      Over 26: must buy own plan — often university health plan
#                      or a low-cost individual marketplace plan.


# ═══════════════════════════════════════════════════════
# TABLE BUILDERS
# ═══════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. DIM_FAMILIES
def build_families(n: int) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the dim_families table.
    #   Represents household groups that are covered under
    #   a FAMILY type contract.
    #
    # PARAMETER:
    #   n : int → the number of family households to generate
    #             in our case n = N_FAMILIES = 2,000
    #
    # RETURNS:
    #   A pandas DataFrame with 2,000 rows and 5 columns.
    #   This DataFrame is then loaded into DuckDB as dim_families.
    #
    # IMPORTANT — WHAT A FAMILY IS AND IS NOT:
    #   ✓  A family is a HOUSEHOLD GROUP sharing one Family contract
    #   ✓  Every family has at least 2 members (family_size >= 2)
    #   ✗  A single person is NOT a family — they buy an Individual contract
    #   ✗  A group of employees is NOT a family — they are under a Group contract
    #
    # HOW dim_families CONNECTS TO THE REST OF THE DATABASE:
    #
    #   dim_families
    #       └── dim_contracts.family_id  (one Family contract per household)
    #               └── dim_policies.contract_id  (one policy for the whole family)
    #                       └── policy_members  (all family members linked here)
    #                               └── dim_members  (each person's details)
    #
    #   So dim_families sits at the TOP of the family coverage chain.
    #   It is a reference/lookup table — it does not store member details.
    #   Member details live in dim_members.
    # ═══════════════════════════════════════════════════════
    print(f"  Building dim_families        ({n:>6,} rows)...")
    return pd.DataFrame([{

        "family_id": f"FAM{i+1:06d}",
        # Unique identifier for this household group.
        # Format: FAM000001, FAM000002, ... FAM002000
        #   "FAM"      → prefix identifying this as a family record
        #   {i+1}      → counter starting at 1 (not 0)
        #   :06d       → always 6 digits, zero-padded on the left
        #                so FAM1 becomes FAM000001, not FAM1
        # This becomes the foreign key in dim_contracts:
        #   dim_contracts.family_id → dim_families.family_id
        # PRIMARY KEY of this table.

        "family_size": random.randint(2, 7),
        # How many people live in this household and share the family contract.
        # Range: 2 to 7 people.
        #   Minimum 2 → because a family must have at least 2 people.
        #               A single person buys Individual, not Family.
        #   Maximum 7 → realistic upper bound for a household size.
        #               Larger families exist but are uncommon.
        #
        # HOW family_size IS USED:
        #   In build_policies_and_bridge(), when processing a Family contract,
        #   we read family_size to know how many members to assign to this family:
        #
        #     fam_size = family_row["family_size"]  # e.g. 4
        #     → assign next 4 members from the member pool to this family
        #     → link all 4 to the same single policy
        #
        # IMPORTANT NOTE:
        #   family_size is the INTENDED size of the household.
        #   The actual number of members assigned may be less
        #   if the member pool runs out before all families are filled.

        "residential_zip": fake.zipcode(),
        # The ZIP code where this family lives.
        # Generated by Faker — produces realistic 5-digit US ZIP codes.
        # Example: "90210", "10001", "60601"
        #
        # WHY THIS MATTERS IN INSURANCE:
        #   - ZIP code determines which provider networks are available nearby
        #   - Affects premium pricing in some states (geographic rating)
        #   - Used for population health analysis:
        #     "Which ZIP codes have the highest claim rates?"
        #     "Are members in rural ZIPs using fewer preventive services?"
        #   - Regulators use it to check for redlining
        #     (unfairly denying coverage in certain areas)

        "state": fake.state_abbr(),
        # The US state where this family lives.
        # Generated by Faker — produces valid 2-letter state abbreviations.
        # Examples: "CA", "TX", "NY", "FL"
        #
        # WHY THIS MATTERS IN INSURANCE:
        #   Insurance is regulated STATE BY STATE in the US — not federally.
        #   Each state has its own rules about:
        #     - Minimum coverage requirements
        #     - Maximum deductible limits
        #     - Which pre-existing conditions must be covered
        #     - How much premiums can vary by age or location
        #   So state is critical for compliance reporting and
        #   understanding regulatory exposure.

        "income_category": random.choice(["Low", "Middle", "High"]),
        # Broad income bracket for this household.
        # Assigned randomly with equal probability (1/3 each).
        #
        # WHY THIS MATTERS IN INSURANCE:
        #   Income affects:
        #     - Eligibility for government subsidies (ACA marketplace subsidies)
        #     - Medicaid eligibility (Low income families may qualify)
        #     - Ability to pay premiums and out-of-pocket costs
        #     - Plan choice — Low income families more likely to choose
        #       HDHP (low premium) even if the high deductible is risky for them
        #
        # ANALYTICS USE CASES:
        #     "Do Low income families have higher claim rejection rates?"
        #     "Do High income families choose PPO plans more often?"
        #     "Is there a correlation between income and health_risk_score?"

    } for i in range(n)])
    # ── LIST COMPREHENSION:
    # [ {...} for i in range(2000) ]
    # → creates a list of 2,000 dictionaries, one per family
    # → each dictionary has 5 keys (the columns)
    # → pd.DataFrame() converts the list of dicts into a table
    # → result: a 2,000 row × 5 column DataFrame

# ─────────────────────────────────────────────
# 2. DIM_EMPLOYERS
def build_employers(n: int) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the dim_employers table.
    #   Represents companies that purchase a GROUP contract
    #   from the insurer to cover all of their employees.
    #
    # PARAMETER:
    #   n : int → the number of employer companies to generate
    #             in our case n = N_EMPLOYERS = 50
    #
    # RETURNS:
    #   A pandas DataFrame with 50 rows and 5 columns.
    #   This DataFrame is then loaded into DuckDB as dim_employers.
    #
    # WHAT AN EMPLOYER IS IN THIS DATABASE:
    #   An employer is a COMPANY that acts as the BUYER of a Group contract.
    #   Instead of each employee going individually to the insurer,
    #   the company negotiates ONE group contract on behalf of ALL employees.
    #
    #   Real world analogy:
    #     Individual buying → one person walks into a store and buys one item
    #     Group buying      → a company negotiates a bulk deal for all employees
    #                         better rates, shared risk, employer pays most of it
    #
    # HOW dim_employers CONNECTS TO THE REST OF THE DATABASE:
    #
    #   dim_employers
    #       └── dim_contracts.employer_id  (one Group contract per company)
    #               └── dim_policies.contract_id  (one policy per employee)
    #                       └── policy_members  (employee + their dependents)
    #                               └── dim_members  (each person's details)
    #
    #   IMPORTANT DIFFERENCE FROM dim_families:
    #     dim_families  → one contract  → one policy  → all family members
    #     dim_employers → one contract  → MANY policies (one per employee)
    #                     because each employee independently chooses
    #                     Individual or Family coverage level
    #
    # REAL WORLD EMPLOYER INSURANCE FACTS:
    #   - Employer typically pays 70–80% of the monthly premium
    #   - Employee pays remaining 20–30% via paycheck deduction
    #   - All employees share the SAME group contract_id
    #   - But each employee gets their OWN policy with their chosen plan
    #   - Losing your job = losing your insurance
    #     (COBRA allows you to continue temporarily but YOU pay 100%)
    # ═══════════════════════════════════════════════════════
    print(f"  Building dim_employers       ({n:>6,} rows)...")
    return pd.DataFrame([{

        "employer_id": f"EMP{i+1:05d}",
        # Unique identifier for this employer company.
        # Format: EMP00001, EMP00002, ... EMP00050
        #   "EMP"    → prefix identifying this as an employer record
        #   {i+1}    → counter starting at 1 (not 0)
        #   :05d     → always 5 digits, zero-padded on the left
        #
        # This becomes the foreign key in dim_contracts:
        #   dim_contracts.employer_id → dim_employers.employer_id
        # PRIMARY KEY of this table.
        # Only 50 values exist — EMP00001 through EMP00050.

        "employer_name": fake.company(),
        # The name of the company, generated by Faker.
        # Produces realistic sounding company names.
        # Examples: "Smith & Associates", "TechCore Solutions", "Global Dynamics LLC"
        #
        # In a real insurance database this would be the LEGAL company name
        # exactly as registered — because the group contract is a legal agreement
        # between the insurer and the company as a legal entity.

        "industry": random.choice(["Healthcare","Finance","Tech",
                                   "Education","Retail","Manufacturing"]),
        # The business sector this company operates in.
        # Randomly chosen from 6 common industries with equal probability.
        #
        # WHY INDUSTRY MATTERS IN INSURANCE:
        #   Different industries have very different health risk profiles:
        #
        #   Healthcare      → workers exposed to illness, high stress, shift work
        #                     higher rates of burnout, back injuries, infections
        #
        #   Finance         → high stress, sedentary work
        #                     higher rates of hypertension (I10), anxiety (F41)
        #
        #   Tech            → sedentary work, screen exposure, high stress
        #                     higher rates of back pain (M54), depression (F32)
        #
        #   Education       → exposure to childhood illnesses, moderate stress
        #                     more respiratory infections (J06)
        #
        #   Retail          → physical work, standing all day, low wages
        #                     higher rates of musculoskeletal issues (M79, M54)
        #
        #   Manufacturing   → physical labor, machinery exposure, injury risk
        #                     higher rates of workplace injuries, back pain
        #
        # ANALYTICS USE CASES:
        #   "Which industry generates the highest average claim amount?"
        #   "Do Tech employees use more mental health services (F32, F41)?"
        #   "Do Manufacturing employees have more Emergency claim types?"

        "employee_count": random.randint(50, 10_000),
        # The total number of employees at this company.
        # Range: 50 to 10,000 employees.
        #   Minimum 50  → small companies below 50 often cannot afford
        #                 group insurance or are not required to offer it
        #   Maximum 10,000 → large enterprise employer
        #
        # IMPORTANT DISTINCTION:
        #   employee_count = total employees at the company in reality
        #   This is DIFFERENT from how many members are assigned to this
        #   employer in our database.
        #
        #   In our simulation:
        #     Each employer gets 10–80 employees assigned (random.randint(10,80))
        #     in build_policies_and_bridge() — much less than employee_count.
        #     This is intentional — we are modeling a SAMPLE of employees,
        #     not the entire company workforce.
        #     employee_count here represents the company SIZE for context,
        #     not the exact number of members in our database.
        #
        # WHY COMPANY SIZE MATTERS IN INSURANCE:
        #   Large employers (500+ employees) → more negotiating power
        #       → better rates from insurers → lower premiums per employee
        #   Small employers (50–100 employees) → less negotiating power
        #       → higher premiums, fewer plan options
        #   This is why large companies can offer better benefits than small ones.

        "state": fake.state_abbr(),
        # The US state where this company is headquartered.
        # Generated by Faker — produces valid 2-letter state abbreviations.
        # Examples: "CA", "TX", "NY", "IL"
        #
        # WHY STATE MATTERS FOR EMPLOYER INSURANCE:
        #   The company's state determines which insurance regulations apply
        #   to the GROUP CONTRACT they sign.
        #
        #   Additionally:
        #   - Multi-state employers have more complex coverage requirements
        #     (employees in different states may need different networks)
        #   - Some states mandate additional benefits beyond federal minimums
        #     Example: New York mandates infertility coverage
        #              California mandates broader mental health coverage
        #   - State taxes on insurance premiums vary
        #
        # NOTE:
        #   The employer state may differ from the employee's residential state
        #   (dim_members.state) — especially for remote workers.
        #   A company headquartered in TX may have employees living in CA.
        #   This creates interesting analytics opportunities:
        #   "How many members live in a different state from their employer?"

    } for i in range(n)])
    # ── LIST COMPREHENSION:
    # [ {...} for i in range(50) ]
    # → creates a list of 50 dictionaries, one per employer
    # → each dictionary has 5 keys (the columns)
    # → pd.DataFrame() converts the list into a table
    # → result: a 50 row × 5 column DataFrame
# ─────────────────────────────────────────────
# 3. DIM_MEMBERS
def build_members(n: int) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the dim_members table.
    #   Represents every single human being covered by any
    #   insurance in this database — regardless of whether
    #   they came through an Individual, Family, or Group contract.
    #
    # PARAMETER:
    #   n : int → the number of members to generate
    #             in our case n = N_MEMBERS = 8,000
    #
    # RETURNS:
    #   A pandas DataFrame with 8,000 rows and 13 columns.
    #   This DataFrame is then loaded into DuckDB as dim_members.
    #
    # THE MOST IMPORTANT DESIGN PRINCIPLE IN v3:
    #   member_id is the UNIVERSAL identifier for every person.
    #   It does not matter HOW they got their insurance:
    #     - John bought Individual coverage for himself    → has a member_id
    #     - Mary is a dependent on her husband's Family plan → has a member_id
    #     - Tommy is an employee on a Group plan           → has a member_id
    #     - Lisa is Tommy's dependent child                → has a member_id
    #   Every single human being = one row = one member_id.
    #   No exceptions.
    #
    # KEY CHANGE FROM v1 AND v2:
    #   ✗  NO insurance_category column  → belongs to dim_contracts
    #   ✗  NO family_id column           → belongs to dim_contracts
    #   ✗  NO employer_id column         → belongs to dim_contracts
    #   ✗  NO premium_amount column      → belongs to dim_policies
    #   ✗  NO coverage_status column     → belongs to dim_policies
    #   ✗  NO benefit_limit column       → belongs to dim_policies
    #
    #   WHY WERE THESE REMOVED?
    #   Because they describe the CONTRACT or POLICY — not the PERSON.
    #   A person's identity and health profile do not change
    #   depending on what insurance plan they are on.
    #   Storing contract details on the member would violate 3NF:
    #     If John switches from HMO to PPO → we would have to update
    #     his row in dim_members. But his name, birthday, gender
    #     did not change — only his plan did.
    #     In v3: only dim_policies changes. dim_members stays untouched.
    #
    # WHAT dim_members STORES IN v3 — THREE CATEGORIES:
    #   1. Pure demographics    → who this person IS
    #   2. Health profile       → this person's medical characteristics
    #   3. Financial info       → this person's personal financial account
    #
    # HOW dim_members CONNECTS TO THE REST OF THE DATABASE:
    #
    #   dim_members
    #       ← policy_members.member_id   (what policy covers this person)
    #               ← fact_claims.member_id      (what claims this person filed)
    #                       ← fact_payments.member_id    (what payments were made)
    #
    #   dim_members is the ENDPOINT of the coverage chain.
    #   Everything flows TO it — it has no foreign keys of its own.
    # ═══════════════════════════════════════════════════════
    print(f"  Building dim_members         ({n:>6,} rows)...")
    rows = []
    for i in range(n):
        rows.append({

            "member_id": f"MEM{i+1:07d}",
            # Unique identifier for this individual person.
            # Format: MEM0000001, MEM0000002, ... MEM0008000
            #   "MEM"    → prefix identifying this as a member record
            #   {i+1}    → counter starting at 1 (not 0)
            #   :07d     → always 7 digits, zero-padded on the left
            #
            # This is the MOST REFERENCED key in the entire database:
            #   policy_members.member_id → here  (who is covered)
            #   fact_claims.member_id    → here  (who filed a claim)
            #   fact_payments.member_id  → here  (who received payment)
            #
            # PRIMARY KEY of this table.
            # Every human being in the database has exactly one member_id
            # that never changes regardless of what plan they are on.

            # ── CATEGORY 1: PURE DEMOGRAPHICS ─────────────────────────
            # These describe WHO this person IS.
            # They never change based on insurance plan.
            # Stored ONCE here — never repeated anywhere else (3NF rule).

            "first_name": fake.first_name(),
            # The person's first name, generated by Faker.
            # Examples: "James", "Maria", "David", "Sarah"
            # Faker produces culturally diverse names reflecting
            # a realistic US population distribution.

            "last_name": fake.last_name(),
            # The person's last name, generated by Faker.
            # Examples: "Smith", "Johnson", "Garcia", "Kim"
            # In a real database, family members would share a last name.
            # In our simulation, names are assigned independently —
            # a known simplification of the simulated data.

            "date_of_birth": rand_date(date(1940,1,1), date(2005,12,31)),
            # The person's date of birth — a random date between:
            #   Earliest: January 1, 1940  → person would be ~85 years old
            #   Latest:   December 31, 2005 → person would be ~19 years old
            #
            # WHY THESE BOUNDS?
            #   1940 → oldest realistic insured person still living
            #   2005 → youngest adult (18+) who can be a policyholder
            #          Children younger than 18 can be dependents but
            #          in our simulation all members are treated as adults
            #          for simplicity.
            #
            # WHY date_of_birth INSTEAD OF age?
            #   Age is a DERIVED value — it changes every year.
            #   Storing age directly would violate 3NF because:
            #     age = today - date_of_birth (it can always be calculated)
            #   Storing a calculated value that changes over time
            #   means you would have to update the database every year.
            #   date_of_birth is a FIXED FACT — it never changes.
            #   Age is always calculated on the fly:
            #     SELECT DATEDIFF('year', date_of_birth, CURRENT_DATE) AS age
            #
            # WHY AGE MATTERS IN INSURANCE:
            #   Age is the single strongest predictor of healthcare costs.
            #   Older members generate significantly more and costlier claims.
            #   Under the ACA, insurers can charge older members up to
            #   3x more than younger members (3:1 age rating ratio).

            "gender": random.choice(GENDERS),
            # The person's gender identity.
            # Randomly chosen from: ["Male", "Female", "Non-binary"]
            # Equal probability for each option.
            #
            # Stored ONCE here — never copied to claims or policies (3NF).
            # Retrieved via JOIN when needed for analysis:
            #   SELECT m.gender, COUNT(c.claim_id)
            #   FROM fact_claims c
            #   JOIN dim_members m ON c.member_id = m.member_id
            #   GROUP BY m.gender

            "marital_status": random.choice(["Single","Married",
                                             "Divorced","Widowed"]),
            # The person's current marital status.
            # Randomly chosen with equal probability from 4 options.
            #
            # WHY THIS MATTERS IN INSURANCE:
            #   Married members often have access to spouse's plan
            #   → explains dual coverage (secondary policies in policy_members)
            #   Divorced members may lose coverage from spouse's plan
            #   → may need to buy their own Individual contract
            #   Widowed members may lose coverage from deceased spouse's plan
            #   → similar situation to divorced

            "zip_code": fake.zipcode(),
            # The person's residential ZIP code.
            # Generated by Faker — realistic 5-digit US ZIP codes.
            #
            # NOTE: This may differ from their employer's state (dim_employers.state)
            # and even from their family's ZIP (dim_families.residential_zip).
            # This reflects reality — a person may live in a different ZIP
            # from where they grew up or where their employer is headquartered.
            #
            # Used for:
            #   - Provider network availability analysis
            #   - Geographic health outcome studies
            #   - Premium geographic rating factors

            "state": fake.state_abbr(),
            # The US state where this person lives.
            # Determines which state insurance regulations protect them.
            # May differ from employer state for remote workers.

            "employment_category": random.choice(EMPLOYMENT_CATS),
            # The person's current employment situation.
            # Randomly chosen from:
            #   ["Employed","Self-Employed","Unemployed","Retired","Student"]
            #
            # Provides context for HOW they likely obtained their insurance.
            # Note: this is DESCRIPTIVE information about the person —
            # it does not determine their contract type in the database.
            # Contract type is determined by dim_contracts.contract_type.
            # A person can be "Employed" but still have an Individual contract
            # if their employer does not offer group insurance.

            # ── CATEGORY 2: HEALTH PROFILE ────────────────────────────
            # These describe this person's medical characteristics.
            # They belong on the PERSON — not on the claim or policy.
            # A person's health profile exists independently of
            # whether they ever file a claim.

            "chronic_condition_flag": random.random() < 0.25,
            # Boolean flag — True if this person has a chronic health condition.
            # 25% of members are flagged as having a chronic condition.
            # This reflects real US statistics:
            #   ~60% of US adults have at least one chronic condition
            #   ~40% have two or more
            #   We use 25% as a conservative estimate for significant conditions.
            #
            # Examples of chronic conditions in our ICD10_CODES:
            #   I10 (hypertension), E11 (diabetes), J45 (asthma),
            #   F32 (depression), I25 (heart disease)
            #
            # ANALYTICS USE CASES:
            #   "Do members with chronic_condition_flag = True
            #    generate significantly more claims?"
            #   "What is the average claim_amount for chronic vs non-chronic members?"
            #   "Which plan types are most common among chronic condition members?"

            "health_risk_score": round(np.random.beta(2, 5) * 100, 2),
            # A numeric score from 0 to 100 representing overall health risk.
            # Higher score = higher risk = more likely to generate costly claims.
            #
            # Generated using a BETA DISTRIBUTION with parameters alpha=2, beta=5:
            #   np.random.beta(2, 5) → produces values between 0 and 1
            #   × 100               → scales to 0–100 range
            #   round(..., 2)       → keeps 2 decimal places
            #
            # WHY BETA(2,5) AND NOT RANDOM.UNIFORM?
            #   Beta(2,5) produces a RIGHT-SKEWED distribution:
            #   → Most members cluster toward LOWER risk scores (0–40)
            #   → Few members have very HIGH risk scores (70–100)
            #   This matches reality — most people are relatively healthy,
            #   a small number of high-risk members drive most of the costs.
            #
            #   Visual shape of Beta(2,5):
            #   Frequency
            #     │▓▓
            #     │▓▓▓▓
            #     │▓▓▓▓▓▓▓
            #     │▓▓▓▓▓▓▓▓▓▓▓░░
            #     └─────────────────→ Score
            #     0    20   40   60   80  100
            #
            # Used by insurers for:
            #   - Predicting future claim costs
            #   - Setting appropriate premium levels
            #   - Identifying high-risk members for care management programs

            "smoking_status": random.choice(["Never","Former","Current"]),
            # Whether this person smokes tobacco.
            # Randomly chosen with equal probability from 3 options.
            #
            # WHY THIS MATTERS IN INSURANCE:
            #   Smoking is the single largest preventable cause of disease.
            #   Under the ACA, insurers CAN charge smokers up to 50% more
            #   in premiums than non-smokers — one of the few demographic
            #   factors where premium discrimination is still legal.
            #
            #   Smoking strongly correlates with:
            #     I25 (heart disease), J45 (asthma), E11 (diabetes complications)
            #     and many cancers not in our simplified ICD10 list
            #
            # ANALYTICS USE CASES:
            #   "Do Current smokers have higher average claim_amount?"
            #   "Is smoking_status correlated with health_risk_score?"
            #   "Do smokers choose HDHP plans more often (lower premium priority)?"

            "bmi": round(random.gauss(27, 5), 1),
            # Body Mass Index — a measure of body fat based on
            # height and weight. Stored as a decimal number.
            # Generated using a GAUSSIAN (normal) distribution:
            #   random.gauss(mean=27, std=5)
            #   → bell curve centered at 27
            #   → standard deviation of 5
            #   → most values fall between 17 and 37
            #   round(..., 1) → one decimal place
            #
            # WHY MEAN = 27?
            #   The average BMI of US adults is ~26–28 (overweight range).
            #   WHO classification:
            #     < 18.5  → Underweight
            #     18.5–24.9 → Normal weight
            #     25–29.9   → Overweight
            #     ≥ 30      → Obese
            #
            # WHY GAUSSIAN AND NOT UNIFORM?
            #   BMI in a real population follows a roughly normal distribution
            #   centered around the population average.
            #   Most people are near the average, fewer at the extremes.
            #
            # WHY BMI MATTERS IN INSURANCE:
            #   High BMI (≥30) strongly correlates with:
            #     E11 (diabetes), I10 (hypertension), I25 (heart disease),
            #     M54 (back pain), E66 (obesity — directly coded)
            #   High BMI members generate significantly more claims
            #   and higher costs over time.

            # ── CATEGORY 3: FINANCIAL INFO ────────────────────────────
            # Personal financial account belonging to this individual.

            "msa_balance": round(random.uniform(0, 5_000), 2),
            # Medical Savings Account balance for this member.
            # A random dollar amount between $0 and $5,000.
            #
            # WHAT IS AN MSA / HSA?
            #   MSA = Medical Savings Account (older term)
            #   HSA = Health Savings Account (modern term, same concept)
            #   A special tax-advantaged bank account that members use
            #   to save money specifically for medical expenses.
            #
            # HOW IT WORKS:
            #   Member deposits pre-tax dollars into the account.
            #   They use this money to pay deductibles, copays, and
            #   any expenses not covered by insurance.
            #   Unused money rolls over to the next year (unlike FSA).
            #   Only available to members on HDHP plans.
            #
            # WHY IT BELONGS ON dim_members AND NOT dim_policies:
            #   The MSA/HSA balance belongs to the PERSON — not the plan.
            #   If John switches insurance plans, his HSA money stays with him.
            #   It is a personal financial asset, not a plan feature.
            #   This is why it lives in dim_members in v3.

        })
    return pd.DataFrame(rows)
    # ── LOOP vs LIST COMPREHENSION:
    # build_members uses a for loop with rows.append() instead of
    # a list comprehension like build_families and build_employers.
    # Both produce identical results — a list of dictionaries.
    # The loop style is used here because the member builder is more
    # complex and may be extended with conditional logic more easily
    # in a loop format than a compressed list comprehension.





# ─────────────────────────────────────────────
# 4. DIM_PROVIDERS
# ─────────────────────────────────────────────

def build_providers(n: int) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the dim_providers table.
    #   Represents every healthcare entity that CAN DELIVER
    #   medical care and submit claims to the insurer.
    #
    # PARAMETER:
    #   n : int → the number of providers to generate
    #             in our case n = N_PROVIDERS = 300
    #
    # RETURNS:
    #   A pandas DataFrame with 300 rows and 8 columns.
    #   This DataFrame is then loaded into DuckDB as dim_providers.
    #
    # WHAT A PROVIDER IS:
    #   A provider is any licensed healthcare entity that:
    #     1. Delivers medical care to a member
    #     2. Submits a claim to the insurer requesting reimbursement
    #     3. Gets paid by the insurer (if the claim is approved)
    #
    #   Providers are NOT:
    #     ✗  The person receiving care (that is dim_members)
    #     ✗  The insurance company itself
    #     ✗  The submission channel (Online/Paper/Mobile App etc.)
    #
    # THE FLOW — where providers fit:
    #
    #   Member gets sick
    #       → visits a Provider  ← dim_providers lives here
    #           → Provider delivers care
    #               → Provider submits a Claim  ← fact_claims links to provider_id
    #                   → Insurer reviews claim
    #                       → Payment goes to Provider  ← fact_payments
    #
    # HOW dim_providers CONNECTS TO THE REST OF THE DATABASE:
    #
    #   dim_providers
    #       ← fact_claims.provider_id
    #               (every claim references which provider delivered the care)
    #
    #   dim_providers has NO foreign keys of its own.
    #   It is a pure reference/lookup table.
    #   It is referenced BY fact_claims — not the other way around.
    #
    # KEY ANALYTICS THIS TABLE ENABLES:
    #   "Which provider type generates the highest average claim amount?"
    #   "What % of claims are filed by out-of-network providers?"
    #   "Which providers have the highest fraud flag rates?"
    #   "Do Hospital claims have longer length_of_stay than Clinic claims?"
    # ═══════════════════════════════════════════════════════
    print(f"  Building dim_providers       ({n:>6,} rows)...")
    return pd.DataFrame([{

        "provider_id": f"PRV{i+1:06d}",
        # Unique identifier for this healthcare provider.
        # Format: PRV000001, PRV000002, ... PRV000300
        #   "PRV"    → prefix identifying this as a provider record
        #   {i+1}    → counter starting at 1 (not 0)
        #   :06d     → always 6 digits, zero-padded on the left
        #
        # This becomes the foreign key in fact_claims:
        #   fact_claims.provider_id → dim_providers.provider_id
        # PRIMARY KEY of this table.
        # Only 300 values exist — meaning all 50,000 claims are spread
        # across just 300 providers → ~167 claims per provider on average.

        "provider_name": fake.company() + " Medical",
        # The name of the healthcare provider, generated by Faker.
        # Faker generates a company name and we append " Medical"
        # to make it sound like a healthcare entity.
        # Examples:
        #   "Smith & Associates Medical"
        #   "Pacific Solutions Medical"
        #   "Anderson Group Medical"
        #
        # In a real insurance database, this would be the LEGAL name
        # of the provider as registered with the state medical board
        # and the insurer's credentialing department.
        # Providers must be credentialed (verified and approved) by
        # the insurer before they can submit claims.

        "provider_type": random.choice(PROVIDER_TYPES),
        # The CATEGORY of healthcare entity this provider is.
        # Randomly chosen from:
        #   ["Hospital","Clinic","Pharmacy","Physician","Specialist","Lab"]
        # Equal probability for each type.
        #
        # This is one of the MOST IMPORTANT fields for analytics because
        # provider_type strongly predicts claim cost and claim type:
        #
        #   Hospital   → Inpatient + Emergency claims → HIGHEST cost
        #                Has overnight beds, ICU, OR, maternity ward
        #                A single hospital admission can cost $10,000–$100,000+
        #
        #   Clinic     → Outpatient claims → MODERATE cost
        #                Same-day visits, no overnight stays
        #                Urgent care centers, community health clinics
        #
        #   Pharmacy   → Pharmacy claims → LOW individual cost, HIGH volume
        #                Every prescription refill = one claim
        #                A member with diabetes may generate 12+ pharmacy
        #                claims per year for insulin and metformin alone
        #
        #   Physician  → Outpatient + Preventive claims → LOW-MODERATE cost
        #                Individual primary care doctor
        #                For HMO members: REQUIRED gatekeeper for all care
        #                Annual checkup (Z00) always goes through Physician
        #
        #   Specialist → Outpatient claims → MODERATE-HIGH cost
        #                Doctor focused on one area of medicine
        #                Examples: cardiologist, dermatologist, psychiatrist
        #                HMO members need a referral from Physician first
        #                Specialist visits cost more than Physician visits
        #
        #   Lab        → Outpatient claims → LOW-MODERATE cost
        #                Does NOT treat patients — only runs diagnostic tests
        #                Blood draws (36415), X-rays (71046), MRIs
        #                Very HIGH VOLUME — almost every hospital visit
        #                generates a separate lab claim
        #
        # STORED HERE ONLY — 3NF RULE:
        #   provider_type lives in dim_providers and NOWHERE ELSE.
        #   In v1 of this database, provider_type was copied onto fact_claims.
        #   That was a normalization violation — fixed in v2 and v3.
        #   To get provider_type for a claim, you must JOIN:
        #     fact_claims → dim_providers → provider_type

        "state": fake.state_abbr(),
        # The US state where this provider is located.
        # Generated by Faker — valid 2-letter state abbreviations.
        #
        # WHY PROVIDER STATE MATTERS:
        #   Medical licenses are issued per state — a doctor licensed
        #   in California cannot legally practice in Texas without
        #   a separate Texas license.
        #
        #   Provider state vs Member state creates important scenarios:
        #     Same state    → normal in-state care
        #     Different state → out-of-state care (common for:
        #                       travel emergencies, specialist referrals,
        #                       members near state borders)
        #
        # ANALYTICS USE CASES:
        #   "How often do members receive care in a different state?"
        #   "Which states have the highest concentration of specialists?"
        #   "Do out-of-state claims have higher rejection rates?"

        "zip_code": fake.zipcode(),
        # The ZIP code where this provider is physically located.
        # Used for geographic proximity analysis:
        #   "How far does the average member travel to their provider?"
        #   (calculated using member zip_code vs provider zip_code)
        #
        # In a real insurer's database, ZIP code is used to:
        #   - Build provider directories ("find a doctor near me")
        #   - Ensure adequate network coverage in each geographic area
        #     (regulators require minimum provider-to-member ratios per region)
        #   - Calculate travel burden for rural members

        "network_flag": random.random() < 0.75,
        # Boolean — True if this provider is IN the insurer's network.
        # 75% of providers are in-network, 25% are out-of-network.
        #
        # THIS IS THE SINGLE MOST IMPACTFUL FIELD IN dim_providers
        # for determining what a member pays out of pocket.
        #
        # IN-NETWORK (network_flag = True) — ~225 of 300 providers:
        #   → Provider has signed a CONTRACT with the insurer
        #   → Pre-negotiated rates apply (insurer pays a set amount)
        #   → Member pays only their copay/deductible share
        #   → Example: $500 bill → insurer pays $400 → member pays $100
        #
        # OUT-OF-NETWORK (network_flag = False) — ~75 of 300 providers:
        #   → No contract with the insurer
        #   → Provider charges whatever they want (no rate agreement)
        #   → Insurer pays little or nothing
        #   → Member pays the majority or ALL of the bill
        #   → Example: $500 bill → insurer pays $0 → member pays $500
        #
        # PLAN TYPE INTERACTION:
        #   HMO  → out-of-network = 0% covered (except emergencies)
        #   EPO  → out-of-network = 0% covered (except emergencies)
        #   PPO  → out-of-network = partially covered (at higher cost)
        #   POS  → out-of-network = partially covered (with referral)
        #   HDHP → depends on underlying network structure
        #
        # NOTE: network_flag here is the PROVIDER'S network status.
        #   fact_claims also has network_provider_flag per claim —
        #   that records whether THIS SPECIFIC CLAIM used an in-network
        #   provider at the time of service. Both fields should ideally
        #   match, but in real data they can differ due to:
        #     - Provider joined/left network between claim date and DB update
        #     - Emergency override rules
        #     - Balance billing disputes

        "accreditation_level": random.choice(["Level1","Level2","Level3"]),
        # The quality/accreditation tier of this provider.
        # Randomly chosen with equal probability from 3 levels.
        #
        # In real insurance systems, accreditation is granted by
        # independent organizations like:
        #   - The Joint Commission (hospitals)
        #   - NCQA (health plans and medical groups)
        #   - URAC (utilization review)
        #
        # What each level means in our simulation:
        #   Level1 → Highest accreditation — meets all quality standards
        #            Top-tier hospitals, academic medical centers
        #            Lowest complication rates, best outcomes
        #
        #   Level2 → Standard accreditation — meets minimum requirements
        #            Most community hospitals and clinics fall here
        #            Adequate quality for routine care
        #
        #   Level3 → Basic or provisional accreditation
        #            Newer providers or those under review
        #            May have some quality concerns
        #
        # ANALYTICS USE CASES:
        #   "Do Level1 providers have lower claim rejection rates?"
        #   "Is there a correlation between accreditation and claim amount?"
        #   "Do members with chronic conditions prefer Level1 providers?"

        "established_year": random.randint(1970, 2020),
        # The year this provider was founded or began operating.
        # A random integer between 1970 and 2020.
        #
        # WHY THIS MATTERS:
        #   Provider age is a proxy for experience and stability:
        #   Older providers (established pre-1990):
        #     → Longer track record with the insurer
        #     → More established billing practices
        #     → Lower likelihood of fraudulent billing patterns
        #
        #   Newer providers (established post-2010):
        #     → Less history to verify against
        #     → Higher scrutiny during credentialing
        #     → More likely to appear in fraud investigations
        #
        # ANALYTICS USE CASES:
        #   "Is there a correlation between established_year
        #    and is_fraud_flagged rate on claims?"
        #   "Do newer providers have higher claim rejection rates?"

    } for i in range(n)])
    # ── LIST COMPREHENSION:
    # [ {...} for i in range(300) ]
    # → creates a list of 300 dictionaries, one per provider
    # → each dictionary has 8 keys (the columns)
    # → pd.DataFrame() converts the list into a table
    # → result: a 300 row × 8 column DataFrame




    
# ─────────────────────────────────────────────
# 5. DIM_CONTRACTS  ← NEW TABLE
#    The BUYING AGREEMENT between the insurer and the buyer.
#    Answers: WHO is buying coverage and at what level?
#
#    contract_type = "Individual"
#        → one person bought coverage for themselves
#        → employer_id = NULL, family_id = NULL
#
#    contract_type = "Family"
#        → one person bought coverage for their whole household
#        → employer_id = NULL
#        → family_id = the household this contract covers
#
#    contract_type = "Group"
#        → a company bought coverage for all its employees
#        → employer_id = the company
#        → family_id = NULL (families handled at policy level)
# ─────────────────────────────────────────────
def build_contracts(families: pd.DataFrame,
                    employers: pd.DataFrame,
                    n_individual: int) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the dim_contracts table.
    #   Represents the BUYING AGREEMENT between the insurer
    #   and whoever is purchasing the insurance coverage.
    #
    # THIS IS THE NEW TABLE IN v3 — did not exist in v1 or v2.
    #
    # PARAMETERS:
    #   families     : pd.DataFrame → the dim_families table
    #                  used to create one Family contract per household
    #   employers    : pd.DataFrame → the dim_employers table
    #                  used to create one Group contract per company
    #   n_individual : int → number of Individual contracts to create
    #                  in our case = N_MEMBERS // 3 = 8,000 // 3 = 2,666
    #
    # RETURNS:
    #   A pandas DataFrame loaded into DuckDB as dim_contracts.
    #   Total rows = n_individual + N_FAMILIES + N_EMPLOYERS
    #              = 2,666 + 2,000 + 50
    #              = 4,716 contracts total
    #
    # CONTRACT vs POLICY — THE KEY DISTINCTION IN v3:
    #
    #   dim_contracts answers: WHO is buying? and at what LEVEL?
    #     → Individual person buying for themselves
    #     → A family buying for their whole household
    #     → A company buying for all its employees
    #
    #   dim_policies answers: WHAT exact plan did they choose?
    #     → HMO or PPO or HDHP?
    #     → What deductible?
    #     → What premium?
    #     → What out-of-pocket maximum?
    #
    #   Real world analogy:
    #     CONTRACT = deciding to buy a car and signing the purchase agreement
    #     POLICY   = choosing the exact model, color, and features of that car
    #
    #   WHY SEPARATE THEM?
    #     A Group contract covers 50 employees.
    #     Each of those 50 employees chooses a DIFFERENT policy
    #     (different plan type, different coverage level).
    #     If we stored everything in one table:
    #       → we would repeat the contract details 50 times (one per employee)
    #       → that violates 3NF (same contract info duplicated 50 times)
    #     With two separate tables:
    #       → contract details stored ONCE in dim_contracts
    #       → each employee's plan stored separately in dim_policies
    #       → dim_policies.contract_id links back to dim_contracts
    #
    # THREE CONTRACT TYPES AND THEIR RULES:
    #
    #   "Individual":
    #     → Buyer   = one person purchasing for themselves
    #     → Covers  = just that one person
    #     → Count   = n_individual = ~2,666 contracts
    #     → family_id   = NULL (no household group involved)
    #     → employer_id = NULL (no company involved)
    #     → Generates exactly 1 policy in dim_policies
    #
    #   "Family":
    #     → Buyer   = one person (policyholder) purchasing for their household
    #     → Covers  = policyholder + all dependents in that household
    #     → Count   = 2,000 contracts (one per family in dim_families)
    #     → family_id   = FAM000001 ... FAM002000 (links to dim_families)
    #     → employer_id = NULL
    #     → Generates exactly 1 policy covering ALL family members
    #
    #   "Group":
    #     → Buyer   = a company purchasing for all its employees
    #     → Covers  = all employees (and their chosen dependents)
    #     → Count   = 50 contracts (one per employer in dim_employers)
    #     → family_id   = NULL
    #     → employer_id = EMP00001 ... EMP00050 (links to dim_employers)
    #     → Generates MANY policies — one per employee within the contract
    #       each employee independently chooses Individual or Family level
    #
    # HOW dim_contracts CONNECTS TO THE REST OF THE DATABASE:
    #
    #   dim_families  ──→  dim_contracts  ←──  dim_employers
    #                            │
    #                      dim_policies.contract_id
    #                            │
    #                      policy_members
    #                            │
    #                       dim_members
    # ═══════════════════════════════════════════════════════
    print(f"  Building dim_contracts...")
    rows = []
    contract_counter = 1

    def new_contract(ctype, family_id=None, employer_id=None):
        # ── INNER HELPER FUNCTION ──────────────────────────────────────
        # Creates ONE contract dictionary.
        # Called three times in a loop:
        #   once for each Individual contract
        #   once for each Family contract
        #   once for each Group contract
        #
        # PARAMETERS:
        #   ctype       : str  → "Individual", "Family", or "Group"
        #   family_id   : str  → FK to dim_families  (Family contracts only)
        #   employer_id : str  → FK to dim_employers (Group contracts only)
        #
        # nonlocal contract_counter:
        #   contract_counter lives in the OUTER function build_contracts().
        #   "nonlocal" tells Python to use that outer variable here —
        #   not create a new local one.
        #   This ensures contract IDs increment continuously across
        #   all three contract types:
        #   Individual contracts → CON0000001 ... CON0002666
        #   Family contracts     → CON0002667 ... CON0004666
        #   Group contracts      → CON0004667 ... CON0004716
        # ──────────────────────────────────────────────────────────────
        nonlocal contract_counter
        c = {

            "contract_id": f"CON{contract_counter:07d}",
            # Unique identifier for this contract.
            # Format: CON0000001, CON0000002, ... CON0004716
            #   "CON"    → prefix identifying this as a contract record
            #   {contract_counter} → increments across ALL contract types
            #   :07d     → always 7 digits, zero-padded
            #
            # This becomes the foreign key in dim_policies:
            #   dim_policies.contract_id → dim_contracts.contract_id
            # PRIMARY KEY of this table.

            "contract_type": ctype,
            # The TYPE of buying agreement.
            # One of: "Individual", "Family", "Group"
            # This is the most important field in dim_contracts —
            # it determines:
            #   - How many policies will be generated under this contract
            #   - How many members will be covered
            #   - Whether family_id or employer_id is populated
            #   - What roles members get in policy_members

            "family_id": family_id,
            # Foreign key → dim_families.family_id
            # ONLY populated for Family contracts.
            # NULL for Individual and Group contracts.
            #
            # Links this contract to the household it covers:
            #   Family contract CON0002667 → FAM000001
            #   means: "This contract covers household FAM000001"
            #
            # Used to find all members of a family:
            #   dim_contracts → dim_policies → policy_members → dim_members

            "employer_id": employer_id,
            # Foreign key → dim_employers.employer_id
            # ONLY populated for Group contracts.
            # NULL for Individual and Family contracts.
            #
            # Links this contract to the company that purchased it:
            #   Group contract CON0004667 → EMP00001
            #   means: "EMP00001 purchased this group contract
            #           to cover all its employees"

            "start_date": rand_date(date(2018,1,1), date(2023,1,1)),
            # The date this contract became active.
            # Random date between January 2018 and January 2023.
            #
            # WHY END AT 2023 (not 2024 like policies)?
            #   Contracts are longer-term agreements than policies.
            #   A contract may span multiple policy renewal years.
            #   Ending at 2023 ensures contracts were active before
            #   the latest policies were written under them.
            #
            # In real insurance:
            #   Individual/Family contracts → typically annual, auto-renewing
            #   Group contracts → typically 1–3 year agreements
            #                     with annual re-pricing/renegotiation

            "status": random.choices(
                          ["Active","Expired","Cancelled"],
                          weights=[70,20,10])[0],
            # Current state of this contract.
            # Assigned with weighted probability:
            #   Active    70% → contract is currently in force
            #   Expired   20% → contract ran its course and was not renewed
            #   Cancelled 10% → contract was terminated early
            #
            # WHAT EACH STATUS MEANS:
            #   Active    → members under this contract are currently covered
            #               new claims can be submitted
            #
            #   Expired   → coverage period ended naturally
            #               no new claims can be submitted
            #               historical claims still exist in fact_claims
            #               member may have renewed under a new contract
            #
            #   Cancelled → coverage was terminated before the end date
            #               reasons: non-payment of premium, fraud discovered,
            #               company went out of business (Group),
            #               member voluntarily cancelled (Individual)
            #
            # NOTE: contract status and policy status are SEPARATE:
            #   A contract can be Active while a specific policy under it
            #   is Expired (member changed plans at renewal).
            #   dim_policies has its own status field for this reason.

        }
        contract_counter += 1
        # Increment the counter AFTER creating the contract
        # so the next call to new_contract() gets the next number.
        return c

    # ── STEP 1: CREATE INDIVIDUAL CONTRACTS ───────────────────────────────
    # One Individual contract per individual member slot.
    # n_individual = N_MEMBERS // 3 = ~2,666
    # These contracts have no family_id and no employer_id.
    for _ in range(n_individual):
        # _ means we do not need the loop counter — just repeat n times
        rows.append(new_contract("Individual"))
        # family_id and employer_id default to None (NULL in DuckDB)

    # ── STEP 2: CREATE FAMILY CONTRACTS ───────────────────────────────────
    # One Family contract per unique family in dim_families.
    # Total: 2,000 Family contracts — one per household.
    # Each gets the family_id of the household it covers.
    for fam_id in families["family_id"]:
        rows.append(new_contract("Family", family_id=fam_id))
        # employer_id defaults to None

    # ── STEP 3: CREATE GROUP CONTRACTS ────────────────────────────────────
    # One Group contract per employer in dim_employers.
    # Total: 50 Group contracts — one per company.
    # Each gets the employer_id of the company that purchased it.
    for emp_id in employers["employer_id"]:
        rows.append(new_contract("Group", employer_id=emp_id))
        # family_id defaults to None

    df = pd.DataFrame(rows)
    print(f"    → {len(df):,} contracts total  "
          f"({n_individual:,} Individual / "
          f"{len(families):,} Family / "
          f"{len(employers):,} Group)")
    return df
    # ── RESULT:
    # A DataFrame with 4,716 rows and 5 columns:
    #   contract_id, contract_type, family_id, employer_id,
    #   start_date, status
    #
    # Row distribution:
    #   ~2,666 rows where contract_type = "Individual"
    #          family_id = NULL, employer_id = NULL
    #    2,000 rows where contract_type = "Family"
    #          family_id = FAM000001...FAM002000, employer_id = NULL
    #       50 rows where contract_type = "Group"
    #          family_id = NULL, employer_id = EMP00001...EMP00050




    
# ─────────────────────────────────────────────
# 6. DIM_POLICIES + POLICY_MEMBERS
#    dim_policies  = the SPECIFIC PLAN within a contract
#                   answers: WHAT plan? (HMO/PPO, deductible, etc.)
#
#    policy_members = bridge table linking policies to members
#                    answers: WHO exactly is covered by this policy?
#
#    COVERAGE RULES ENFORCED HERE:
#
#    Individual contract:
#        → 1 policy → 1 member (role = "Policyholder")
#
#    Family contract:
#        → 1 policy → ALL members of that family
#        → first member = "Policyholder"
#        → rest = "Dependent"
#        → ALL family members share EXACTLY THE SAME policy
#          (no exceptions — this is the rule set in v3)
#
#    Group contract:
#        → each employee independently chooses:
#            "Individual" coverage → 1 policy → just the employee
#            "Family" coverage     → 1 policy → employee + their family members
#        → all members of an employee's family share the SAME policy
#          (same rule as above — no mixing within a family)
# ─────────────────────────────────────────────
def build_policies_and_bridge(
        members: pd.DataFrame,
        contracts: pd.DataFrame,
        families: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds TWO tables simultaneously:
    #     1. dim_policies   → the specific plan details for each coverage unit
    #     2. policy_members → the bridge table linking policies to members
    #
    #   These two tables are built together because they are
    #   deeply interdependent — every policy created immediately
    #   needs members assigned to it via the bridge table.
    #   Building them separately would require complex cross-referencing.
    #
    # PARAMETERS:
    #   members   : pd.DataFrame → dim_members (all 8,000 people)
    #   contracts : pd.DataFrame → dim_contracts (all 4,716 contracts)
    #   families  : pd.DataFrame → dim_families (all 2,000 households)
    #                              needed to look up family_size
    #
    # RETURNS:
    #   A TUPLE of two DataFrames:
    #     [0] → dim_policies   (loaded into DuckDB as dim_policies)
    #     [1] → policy_members (loaded into DuckDB as policy_members)
    #
    #   tuple[pd.DataFrame, pd.DataFrame] means:
    #   the function returns exactly two DataFrames packaged together.
    #   The caller unpacks them like this:
    #     policies, bridge = build_policies_and_bridge(members, contracts, families)
    #
    # THE MEMBER POOL CONCEPT — CRITICAL TO UNDERSTAND:
    #   all_member_ids = [MEM0000001, MEM0000002, ... MEM0008000]
    #   member_idx     = a pointer that starts at 0 and moves forward
    #
    #   Think of the 8,000 members as a LINE of people waiting to be assigned:
    #     Position 0  → MEM0000001 (first in line)
    #     Position 1  → MEM0000002
    #     ...
    #     Position 7999 → MEM0008000 (last in line)
    #
    #   As each contract is processed, members are taken from the front
    #   of the line and assigned to that contract's policy.
    #   member_idx tracks how far along the line we are.
    #   This ensures NO member is assigned twice as a primary member
    #   and NO member is left without coverage.
    #
    # THE THREE COVERAGE RULES ENFORCED HERE:
    #
    #   RULE 1 — Individual contracts:
    #     One policy → covers exactly 1 member
    #     That member gets role = "Policyholder"
    #
    #   RULE 2 — Family contracts:
    #     One policy → covers ALL members of that family
    #     First member  = "Policyholder"
    #     Rest          = "Dependent"
    #     ALL members share EXACTLY THE SAME policy_id (v3 rule)
    #     No family member can have a different policy
    #
    #   RULE 3 — Group contracts:
    #     Each EMPLOYEE independently chooses their coverage level:
    #       "Individual" (40%) → policy covers just the employee
    #       "Family"     (60%) → policy covers employee + dependents
    #     Employee gets role = "Employee"
    #     Dependents get role = "Dependent"
    #     All dependents of one employee share that employee's policy
    #     (same v3 family rule applied at the employee level)
    #
    # HOW dim_policies CONNECTS TO THE REST OF THE DATABASE:
    #
    #   dim_contracts
    #       └── dim_policies.contract_id  (what contract is this plan under?)
    #               └── policy_members.policy_id  (who is covered?)
    #                       └── dim_members.member_id  (their details)
    #                       └── fact_claims.policy_id  (claims under this plan)
    #
    # HOW policy_members CONNECTS TO THE REST OF THE DATABASE:
    #
    #   dim_policies  ──→  policy_members  ←──  dim_members
    #                            │
    #                       fact_claims
    #                    (claim filed under a policy that covers this member)
    # ═══════════════════════════════════════════════════════

    policy_rows = []   # will become dim_policies
    bridge_rows = []   # will become policy_members
    policy_counter = 1
    bridge_counter = 1

    def new_policy(contract_id: str, coverage_level: str) -> dict:
        # ── INNER HELPER FUNCTION ──────────────────────────────────────
        # Creates ONE policy dictionary.
        # Called once per Individual member,
        #        once per Family household,
        #        once per Employee within a Group contract.
        #
        # PARAMETERS:
        #   contract_id    : str → FK linking this policy to dim_contracts
        #   coverage_level : str → "Individual" or "Family"
        #                          Individual = covers one person only
        #                          Family     = covers a household unit
        # ──────────────────────────────────────────────────────────────
        nonlocal policy_counter
        start = rand_date(date(2018,1,1), date(2024,1,1))
        p = {

            "policy_id": f"POL{policy_counter:07d}",
            # Unique identifier for this specific plan.
            # Format: POL0000001, POL0000002, ...
            # Increments continuously across ALL policy types.
            # PRIMARY KEY of dim_policies.
            # Referenced by:
            #   policy_members.policy_id → here
            #   fact_claims.policy_id    → here

            "contract_id": contract_id,
            # Foreign key → dim_contracts.contract_id
            # Links this policy to the buying agreement it belongs to.
            #
            # For Individual contract CON0000001:
            #   → generates 1 policy with contract_id = CON0000001
            #
            # For Family contract CON0002667:
            #   → generates 1 policy with contract_id = CON0002667
            #     covering the whole family
            #
            # For Group contract CON0004667:
            #   → generates MANY policies all with contract_id = CON0004667
            #     one policy per employee at that company

            "coverage_level": coverage_level,
            # What LEVEL of coverage this policy provides.
            # Either "Individual" or "Family".
            #
            # "Individual":
            #   → policy covers exactly ONE person
            #   → lower premium (only one person's risk)
            #   → policy_members will have exactly 1 row for this policy
            #
            # "Family":
            #   → policy covers a household unit (2+ people)
            #   → higher premium (covers multiple people's risk)
            #   → policy_members will have 2+ rows for this policy
            #   → shared deductible: the family has ONE combined deductible
            #     that all members contribute to together
            #
            # IMPORTANT: For Group contracts, this is the employee's CHOICE:
            #   Employee A (single)   → chooses "Individual" → lower paycheck deduction
            #   Employee B (has kids) → chooses "Family"     → higher paycheck deduction
            #   Both are under the SAME group contract_id
            #   but have DIFFERENT coverage_level policies

            "plan_type": random.choice(PLAN_TYPES),
            # The type of insurance plan.
            # Randomly chosen from: ["HMO","PPO","EPO","HDHP","POS"]
            # Each plan type has different rules about:
            #   - Which providers the member can see
            #   - Whether referrals are needed for specialists
            #   - How much the member pays out of pocket
            # See PLAN_TYPES comments for full details.

            "start_date": start,
            # The date this specific policy became active.
            # Random date between January 2018 and January 2024.
            # Generated once and stored in the 'start' variable
            # so end_date can be calculated from it below.

            "end_date": start + timedelta(days=365),
            # The date this policy expires.
            # Always exactly 365 days after start_date.
            # This models the standard annual policy renewal cycle:
            #   Policy starts January 1, 2022
            #   Policy ends   December 31, 2022
            #   Member must renew or choose a new plan for next year.
            #
            # NOTE: contract start_date and policy start_date may differ:
            #   A Group contract signed in 2019 may have an employee
            #   who joined in 2022 → their policy starts in 2022
            #   but the contract has been active since 2019.

            "deductible": random.choice([500,1000,2000,3000,5000]),
            # The amount the member must pay OUT OF POCKET
            # before the insurance company starts paying its share.
            # Chosen randomly from 5 common deductible amounts.
            #
            # HOW DEDUCTIBLE WORKS:
            #   deductible = $2,000
            #
            #   Claim 1: $800  → member pays $800   (total paid: $800)
            #   Claim 2: $900  → member pays $900   (total paid: $1,700)
            #   Claim 3: $500  → member pays $300   (deductible met at $2,000!)
            #                 → insurer pays $200
            #   All subsequent claims → insurer pays their full share
            #                           until out_of_pocket_max is hit
            #
            # FAMILY DEDUCTIBLE RULE:
            #   For Family coverage_level policies:
            #   ALL family members contribute to ONE shared deductible.
            #   Example: deductible = $3,000
            #     Child visits doctor: pays $400  (family total: $400)
            #     Parent visits doctor: pays $800 (family total: $1,200)
            #     Another parent visit: pays $800 (family total: $2,000)
            #     etc. until the $3,000 family deductible is met.
            #
            # PLAN TYPE INTERACTION:
            #   HDHP → always has HIGH deductible ($1,400+ for Individual)
            #   HMO  → typically lower deductible
            #   PPO  → moderate deductible

            "co_payment_rate": round(random.uniform(0.05, 0.30), 2),
            # The percentage of each bill the member pays AFTER
            # the deductible has been met. Also called "coinsurance".
            # Random value between 5% and 30%.
            # Examples:
            #   0.20 → member pays 20% of every bill, insurer pays 80%
            #   0.10 → member pays 10% of every bill, insurer pays 90%
            #
            # DIFFERENCE BETWEEN COPAY AND COINSURANCE:
            #   Copay       → FIXED dollar amount per visit ($20 per visit)
            #   Coinsurance → PERCENTAGE of the bill (20% of whatever it costs)
            #   Our database models coinsurance (percentage-based).
            #
            # Used in fact_claims:
            #   co_payment_amount = claim_amount × co_payment_rate

            "out_of_pocket_max": random.choice([3000,5000,7000,10000]),
            # The MAXIMUM total amount the member will EVER pay in one year.
            # Once this limit is reached, the insurer pays 100% of everything.
            # Chosen randomly from 4 common amounts.
            #
            # This is the member's FINANCIAL SAFETY NET:
            #   Without out_of_pocket_max → a catastrophic illness could
            #   cost the member unlimited money even with insurance.
            #   With out_of_pocket_max    → member's exposure is capped.
            #
            # Under the ACA, there is a LEGAL MAXIMUM out-of-pocket limit:
            #   2024: $9,450 for Individual / $18,900 for Family
            #   Our values (3,000–10,000) are within realistic range.
            #
            # INTERACTION WITH DEDUCTIBLE:
            #   deductible is always LESS THAN out_of_pocket_max.
            #   The deductible counts TOWARD the out_of_pocket_max.
            #   Example:
            #     deductible = $2,000, out_of_pocket_max = $5,000
            #     After paying $2,000 deductible → insurer starts sharing costs
            #     Member continues paying coinsurance until total reaches $5,000
            #     After $5,000 total paid → insurer pays 100% for rest of year

            "premium_amount": round(random.uniform(150, 900), 2),
            # The monthly amount the member (or employer) pays to keep
            # this policy active — regardless of whether any claims are filed.
            # Random value between $150 and $900 per month.
            #
            # Real world premium ranges (2024):
            #   Individual plan: $300–$600/month average
            #   Family plan:     $800–$1,800/month average
            #   Our range ($150–$900) is simplified for simulation purposes.
            #
            # PREMIUM vs DEDUCTIBLE TRADEOFF:
            #   High premium + Low deductible  → better for frequent care users
            #   Low premium  + High deductible → better for healthy people (HDHP)
            #
            # FOR GROUP CONTRACTS:
            #   The employer pays ~70-80% of this premium amount.
            #   The employee pays remaining 20-30% via paycheck deduction.
            #   This split is not explicitly modeled in our database
            #   but is implied by the employment context.

            "status": random.choices(
                          ["Active","Expired","Cancelled"],
                          weights=[70,20,10])[0],
            # Current state of this specific policy.
            # Assigned with weighted probability:
            #   Active    70% → policy is currently in force
            #   Expired   20% → policy period ended, not renewed
            #   Cancelled 10% → policy terminated before end date
            #
            # NOTE: policy status is INDEPENDENT of contract status.
            #   A Group contract can be Active (company still has coverage)
            #   while one employee's specific policy is Expired
            #   (that employee left the company or changed plans).
            #   This is why both dim_contracts and dim_policies have
            #   their own status fields.

        }
        policy_counter += 1
        return p

    def add_bridge(policy_id: str, member_id: str, role: str):
        # ── INNER HELPER FUNCTION ──────────────────────────────────────
        # Adds ONE row to the policy_members bridge table.
        # Called every time a member needs to be linked to a policy.
        #
        # PARAMETERS:
        #   policy_id : str → FK to dim_policies
        #   member_id : str → FK to dim_members
        #   role      : str → the role this person plays under this policy
        #
        # ROLES IN THIS DATABASE:
        #   "Policyholder" → the person who bought an Individual or Family policy
        #                    they are legally responsible for the contract
        #                    one per Individual policy
        #                    one per Family policy (the head of household)
        #
        #   "Dependent"    → a family member covered under the Policyholder's plan
        #                    spouse, children, sometimes parents
        #                    has no financial responsibility for the contract
        #                    multiple per Family policy
        #
        #   "Employee"     → a worker covered under their employer's Group contract
        #                    similar to Policyholder but in an employment context
        #                    one per policy within a Group contract
        #
        # THE BRIDGE TABLE CONCEPT:
        #   policy_members solves a many-to-many relationship:
        #     One policy  → can cover MANY members (family plan)
        #     One member  → can appear in MANY policies (dual coverage)
        #   Without this bridge table, we would have to either:
        #     (a) put multiple member_ids on one policy row (violates 1NF)
        #     (b) duplicate the policy row for each member (violates 3NF)
        #   The bridge table is the correct normalized solution.
        # ──────────────────────────────────────────────────────────────
        nonlocal bridge_counter
        bridge_rows.append({

            "bridge_id": f"BRG{bridge_counter:09d}",
            # Unique identifier for this specific policy-member link.
            # Format: BRG000000001, BRG000000002, ...
            # PRIMARY KEY of policy_members.
            # 9 digits because there will be many more bridge rows
            # than policies or members — each family policy generates
            # multiple bridge rows (one per family member).

            "policy_id": policy_id,
            # Foreign key → dim_policies.policy_id
            # Which policy covers this member?
            # For a family of 4 sharing one policy:
            #   All 4 bridge rows have the SAME policy_id
            #   This is how we know they share the same plan.

            "member_id": member_id,
            # Foreign key → dim_members.member_id
            # Which person is covered by this policy?
            # For a family of 4 sharing one policy:
            #   Each bridge row has a DIFFERENT member_id
            #   This is how we know which specific people are covered.

            "role": role,
            # The role this person plays under this policy.
            # "Policyholder", "Dependent", or "Employee"
            # Determines financial and legal responsibility.

        })
        bridge_counter += 1

    # Prepare member pool — the ordered line of 8,000 members
    all_member_ids = members["member_id"].tolist()
    member_idx     = 0
    # member_idx is the POINTER into the member pool.
    # Starts at 0 (pointing at MEM0000001).
    # Increments every time a member is assigned to a policy.
    # When member_idx reaches 8,000 → all members have been assigned.

    # Build family lookup: family_id → list of member_ids
    # Used to track which members belong to which family
    family_ids     = families["family_id"].tolist()
    family_members = {fid: [] for fid in family_ids}
    # Creates a dictionary like:
    # { "FAM000001": [], "FAM000002": [], ... "FAM002000": [] }
    # Empty lists that get filled as members are assigned to families.

    # ── STEP 1: PROCESS INDIVIDUAL CONTRACTS ──────────────────────────────
    # For each Individual contract → create 1 policy → assign 1 member
    # Result: 1 policy row + 1 bridge row per Individual contract
    ind_contracts = contracts[contracts["contract_type"] == "Individual"]
    print(f"    → Processing {len(ind_contracts):,} Individual contracts...")

    for _, contract in ind_contracts.iterrows():
        # iterrows() loops through the DataFrame one row at a time
        # _ = the row index (we don't need it)
        # contract = the current row as a pandas Series

        if member_idx >= len(all_member_ids):
            break
        # Safety check: stop if we have run out of members.
        # This prevents an IndexError if there are more contracts
        # than members available in the pool.

        member_id = all_member_ids[member_idx]
        member_idx += 1
        # Take the NEXT available member from the pool.
        # Advance the pointer by 1 so the next contract
        # gets a different member.

        pol = new_policy(contract["contract_id"], "Individual")
        policy_rows.append(pol)
        # Create ONE policy under this Individual contract.
        # coverage_level = "Individual" (covers just this one person).
        # Add it to the policy_rows list.

        add_bridge(pol["policy_id"], member_id, "Policyholder")
        # Link this member to their policy in the bridge table.
        # Role = "Policyholder" because they bought their own Individual plan.

    # ── STEP 2: PROCESS FAMILY CONTRACTS ──────────────────────────────────
    # For each Family contract → create 1 policy → assign ALL family members
    # Result: 1 policy row + family_size bridge rows per Family contract
    fam_contracts = contracts[contracts["contract_type"] == "Family"]
    print(f"    → Processing {len(fam_contracts):,} Family contracts...")

    for _, contract in fam_contracts.iterrows():
        fam_id      = contract["family_id"]
        # Get the family_id for this contract.
        # This links back to dim_families to find the family_size.

        family_row  = families[families["family_id"] == fam_id].iloc[0]
        family_size = family_row["family_size"]
        # Look up this family's size from dim_families.
        # .iloc[0] gets the first (and only) matching row as a Series.
        # family_size tells us how many members to assign to this family.

        fam_member_ids = []
        for _ in range(family_size):
            if member_idx >= len(all_member_ids):
                break
            fam_member_ids.append(all_member_ids[member_idx])
            member_idx += 1
        # Take the next 'family_size' members from the pool.
        # Example: family_size = 4 → take MEM0002667, MEM0002668,
        #          MEM0002669, MEM0002670 from the pool.
        # These 4 people will ALL share the same single policy.

        if not fam_member_ids:
            continue
        # Skip this family if no members were available.
        # This can happen if member pool is exhausted.

        pol = new_policy(contract["contract_id"], "Family")
        policy_rows.append(pol)
        # Create ONE policy for the ENTIRE family.
        # coverage_level = "Family" (covers all members together).
        # This enforces the v3 rule:
        # ALL family members share EXACTLY ONE policy — no exceptions.

        for i, mid in enumerate(fam_member_ids):
            role = "Policyholder" if i == 0 else "Dependent"
            add_bridge(pol["policy_id"], mid, role)
            family_members[fam_id].append(mid)
        # Link ALL family members to the SAME policy_id.
        # enumerate() gives us both the index (i) and value (mid):
        #   i=0 → first member  → role = "Policyholder" (head of household)
        #   i=1 → second member → role = "Dependent"    (spouse)
        #   i=2 → third member  → role = "Dependent"    (child)
        #   i=3 → fourth member → role = "Dependent"    (child)
        # Also track which members belong to this family
        # in the family_members dictionary.

    # ── STEP 3: PROCESS GROUP CONTRACTS ───────────────────────────────────
    # For each Group contract → assign 10–80 employees
    # Each employee INDEPENDENTLY chooses Individual or Family coverage
    # Result: many policies per Group contract (one per employee)
    grp_contracts = contracts[contracts["contract_type"] == "Group"]
    print(f"    → Processing {len(grp_contracts):,} Group contracts...")

    for _, contract in grp_contracts.iterrows():

        n_employees = random.randint(10, 80)
        emp_member_ids = []
        for _ in range(n_employees):
            if member_idx >= len(all_member_ids):
                break
            emp_member_ids.append(all_member_ids[member_idx])
            member_idx += 1
        # Assign 10–80 employees to this employer's group contract.
        # Each employee is a member taken from the pool.
        # The random range reflects different company sizes:
        #   Small company  → 10 employees
        #   Large company  → 80 employees

        if not emp_member_ids:
            continue

        for employee_id in emp_member_ids:

            chosen_level = random.choices(
                ["Individual", "Family"],
                weights=[40, 60]
            )[0]
            # Each employee independently chooses their coverage level:
            #   40% choose "Individual" → covers just themselves
            #   60% choose "Family"     → covers themselves + dependents
            #
            # This models real workplace enrollment:
            # During open enrollment, each employee fills out a form
            # choosing their coverage tier for the coming year.
            # Single employees often choose Individual (lower premium deduction).
            # Employees with families choose Family (covers their household).

            pol = new_policy(contract["contract_id"], chosen_level)
            policy_rows.append(pol)
            # Create ONE policy per employee.
            # All employees at the same company share contract_id
            # but each has their own unique policy_id with their chosen plan.

            if chosen_level == "Individual":
                add_bridge(pol["policy_id"], employee_id, "Employee")
                # Individual coverage → just the employee on this policy.
                # One bridge row only.
                # Role = "Employee" (not "Policyholder" — they are in a
                # Group context, not a personal Individual contract).

            else:
                add_bridge(pol["policy_id"], employee_id, "Employee")
                n_dependents = random.randint(1, 4)
                for _ in range(n_dependents):
                    if member_idx >= len(all_member_ids):
                        break
                    dep_id = all_member_ids[member_idx]
                    member_idx += 1
                    add_bridge(pol["policy_id"], dep_id, "Dependent")
                # Family coverage → employee + 1 to 4 dependents.
                # First add the employee (role = "Employee").
                # Then take 1–4 more members from the pool as dependents.
                # ALL dependents share the SAME policy as the employee —
                # enforcing the v3 rule:
                # no family member can be on a different policy.
                # Role = "Dependent" for spouse and children.

    policies_df = pd.DataFrame(policy_rows)
    bridge_df   = pd.DataFrame(bridge_rows)
    # Convert the accumulated lists of dictionaries into DataFrames.
    # policy_rows → dim_policies
    # bridge_rows → policy_members

    print(f"  Building dim_policies        ({len(policies_df):>6,} rows)")
    print(f"  Building policy_members      ({len(bridge_df):>6,} rows)  [Bridge table]")

    return policies_df, bridge_df
    # Return BOTH DataFrames as a tuple.
    # The caller unpacks them:
    #   policies, bridge = build_policies_and_bridge(members, contracts, families)
    #
    # SUMMARY OF WHAT WAS BUILT:
    #
    # dim_policies rows (approximately):
    #   ~2,666 Individual policies  (one per Individual contract member)
    #   ~2,000 Family policies      (one per household)
    #   ~1,450 Group policies       (10–80 per employer × 50 employers)
    #   Total: ~6,116 policies
    #
    # policy_members rows (approximately):
    #   ~2,666 rows for Individual policies    (1 member each)
    #   ~6,000 rows for Family policies        (avg 3 members each)
    #   ~5,000 rows for Group policies         (employees + dependents)
    #   Total: ~13,666 bridge rows
    #   → every row guarantees one member is covered by one policy


    
# ─────────────────────────────────────────────
# 7. FACT_CLAIMS
#    Every claim is filed by a member under a policy
#    that actually covers them — enforced via bridge lookup.
# ─────────────────────────────────────────────

def build_claims(n: int, bridge: pd.DataFrame,
                 providers: pd.DataFrame) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the fact_claims table.
    #   This is the LARGEST and most CENTRAL table in the
    #   entire database — almost all analytics start here.
    #
    # PARAMETERS:
    #   n         : int          → number of claims to generate
    #                              in our case N_CLAIMS = 50,000
    #   bridge    : pd.DataFrame → the policy_members table
    #                              used to ensure every claim is filed
    #                              under a policy that ACTUALLY covers
    #                              the member making the claim
    #   providers : pd.DataFrame → the dim_providers table
    #                              used to assign a provider to each claim
    #
    # RETURNS:
    #   A pandas DataFrame with 50,000 rows.
    #   Loaded into DuckDB as fact_claims.
    #
    # WHY IS THIS CALLED "fact_claims" AND NOT "dim_claims"?
    #   Tables in a database are classified as either:
    #
    #   DIMENSION tables (dim_):
    #     → Descriptive reference data that changes slowly
    #     → Examples: dim_members, dim_providers, dim_policies
    #     → Answer WHO, WHAT, WHERE questions
    #     → Relatively small number of rows
    #     → A member record rarely changes once created
    #
    #   FACT tables (fact_):
    #     → Transactional events that happened at a point in time
    #     → Examples: fact_claims, fact_payments
    #     → Answer WHEN, HOW MUCH, HOW MANY questions
    #     → Large number of rows — one row per event
    #     → A claim is a snapshot of one medical event in time
    #     → Facts never change once recorded (immutable history)
    #
    #   fact_claims is a FACT table because:
    #     → Each row = one specific medical event on a specific date
    #     → Once a claim is filed it is a permanent historical record
    #     → It contains measurable amounts (claim_amount, paid_amount)
    #     → It connects to MULTIPLE dimension tables via foreign keys
    #
    # THE MOST IMPORTANT INTEGRITY RULE IN v3:
    #   Every claim must be filed under a policy that ACTUALLY covers
    #   the member who received the care.
    #
    #   In v1 and v2 this was violated:
    #     member_id and policy_id were assigned INDEPENDENTLY at random
    #     → a claim could have member_id = MEM001 and policy_id = POL999
    #       even if POL999 does not cover MEM001 at all
    #     → completely unrealistic and analytically meaningless
    #
    #   In v3 this is fixed via the bridge table lookup:
    #     member_policies = member_id → [list of policy_ids that cover them]
    #     When building a claim:
    #       1. Pick a random member
    #       2. Look up which policies actually cover that member
    #       3. Pick one of THOSE policies for the claim
    #     → guaranteed referential integrity between member and policy
    #
    # HOW fact_claims CONNECTS TO THE REST OF THE DATABASE:
    #
    #   fact_claims
    #       → dim_members.member_id      (who received care)
    #       → dim_policies.policy_id     (which plan covers it)
    #       → dim_providers.provider_id  (who delivered care)
    #       ← fact_claim_diagnoses       (what diagnosis/procedure)
    #       ← fact_payments              (what payment was made)
    #
    #   fact_claims sits at the CENTER of the star schema:
    #   all dimension tables point TO it,
    #   all other fact tables point FROM it.
    # ═══════════════════════════════════════════════════════
    print(f"  Building fact_claims         ({n:>6,} rows)...")

    # ── BUILD THE MEMBER → POLICY LOOKUP ──────────────────────────────────
    member_policies = (
        bridge.groupby("member_id")["policy_id"]
              .apply(list)
              .to_dict()
    )
    # This is the KEY data structure that enforces referential integrity.
    # It transforms the bridge table into a fast lookup dictionary:
    #
    # bridge table (raw):
    #   policy_id    member_id    role
    #   POL0000001   MEM0000001   Policyholder
    #   POL0000002   MEM0000002   Policyholder
    #   POL0000003   MEM0000003   Policyholder
    #   POL0000003   MEM0000004   Dependent      ← MEM0000004 is on same policy
    #   POL0000004   MEM0000004   Employee       ← MEM0000004 also has a 2nd policy
    #
    # After groupby + apply(list) + to_dict():
    # member_policies = {
    #   "MEM0000001": ["POL0000001"],
    #   "MEM0000002": ["POL0000002"],
    #   "MEM0000003": ["POL0000003"],
    #   "MEM0000004": ["POL0000003", "POL0000004"],  ← dual coverage!
    #   ...
    # }
    #
    # Now when building a claim for MEM0000004:
    #   random.choice(member_policies["MEM0000004"])
    #   → picks either POL0000003 or POL0000004
    #   → both are valid because both cover MEM0000004
    #   → never picks a policy that does NOT cover this member

    # Only use members that actually have a policy assigned
    covered_members = list(member_policies.keys())
    # In v3, every member should have at least one policy.
    # This line extracts just the member_ids that appear in
    # the bridge table — i.e. members who are actually covered.
    # In a perfectly built database this = all 8,000 members.
    # The safety check handles edge cases where member pool
    # ran out during policy assignment.

    provider_ids = providers["provider_id"].tolist()
    # Simple list of all 300 provider IDs.
    # Used to randomly assign a provider to each claim.

    rows = []

    for i in range(n):
        # Loop 50,000 times — one iteration = one claim

        member_id = random.choice(covered_members)
        # Pick a random member from the 8,000 covered members.
        # Every member has an equal chance of being picked.
        # Over 50,000 iterations → each member gets ~6.25 claims on average.
        # But because it is random, some members will have more,
        # some will have fewer — realistic variation.

        policy_id = random.choice(member_policies[member_id])
        # Look up this member's policies in our dictionary.
        # Pick one at random from the list of policies that cover them.
        # For most members: only 1 policy → always picks that one.
        # For dual-coverage members: 2 policies → randomly picks one.
        # CRITICAL: this can NEVER pick a policy that doesn't cover the member.

        claim_date = rand_date(date(2019,1,1), date(2024,12,31))
        # Random claim date spanning 6 years of history.
        # 2019 → earliest (gives historical depth)
        # 2024 → latest   (most recent year)
        # Spread across 6 years → realistic multi-year claims history.
        # More years = better for trend analysis:
        #   "Are claim amounts increasing year over year?"
        #   "Which months have the highest claim volume?"

        is_inpatient = random.random() < 0.20
        # 20% chance this claim is an inpatient (hospital admission) claim.
        # random.random() generates a float between 0.0 and 1.0.
        # If the value is less than 0.20 → is_inpatient = True
        # If the value is 0.20 or more  → is_inpatient = False
        #
        # WHY 20%?
        # In real US insurance data:
        #   Inpatient claims  → ~5–15% of total claims by volume
        #                       but ~40–60% of total costs
        #   20% is slightly high but produces enough inpatient records
        #   for meaningful analytics without overwhelming the dataset.
        #
        # is_inpatient drives THREE other fields below:
        #   length_of_stay, admission_date, discharge_date

        los = random.randint(1, 14) if is_inpatient else 0
        # Length of Stay in days.
        # Only relevant for inpatient claims:
        #   is_inpatient = True  → random 1 to 14 days in hospital
        #   is_inpatient = False → 0 (patient left same day)
        #
        # Real world average length of stay in US hospitals: ~4–5 days.
        # Our range (1–14) covers:
        #   1 day    → minor surgery, observation stay
        #   3–5 days → typical medical admission
        #   7–14 days → serious illness, ICU stay, major surgery recovery

        discharge = claim_date + timedelta(days=los) if is_inpatient else None
        # The date the patient was discharged from the hospital.
        # Only set for inpatient claims:
        #   is_inpatient = True  → discharge = claim_date + length_of_stay
        #   is_inpatient = False → None (NULL in DuckDB — no discharge needed)
        #
        # Example:
        #   claim_date = 2022-03-10  (admission date)
        #   los        = 5 days
        #   discharge  = 2022-03-15

        claim_amt = round(np.random.lognormal(7, 1.2), 2)
        # The total dollar amount billed by the provider for this claim.
        # Generated using a LOG-NORMAL distribution:
        #   np.random.lognormal(mean=7, sigma=1.2)
        #
        # WHY LOG-NORMAL AND NOT GAUSSIAN?
        #   Medical claim amounts are NOT normally distributed.
        #   They are HEAVILY RIGHT-SKEWED:
        #     → Most claims are small ($100–$2,000)
        #     → Some claims are medium ($2,000–$20,000)
        #     → A few claims are extremely large ($50,000–$500,000+)
        #   A log-normal distribution perfectly models this shape.
        #
        # What lognormal(7, 1.2) produces:
        #   The underlying normal distribution has:
        #     mean  = 7    → e^7 ≈ $1,097 (median claim amount)
        #     sigma = 1.2  → wide spread covering small to very large claims
        #   Typical output range: $50 to $500,000+
        #   Most values: $200 to $5,000
        #   Rare large values: $50,000+ (catastrophic claims)
        #
        # Visual shape:
        #   Frequency
        #     │▓▓▓▓▓
        #     │▓▓▓▓▓▓▓▓
        #     │▓▓▓▓▓▓▓▓▓▓▓░░░░
        #     └──────────────────────→ Claim Amount ($)
        #     $0  $1k  $5k  $20k  $100k+

        rows.append({

            "claim_id": f"CLM{i+1:08d}",
            # Unique identifier for this claim transaction.
            # Format: CLM00000001, CLM00000002, ... CLM00050000
            #   "CLM"    → prefix identifying this as a claim record
            #   {i+1}    → counter starting at 1
            #   :08d     → always 8 digits (50,000 needs only 5 but
            #              8 digits future-proofs for larger datasets)
            # PRIMARY KEY of fact_claims.

            "policy_id": policy_id,
            # Foreign key → dim_policies.policy_id
            # Which insurance policy is this claim filed under?
            # GUARANTEED to be a policy that covers this member
            # (enforced by the member_policies lookup above).
            # Used to apply the correct deductible and co_payment_rate
            # when calculating how much the insurer owes.

            "member_id": member_id,
            # Foreign key → dim_members.member_id
            # Which person received the medical care?
            # This is the PATIENT — the human being who was treated.
            # Used to retrieve demographics, health profile, and
            # all historical claims for this person.

            "provider_id": random.choice(provider_ids),
            # Foreign key → dim_providers.provider_id
            # Which healthcare entity delivered the care and filed this claim?
            # Randomly chosen from all 300 providers.
            # Used to retrieve provider_type, network_flag, state etc.
            # via JOIN to dim_providers.

            "claim_date": claim_date,
            # The date the medical service was provided.
            # For inpatient claims: this is the ADMISSION date.
            # For outpatient claims: this is the date of the visit.
            # Random date between 2019 and 2024.
            # The single most important date field for time-series analysis.

            "claim_type": random.choice(CLAIM_TYPES),
            # The setting/context of this medical event.
            # Randomly chosen from:
            #   ["Inpatient","Outpatient","Pharmacy","Emergency","Preventive"]
            # Each type has different cost profiles and coverage rules.
            # See CLAIM_TYPES comments for full details.
            #
            # NOTE: claim_type is assigned independently of is_inpatient.
            # In a perfectly consistent database:
            #   is_inpatient = True  → claim_type should always = "Inpatient"
            #   is_inpatient = False → claim_type should be one of the others
            # This is a known simplification in our simulated data.

            "claim_status": random.choices(
                                CLAIM_STATUSES,
                                weights=[65,10,15,10])[0],
            # The current processing state of this claim.
            # Assigned with WEIGHTED probability (not equal chance):
            #   "Approved"     65% → ~32,500 claims → triggers fact_payments
            #   "Rejected"     10% → ~5,000  claims → no payment created
            #   "Pending"      15% → ~7,500  claims → awaiting processing
            #   "Under Review" 10% → ~5,000  claims → flagged for investigation
            #
            # random.choices() vs random.choice():
            #   random.choice()  → equal probability for all options
            #   random.choices() → weighted probability via weights parameter
            #   weights=[65,10,15,10] maps to CLAIM_STATUSES in order:
            #     Approved=65, Rejected=10, Pending=15, Under Review=10
            #   [0] takes the first (and only) result from random.choices()
            #   because random.choices() always returns a LIST.

            "claim_amount": claim_amt,
            # The TOTAL amount billed by the provider.
            # The "sticker price" before insurance applies.
            # Generated via lognormal distribution (explained above).
            # This is what the provider CHARGES — not what they get paid.
            # In real insurance, providers often bill much more than
            # they expect to receive (due to negotiated rate discounts).

            "paid_amount": round(claim_amt * random.uniform(0.5, 0.95), 2),
            # The amount the insurer ACTUALLY PAYS to the provider.
            # Calculated as a random 50%–95% of claim_amount.
            #
            # WHY LESS THAN claim_amount?
            #   Insurers negotiate DISCOUNTED RATES with in-network providers.
            #   Example:
            #     Provider bills:     $1,000 (claim_amount)
            #     Negotiated rate:    $700   (what insurer agrees to pay)
            #     Insurer pays:       $560   (80% of $700 after member's 20% coinsurance)
            #     Member pays copay:  $140   (co_payment_amount)
            #     Provider writes off: $300  (contractual adjustment — never collected)
            #
            # Our simplified model:
            #   paid_amount = claim_amount × random factor (0.50 to 0.95)
            #   This captures the discount without modeling the full
            #   negotiated rate + coinsurance calculation.

            "admission_date": claim_date if is_inpatient else None,
            # The date the patient was admitted to the hospital.
            # Only set for inpatient claims → same as claim_date.
            # NULL for all outpatient, pharmacy, emergency, preventive claims.
            # Stored separately from claim_date for clarity in reporting:
            #   "Show me all admissions in Q1 2023"

            "discharge_date": discharge,
            # The date the patient left the hospital.
            # Only set for inpatient claims → claim_date + length_of_stay.
            # NULL for all non-inpatient claims.
            # Used to calculate actual length of stay in SQL:
            #   SELECT DATEDIFF('day', admission_date, discharge_date)
            #   AS actual_los FROM fact_claims WHERE discharge_date IS NOT NULL

            "length_of_stay": los,
            # Number of days the patient was hospitalized.
            # 0 for all non-inpatient claims.
            # 1–14 for inpatient claims.
            # Stored directly for convenience — same as:
            #   DATEDIFF('day', admission_date, discharge_date)
            # Having it pre-calculated avoids repeated computation in queries.

            "submission_channel": random.choice(CHANNELS),
            # HOW the claim was submitted to the insurer.
            # Randomly chosen from:
            #   ["Online","Mobile App","Paper","Agent","Hospital Portal"]
            # This is about the TECHNICAL METHOD of filing — not where
            # the care was received or who the provider was.
            # Used for operational analytics:
            #   "What % of claims are still submitted on paper?"
            #   "Do online submissions have faster approval times?"

            "network_provider_flag": random.random() < 0.75,
            # Boolean — True if the provider used for this claim
            # is IN the insurer's network.
            # 75% of claims use in-network providers.
            # Mirrors the 75% network_flag rate in dim_providers.
            #
            # IMPACT ON MEMBER COSTS:
            #   True  → member pays lower out-of-pocket share
            #   False → member pays much more (or everything for HMO)
            #
            # NOTE: this field and dim_providers.network_flag should
            # ideally match for the same provider. In our simulation
            # they are assigned independently — a known simplification.

            "co_payment_amount": round(claim_amt * random.uniform(0.05, 0.20), 2),
            # The dollar amount the MEMBER pays for this specific claim.
            # Calculated as 5%–20% of claim_amount.
            #
            # In reality this would be calculated precisely as:
            #   co_payment_amount = MAX(0,
            #     MIN(claim_amount, deductible_remaining)
            #     + (claim_amount - deductible_remaining) × co_payment_rate
            #   )
            # Our simplified model uses a random percentage for simulation.
            #
            # This flows into fact_payments:
            #   fact_payments.co_payment_amount = this value
            #   representing what the member pays directly.

            "is_fraud_flagged": random.random() < 0.03,
            # Boolean — True if this claim has been flagged as
            # potentially fraudulent.
            # 3% of claims are flagged → ~1,500 out of 50,000.
            #
            # Real world insurance fraud examples:
            #   - Provider billing for services never rendered
            #   - Upcoding (billing for more expensive procedure than performed)
            #   - Duplicate claims (same service billed twice)
            #   - Identity theft (using someone else's insurance)
            #   - Kickback schemes between providers and patients
            #
            # Flagged claims typically get:
            #   → claim_status set to "Under Review"
            #   → payment held until investigation completes
            #   → possible referral to Special Investigations Unit (SIU)
            #
            # In our simulation, is_fraud_flagged and claim_status
            # are assigned independently — a known simplification.
            # In reality, is_fraud_flagged = True would always
            # result in claim_status = "Under Review".

        })

    return pd.DataFrame(rows)
    # Convert the 50,000 row list of dictionaries into a DataFrame.
    # Each dictionary = one claim = one row.
    # Result: 50,000 rows × 16 columns.
    #
    # COLUMN SUMMARY:
    #   Identifiers  : claim_id, policy_id, member_id, provider_id
    #   Dates        : claim_date, admission_date, discharge_date
    #   Classification: claim_type, submission_channel
    #   Amounts      : claim_amount, paid_amount, co_payment_amount
    #   Status       : claim_status
    #   Metrics      : length_of_stay
    #   Flags        : network_provider_flag, is_fraud_flagged


    
# ─────────────────────────────────────────────
# 8. FACT_CLAIM_DIAGNOSES
# ─────────────────────────────────────────────

def build_claim_diagnoses(claims: pd.DataFrame) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the fact_claim_diagnoses table.
    #   Records the medical diagnosis and procedure codes
    #   associated with each claim.
    #
    # PARAMETER:
    #   claims : pd.DataFrame → the fact_claims table
    #                           used to get the list of all claim_ids
    #                           so every claim gets at least one diagnosis
    #
    # RETURNS:
    #   A pandas DataFrame loaded into DuckDB as fact_claim_diagnoses.
    #   Will have MORE rows than fact_claims because each claim
    #   can have 1, 2, or 3 diagnosis/procedure pairs.
    #   Approximately:
    #     60% of claims → 1 diagnosis  = 30,000 rows
    #     30% of claims → 2 diagnoses  = 30,000 rows
    #     10% of claims → 3 diagnoses  = 15,000 rows
    #     Total → approximately 75,000 rows for 50,000 claims
    #
    # WHY IS THIS A SEPARATE TABLE AND NOT COLUMNS ON fact_claims?
    #   This is one of the most important normalization decisions
    #   in the entire database. There are three bad alternatives
    #   and one correct solution:
    #
    #   BAD OPTION 1 — Multiple columns on fact_claims:
    #     claim_id  diagnosis_1  diagnosis_2  diagnosis_3
    #     CLM001    I10          E11          NULL
    #     CLM002    J45          NULL         NULL
    #     → violates 1NF (repeating groups of columns)
    #     → wastes space (NULL columns for single-diagnosis claims)
    #     → breaks if a claim ever needs a 4th diagnosis
    #     → cannot easily query "all claims with diagnosis I10"
    #
    #   BAD OPTION 2 — Comma-separated values in one column:
    #     claim_id  diagnoses
    #     CLM001    "I10,E11"
    #     CLM002    "J45"
    #     → violates 1NF (multiple values in one cell)
    #     → impossible to filter or aggregate by individual diagnosis
    #     → no referential integrity possible
    #
    #   BAD OPTION 3 — Duplicate claim rows:
    #     claim_id  diagnosis_code
    #     CLM001    I10
    #     CLM001    E11             ← duplicate claim_id
    #     CLM002    J45
    #     → violates 2NF (claim details repeated for each diagnosis)
    #     → claim_amount, paid_amount etc. would be duplicated
    #     → aggregations would double-count monetary amounts
    #
    #   CORRECT SOLUTION — Separate child table (this approach):
    #     fact_claims has ONE row per claim (no diagnosis info)
    #     fact_claim_diagnoses has ONE row per diagnosis per claim
    #     linked by claim_id foreign key
    #     → each diagnosis stored exactly once
    #     → unlimited diagnoses per claim without schema changes
    #     → clean aggregations on fact_claims (no double counting)
    #     → easy filtering: "all claims with diagnosis I10"
    #
    # PARENT-CHILD RELATIONSHIP:
    #   fact_claims          = PARENT table (one row per claim)
    #   fact_claim_diagnoses = CHILD table  (many rows per claim)
    #
    #   fact_claims.claim_id (1)
    #       └──→ fact_claim_diagnoses.claim_id (many)
    #
    #   This is a ONE-TO-MANY relationship:
    #     One claim → many diagnosis rows
    #     One diagnosis row → belongs to exactly one claim
    #
    # HOW fact_claim_diagnoses CONNECTS TO THE REST:
    #
    #   fact_claims
    #       └── fact_claim_diagnoses.claim_id
    #               (diagnosis details for each claim)
    #
    #   This table has NO other foreign keys.
    #   All context (member, provider, policy) is retrieved
    #   by first joining to fact_claims, then to dimension tables.
    # ═══════════════════════════════════════════════════════
    rows = []
    diag_counter = 1
    # diag_counter tracks the diagnosis_id sequence.
    # Starts at 1 and increments for every diagnosis row created.
    # Will reach approximately 75,000 by the end of the loop.

    for cid in claims["claim_id"]:
        # Loop through EVERY claim_id in fact_claims.
        # claims["claim_id"] = a pandas Series of all 50,000 claim IDs.
        # Every single claim MUST get at least one diagnosis row —
        # in real insurance a claim without a diagnosis code
        # would be automatically rejected by the insurer.

        n_diags = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        # How many diagnoses does THIS claim have?
        # Chosen with WEIGHTED probability:
        #   1 diagnosis  → 60% of claims
        #                  Most routine visits have one primary diagnosis:
        #                  "Patient came in for back pain" → M54 only
        #
        #   2 diagnoses  → 30% of claims
        #                  Common when a secondary condition is also noted:
        #                  "Patient came in for back pain, also has hypertension"
        #                  → M54 (primary) + I10 (secondary)
        #
        #   3 diagnoses  → 10% of claims
        #                  Complex patients with multiple conditions:
        #                  "Diabetic patient admitted for chest pain,
        #                   also has obesity"
        #                  → I25 (primary) + E11 (secondary) + E66 (tertiary)
        #
        # random.choices() returns a LIST → [0] takes the single value.
        # weights=[60,30,10] maps to [1,2,3] in order.

        for rank in range(1, n_diags + 1):
            # Loop n_diags times — once per diagnosis for this claim.
            # rank starts at 1 (not 0) because in medical coding:
            #   rank 1 = PRIMARY diagnosis   (main reason for visit)
            #   rank 2 = SECONDARY diagnosis (additional condition noted)
            #   rank 3 = TERTIARY diagnosis  (third condition noted)
            #
            # range(1, n_diags + 1) produces:
            #   n_diags=1 → [1]       (just primary)
            #   n_diags=2 → [1, 2]    (primary + secondary)
            #   n_diags=3 → [1, 2, 3] (primary + secondary + tertiary)

            rows.append({

                "diagnosis_id": f"DGN{diag_counter:09d}",
                # Unique identifier for this specific diagnosis row.
                # Format: DGN000000001, DGN000000002, ... DGN000075000
                #   "DGN"          → prefix identifying this as a diagnosis
                #   {diag_counter} → increments across ALL claims
                #   :09d           → 9 digits (75,000 rows needs only 5
                #                    but 9 digits future-proofs the schema)
                # PRIMARY KEY of fact_claim_diagnoses.
                # Note: this is different from claim_id —
                # multiple diagnosis_ids can share the same claim_id.

                "claim_id": cid,
                # Foreign key → fact_claims.claim_id
                # Links this diagnosis row back to its parent claim.
                # This is the JOIN key used in almost every query
                # that combines claim details with diagnosis details:
                #   SELECT c.claim_amount, d.diagnosis_code
                #   FROM fact_claims c
                #   JOIN fact_claim_diagnoses d ON c.claim_id = d.claim_id
                #
                # For a claim with 3 diagnoses, all 3 rows share
                # the SAME claim_id — that is the point of this design.

                "diagnosis_rank": rank,
                # The priority order of this diagnosis for this claim.
                # Values: 1, 2, or 3.
                #
                # WHY RANK MATTERS:
                #   Rank 1 (PRIMARY) is the MOST IMPORTANT:
                #     → The main reason the patient sought care
                #     → Drives the reimbursement decision
                #     → Used for population health statistics
                #     → Insurers look at primary diagnosis to determine
                #       if the service was medically necessary
                #
                #   Rank 2 (SECONDARY):
                #     → Additional condition documented during the visit
                #     → May affect treatment complexity
                #     → Affects risk scoring and case mix calculations
                #
                #   Rank 3 (TERTIARY):
                #     → Third condition — often a chronic background condition
                #     → Example: a diabetic patient who came in for a cold
                #       → rank 1: J06 (cold)
                #       → rank 2: I10 (hypertension, managed)
                #       → rank 3: E11 (diabetes, chronic background)
                #
                # FILTERING TIP:
                #   Always filter by diagnosis_rank = 1 when you want
                #   to count claims by primary diagnosis:
                #     SELECT diagnosis_code, COUNT(*)
                #     FROM fact_claim_diagnoses
                #     WHERE diagnosis_rank = 1
                #     GROUP BY diagnosis_code
                #   Without this filter, claims with 3 diagnoses
                #   would be counted 3 times.

                "diagnosis_code": random.choice(ICD10_CODES),
                # The ICD-10 medical diagnosis code for this entry.
                # Randomly chosen from the 16 codes in ICD10_CODES.
                # Identifies WHAT CONDITION the patient has.
                # Examples: I10 (hypertension), E11 (diabetes), J45 (asthma)
                #
                # In a real insurance system:
                #   → Providers assign codes based on clinical documentation
                #   → Wrong codes = claim rejection or fraud investigation
                #   → Codes must match the procedure performed
                #     (you cannot bill for cardiac surgery with a cold diagnosis)
                #
                # In our simulation:
                #   → Codes are assigned randomly — no clinical logic
                #   → A known simplification — in reality diagnosis and
                #     procedure codes would be clinically correlated
                #   → Future improvement: pair specific diagnosis codes
                #     with matching procedure codes (e.g. I10 → 93000 ECG)

                "procedure_code": random.choice(PROCEDURE_CODES),
                # The CPT procedure code describing WHAT SERVICE was performed.
                # Randomly chosen from the 12 codes in PROCEDURE_CODES.
                # Identifies WHAT WAS DONE to address the diagnosis.
                # Examples: 99213 (office visit), 85025 (blood test), 71046 (X-ray)
                #
                # DIAGNOSIS + PROCEDURE = the complete clinical picture:
                #   diagnosis_code = WHY the patient needed care
                #   procedure_code = WHAT was done to treat them
                #
                # Together they answer the insurer's core question:
                #   "Was this service medically necessary and appropriate
                #    for this diagnosis, and how much should we pay?"
                #
                # In a real system, certain procedure codes are only
                # valid with certain diagnosis codes — called
                # "diagnosis-procedure code pairing rules".
                # Our simulation assigns them independently (simplification).

            })
            diag_counter += 1
            # Increment AFTER appending the row so the next
            # diagnosis gets a unique sequential ID.

    print(f"  Building fact_claim_diagnoses ({len(rows):>6,} rows)...")
    return pd.DataFrame(rows)
    # Convert accumulated list of dictionaries into a DataFrame.
    # Result: ~75,000 rows × 5 columns:
    #   diagnosis_id, claim_id, diagnosis_rank,
    #   diagnosis_code, procedure_code
    #
    # QUICK SANITY CHECK after building:
    #   len(rows) should be between 50,000 (all single diagnosis)
    #   and 150,000 (all triple diagnosis).
    #   Expected ~75,000 based on weighted probabilities:
    #     50,000 × (0.60×1 + 0.30×2 + 0.10×3)
    #   = 50,000 × (0.60 + 0.60 + 0.30)
    #   = 50,000 × 1.50
    #   = 75,000 rows expected

    
# ─────────────────────────────────────────────
# 9. FACT_PAYMENTS
# ─────────────────────────────────────────────

def build_payments(claims: pd.DataFrame) -> pd.DataFrame:
    # ═══════════════════════════════════════════════════════
    # PURPOSE:
    #   Builds the fact_payments table.
    #   Records the actual financial transactions made by
    #   the insurer to settle approved claims.
    #
    # PARAMETER:
    #   claims : pd.DataFrame → the fact_claims table
    #                           used to filter ONLY approved claims
    #                           because only approved claims
    #                           generate a payment record
    #
    # RETURNS:
    #   A pandas DataFrame loaded into DuckDB as fact_payments.
    #   Will have FEWER rows than fact_claims because only
    #   ~65% of claims are Approved → ~32,500 payment rows.
    #
    # THE FUNDAMENTAL RULE OF THIS TABLE:
    #   ONE approved claim → generates EXACTLY ONE payment record.
    #   No approved claim should ever have zero payment records.
    #   No rejected, pending, or under-review claim should ever
    #   have a payment record.
    #
    #   claim_status = "Approved"     → 1 row in fact_payments ✓
    #   claim_status = "Rejected"     → 0 rows in fact_payments ✓
    #   claim_status = "Pending"      → 0 rows in fact_payments ✓
    #   claim_status = "Under Review" → 0 rows in fact_payments ✓
    #
    # WHY IS THIS A SEPARATE TABLE FROM fact_claims?
    #   You might wonder: why not just add payment columns
    #   directly to fact_claims?
    #
    #   Reason 1 — SEPARATION OF CONCERNS:
    #     A claim is a REQUEST for reimbursement.
    #     A payment is a FINANCIAL TRANSACTION settling that request.
    #     These are genuinely different business events:
    #       Claim filed:   March 1, 2023  (claim_date)
    #       Claim approved: March 15, 2023
    #       Payment made:  March 20, 2023 (payment_date)
    #     They happen at DIFFERENT TIMES and involve different systems.
    #     Claims processing = medical/clinical system
    #     Payments processing = financial/accounting system
    #
    #   Reason 2 — ONE CLAIM CAN HAVE MULTIPLE PAYMENTS:
    #     In real insurance, a single claim can be paid in stages:
    #       Partial payment 1: $500 (initial payment)
    #       Partial payment 2: $300 (after appeal)
    #       Reversal:         -$800 (if overpayment discovered)
    #     A separate table handles this naturally — our simulation
    #     uses one payment per claim for simplicity but the schema
    #     supports multiple payments if needed.
    #
    #   Reason 3 — NOT ALL CLAIMS HAVE PAYMENTS:
    #     Adding payment columns to fact_claims would mean
    #     ~35,000 rows (rejected/pending/under review) have
    #     NULL payment values — wasted space and confusing queries.
    #     A separate table contains ONLY rows where payment occurred.
    #
    #   Reason 4 — DIFFERENT ANALYTICS NEEDS:
    #     Claims analytics: "How many claims were filed per month?"
    #     Payment analytics: "How much did we pay out per month?"
    #     "What is the average days between claim_date and payment_date?"
    #     "How many payments were reversed?"
    #     These questions are cleaner with separate tables.
    #
    # HOW fact_payments CONNECTS TO THE REST OF THE DATABASE:
    #
    #   fact_claims.claim_id (1)
    #       └──→ fact_payments.claim_id  (0 or 1)
    #
    #   fact_payments also has:
    #       → dim_members.member_id
    #         (denormalized shortcut — member_id is already on fact_claims
    #          but stored here too for faster payment-focused queries
    #          without needing to join through fact_claims first)
    #
    # POSITION IN THE DATA FLOW:
    #
    #   Member visits provider
    #       → fact_claims created          (claim filed)
    #           → fact_claim_diagnoses     (diagnosis recorded)
    #               → claim_status = Approved
    #                   → fact_payments created  ← THIS TABLE
    #                       → money moves from insurer to provider
    # ═══════════════════════════════════════════════════════

    approved = claims[claims["claim_status"] == "Approved"].copy()
    # FILTER: keep ONLY approved claims.
    # claims["claim_status"] == "Approved" creates a boolean mask:
    #   True  for every row where claim_status = "Approved"
    #   False for Rejected, Pending, Under Review
    # claims[mask] applies the filter → keeps only True rows.
    # .copy() creates an independent copy of the filtered DataFrame.
    #
    # WHY .copy()?
    #   Without .copy(), pandas may give a SettingWithCopyWarning
    #   if we try to modify the filtered DataFrame later.
    #   .copy() prevents this by making a completely independent copy.
    #
    # RESULT:
    #   approved contains ~32,500 rows (65% of 50,000 claims).
    #   Each row will generate exactly one payment record.

    print(f"  Building fact_payments       ({len(approved):>6,} rows)...")

    rows = []

    for i, (_, row) in enumerate(approved.iterrows()):
        # Loop through every approved claim — one iteration = one payment.
        #
        # enumerate() wraps iterrows() to give us BOTH:
        #   i   → a sequential counter (0, 1, 2, ...)
        #          used to generate payment_id
        #   _   → the DataFrame index (we don't need it → named _)
        #   row → the current claim row as a pandas Series
        #          contains all columns of that approved claim
        #
        # ACCESSING CLAIM DATA:
        #   row["claim_id"]          → the claim this payment settles
        #   row["member_id"]         → who received the care
        #   row["claim_date"]        → when the service happened
        #   row["paid_amount"]       → how much the insurer pays
        #   row["co_payment_amount"] → how much the member pays

        rows.append({

            "payment_id": f"PAY{i+1:08d}",
            # Unique identifier for this payment transaction.
            # Format: PAY00000001, PAY00000002, ... PAY00032500
            #   "PAY"  → prefix identifying this as a payment record
            #   {i+1}  → counter starting at 1 (not 0)
            #   :08d   → always 8 digits
            # PRIMARY KEY of fact_payments.
            # Note: payment_id is DIFFERENT from claim_id —
            # a payment is a separate financial event from the claim.

            "claim_id": row["claim_id"],
            # Foreign key → fact_claims.claim_id
            # Links this payment back to the claim it settles.
            # The most important JOIN key in this table:
            #   SELECT c.claim_amount, p.paid_amount, p.payment_date
            #   FROM fact_claims c
            #   JOIN fact_payments p ON c.claim_id = p.claim_id
            #
            # INTEGRITY RULE:
            #   Every claim_id here must exist in fact_claims.
            #   Every claim_id here must have claim_status = "Approved".
            #   No claim_id should appear more than once
            #   (one payment per claim in our simplified model).

            "member_id": row["member_id"],
            # Foreign key → dim_members.member_id
            # Who received the care that this payment covers?
            #
            # WHY IS member_id HERE WHEN IT IS ALREADY ON fact_claims?
            #   This is a deliberate DENORMALIZATION for performance.
            #   Strictly speaking, member_id is redundant here because:
            #     fact_payments.claim_id → fact_claims.member_id
            #   But storing member_id directly on fact_payments allows:
            #     "Total payments per member" query WITHOUT joining fact_claims:
            #       SELECT member_id, SUM(paid_amount)
            #       FROM fact_payments
            #       GROUP BY member_id
            #   This is a common analytics query that runs faster
            #   without the extra JOIN.
            #   This is an accepted exception to strict 3NF when
            #   query performance justifies it.

            "payment_date": row["claim_date"] + timedelta(days=random.randint(3, 45)),
            # The date the money was actually transferred to the provider.
            # Calculated as: claim_date + random 3 to 45 days.
            #
            # WHY NOT THE SAME AS claim_date?
            #   In real insurance, there is ALWAYS a delay between:
            #     1. The claim being filed (claim_date)
            #     2. The claim being reviewed and approved
            #     3. The payment being processed and sent
            #   This delay is called the "claims lag" or "payment lag".
            #
            # Our range (3–45 days) models realistic scenarios:
            #   3–7 days   → fast electronic claims (Hospital Portal/Online)
            #                simple outpatient visits, auto-approved
            #   7–21 days  → standard processing time
            #                most routine claims fall here
            #   21–45 days → slower processing
            #                complex claims, paper submissions,
            #                claims requiring manual review before approval
            #
            # ANALYTICS USE CASES:
            #   "What is the average payment lag by submission channel?"
            #   "Do online submissions get paid faster than paper?"
            #   "What is the average days to payment by provider type?"
            #   DATEDIFF('day', c.claim_date, p.payment_date) AS days_to_payment

            "paid_amount": row["paid_amount"],
            # The dollar amount the insurer pays to the provider.
            # Copied directly from fact_claims.paid_amount.
            # This is the INSURER'S share of the bill after:
            #   → Applying the negotiated rate discount
            #   → Subtracting the member's deductible contribution
            #   → Subtracting the member's coinsurance share
            #
            # RELATIONSHIP TO OTHER AMOUNTS:
            #   claim_amount     = what provider billed ($1,000)
            #   paid_amount      = what insurer pays   ($700)
            #   co_payment_amount = what member pays    ($140)
            #   contractual_adj  = written off amount   ($160)
            #                      (claim_amount - paid_amount - co_payment_amount)
            #
            # KEY METRIC for financial analysis:
            #   "Total paid_amount by month" = insurer's monthly cash outflow
            #   "paid_amount / claim_amount" = payment ratio (efficiency metric)

            "co_payment_amount": row["co_payment_amount"],
            # The dollar amount the MEMBER pays for this claim.
            # Copied directly from fact_claims.co_payment_amount.
            # This is what the member owes to the provider directly —
            # separate from the insurer's payment.
            #
            # HOW CO-PAYMENT WORKS IN PRACTICE:
            #   At the doctor's office:
            #     Member pays $140 at the front desk (co_payment_amount)
            #   Later, insurer sends $700 to the provider (paid_amount)
            #   Provider receives total: $840
            #   Provider writes off: $160 (contractual adjustment)
            #
            # WHY STORED ON BOTH fact_claims AND fact_payments?
            #   fact_claims.co_payment_amount → what was calculated
            #                                   at time of claim filing
            #   fact_payments.co_payment_amount → what was actually collected
            #   In a real system these might differ (after appeals, adjustments)
            #   In our simulation they are identical — a simplification.

            "payment_method": random.choice(["Bank Transfer","Check",
                                             "Direct Deposit","Digital Wallet"]),
            # HOW the money was transferred from insurer to provider.
            # Randomly chosen with equal probability from 4 methods.
            #
            #   "Bank Transfer"   → Electronic fund transfer between bank accounts
            #                       Most common for large hospital payments
            #                       Fast, traceable, low cost
            #
            #   "Check"           → Physical paper check mailed to provider
            #                       Old-fashioned but still used by some
            #                       small providers or rural practices
            #                       Slowest method — adds days to payment lag
            #
            #   "Direct Deposit"  → Automatic recurring deposit to provider's account
            #                       Common for high-volume providers with
            #                       established electronic relationships
            #                       Fastest and most efficient method
            #
            #   "Digital Wallet"  → Modern electronic payment platforms
            #                       Growing in usage for smaller providers,
            #                       pharmacies, and individual physicians
            #
            # ANALYTICS USE CASES:
            #   "What % of payments are still made by check?"
            #   "Do check payments have longer days_to_payment?"
            #   "Which provider types prefer which payment methods?"

            "payment_status": random.choices(
                                   ["Completed","Failed","Reversed"],
                                   weights=[92, 5, 3])[0],
            # The final outcome of this payment transaction.
            # Assigned with WEIGHTED probability:
            #   "Completed" 92% → ~29,900 payments
            #                     Money successfully transferred.
            #                     The normal expected outcome.
            #                     Claim is fully settled.
            #
            #   "Failed"     5% → ~1,625 payments
            #                     Payment attempt was unsuccessful.
            #                     Reasons:
            #                       - Invalid bank account details
            #                       - Provider bank account closed
            #                       - Technical system failure
            #                       - Insufficient funds (rare for insurers)
            #                     A failed payment triggers a retry process.
            #                     Claim remains technically unpaid.
            #
            #   "Reversed"   3% → ~975 payments
            #                     Payment was made but then CLAWED BACK.
            #                     Reasons:
            #                       - Fraud discovered after payment
            #                       - Duplicate payment identified
            #                       - Claim re-adjudicated to lower amount
            #                       - Provider returned payment due to error
            #                     A reversal is a serious event — it means
            #                     money already sent must be recovered.
            #                     Often triggers a new corrected payment.
            #
            # NOTE: In our simulation, payment_status is assigned randomly
            # and independently of other fields. In reality:
            #   is_fraud_flagged = True → higher chance of Reversed
            #   payment_method = "Check" → higher chance of Failed
            #   These correlations are a known simplification.

        })

    return pd.DataFrame(rows)
    # Convert the ~32,500 row list of dictionaries into a DataFrame.
    # Result: ~32,500 rows × 8 columns:
    #   payment_id, claim_id, member_id, payment_date,
    #   paid_amount, co_payment_amount, payment_method, payment_status
    #
    # FINAL SUMMARY OF ALL FACT TABLES:
    #
    #   fact_claims             50,000 rows  → one per medical event
    #   fact_claim_diagnoses   ~75,000 rows  → one per diagnosis per claim
    #   fact_payments          ~32,500 rows  → one per approved claim
    #
    # THE COMPLETE FINANCIAL STORY OF ONE CLAIM:
    #
    #   March 1:  Member visits hospital (provider)
    #             → fact_claims row created
    #               claim_amount = $1,000
    #               claim_status = "Pending"
    #
    #   March 1:  Hospital records diagnosis and procedure
    #             → fact_claim_diagnoses row created
    #               diagnosis_code = I25 (heart disease)
    #               procedure_code = 93000 (ECG)
    #
    #   March 15: Insurer reviews and approves the claim
    #             → fact_claims.claim_status updated to "Approved"
    #               paid_amount      = $700
    #               co_payment_amount = $140
    #
    #   March 20: Payment processed
    #             → fact_payments row created
    #               payment_date   = March 20
    #               paid_amount    = $700  (insurer → provider)
    #               co_payment_amount = $140 (member → provider, already paid)
    #               payment_method = "Bank Transfer"
    #               payment_status = "Completed"


