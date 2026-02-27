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
#    Represents a household group.
#    family_size >= 2 because a single person
#    would buy an Individual contract, not a Family one.
# ─────────────────────────────────────────────
def build_families(n: int) -> pd.DataFrame:
    print(f"  Building dim_families        ({n:>6,} rows)...")
    return pd.DataFrame([{
        "family_id":       f"FAM{i+1:06d}",
        "family_size":     random.randint(2, 7),
        "residential_zip": fake.zipcode(),
        "state":           fake.state_abbr(),
        "income_category": random.choice(["Low","Middle","High"]),
    } for i in range(n)])


# ─────────────────────────────────────────────
# 2. DIM_EMPLOYERS
#    Represents a company that buys a Group contract
#    from the insurer on behalf of its employees.
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


# ─────────────────────────────────────────────
# 3. DIM_MEMBERS
#    Every human being covered by any insurance
#    gets exactly ONE row here with a unique member_id.
#
#    IMPORTANT CHANGES FROM v2:
#    ✓  NO insurance_category column here anymore
#       → that belongs to dim_contracts, not the person
#    ✓  family_id and employer_id are REMOVED from here
#       → those relationships are captured through
#         contracts and policies instead
#    ✓  A member is just a PERSON — their coverage
#       type is determined by what contract covers them
# ─────────────────────────────────────────────
def build_members(n: int) -> pd.DataFrame:
    print(f"  Building dim_members         ({n:>6,} rows)...")
    rows = []
    for i in range(n):
        rows.append({
            "member_id":              f"MEM{i+1:07d}",
            # Pure demographics — no coverage info here (belongs to contracts/policies)
            "first_name":             fake.first_name(),
            "last_name":              fake.last_name(),
            "date_of_birth":          rand_date(date(1940,1,1), date(2005,12,31)),
            "gender":                 random.choice(GENDERS),
            "marital_status":         random.choice(["Single","Married","Divorced","Widowed"]),
            "zip_code":               fake.zipcode(),
            "state":                  fake.state_abbr(),
            "employment_category":    random.choice(EMPLOYMENT_CATS),
            # Health info — belongs to the person, not the contract
            "chronic_condition_flag": random.random() < 0.25,
            "health_risk_score":      round(np.random.beta(2, 5) * 100, 2),
            "smoking_status":         random.choice(["Never","Former","Current"]),
            "bmi":                    round(random.gauss(27, 5), 1),
            "msa_balance":            round(random.uniform(0, 5_000), 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 4. DIM_PROVIDERS
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
    print(f"  Building dim_contracts...")
    rows = []
    contract_counter = 1

    def new_contract(ctype, family_id=None, employer_id=None):
        nonlocal contract_counter
        c = {
            "contract_id":   f"CON{contract_counter:07d}",
            "contract_type": ctype,
            # WHO is the buyer?
            "family_id":     family_id,    # set only for Family contracts
            "employer_id":   employer_id,  # set only for Group contracts
            "start_date":    rand_date(date(2018,1,1), date(2023,1,1)),
            "status":        random.choices(
                                 ["Active","Expired","Cancelled"],
                                 weights=[70,20,10])[0],
        }
        contract_counter += 1
        return c

    # One Individual contract per individual member slot
    for _ in range(n_individual):
        rows.append(new_contract("Individual"))

    # One Family contract per unique family
    for fam_id in families["family_id"]:
        rows.append(new_contract("Family", family_id=fam_id))

    # One Group contract per employer
    for emp_id in employers["employer_id"]:
        rows.append(new_contract("Group", employer_id=emp_id))

    df = pd.DataFrame(rows)
    print(f"    → {len(df):,} contracts total  "
          f"({n_individual:,} Individual / "
          f"{len(families):,} Family / "
          f"{len(employers):,} Group)")
    return df


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

    policy_rows = []
    bridge_rows = []
    policy_counter = 1
    bridge_counter = 1

    def new_policy(contract_id: str, coverage_level: str) -> dict:
        """Creates one policy record."""
        nonlocal policy_counter
        start = rand_date(date(2018,1,1), date(2024,1,1))
        p = {
            "policy_id":       f"POL{policy_counter:07d}",
            "contract_id":     contract_id,     # FK → dim_contracts
            # coverage_level = what level of coverage this policy provides
            # "Individual" → covers one person only
            # "Family"     → covers a whole family unit
            "coverage_level":  coverage_level,
            "plan_type":       random.choice(PLAN_TYPES),
            "start_date":      start,
            "end_date":        start + timedelta(days=365),
            "deductible":      random.choice([500,1000,2000,3000,5000]),
            "co_payment_rate": round(random.uniform(0.05, 0.30), 2),
            "out_of_pocket_max": random.choice([3000,5000,7000,10000]),
            "premium_amount":  round(random.uniform(150, 900), 2),
            "status":          random.choices(
                                   ["Active","Expired","Cancelled"],
                                   weights=[70,20,10])[0],
        }
        policy_counter += 1
        return p

    def add_bridge(policy_id: str, member_id: str, role: str):
        nonlocal bridge_counter
        bridge_rows.append({
            "bridge_id":  f"BRG{bridge_counter:09d}",
            "policy_id":  policy_id,
            "member_id":  member_id,
            "role":       role,
        })
        bridge_counter += 1

    # Prepare member pool and family membership lookup
    all_member_ids  = members["member_id"].tolist()
    member_idx      = 0   # pointer to assign members sequentially

    # Build a lookup of family_id → list of member_ids
    # We will assign members to families as we go
    family_ids      = families["family_id"].tolist()
    family_members  = {fid: [] for fid in family_ids}

    # ── INDIVIDUAL CONTRACTS ───────────────────────────────────────────────
    ind_contracts = contracts[contracts["contract_type"] == "Individual"]
    print(f"    → Processing {len(ind_contracts):,} Individual contracts...")

    for _, contract in ind_contracts.iterrows():
        if member_idx >= len(all_member_ids):
            break
        member_id = all_member_ids[member_idx]
        member_idx += 1

        pol = new_policy(contract["contract_id"], "Individual")
        policy_rows.append(pol)
        add_bridge(pol["policy_id"], member_id, "Policyholder")

    # ── FAMILY CONTRACTS ───────────────────────────────────────────────────
    fam_contracts = contracts[contracts["contract_type"] == "Family"]
    print(f"    → Processing {len(fam_contracts):,} Family contracts...")

    for _, contract in fam_contracts.iterrows():
        fam_id       = contract["family_id"]
        family_row   = families[families["family_id"] == fam_id].iloc[0]
        family_size  = family_row["family_size"]

        # Assign 'family_size' members to this family
        fam_member_ids = []
        for _ in range(family_size):
            if member_idx >= len(all_member_ids):
                break
            fam_member_ids.append(all_member_ids[member_idx])
            member_idx += 1

        if not fam_member_ids:
            continue

        # ONE policy for the ENTIRE family — no exceptions
        pol = new_policy(contract["contract_id"], "Family")
        policy_rows.append(pol)

        # All members share this same policy
        for i, mid in enumerate(fam_member_ids):
            role = "Policyholder" if i == 0 else "Dependent"
            add_bridge(pol["policy_id"], mid, role)
            family_members[fam_id].append(mid)

    # ── GROUP CONTRACTS ────────────────────────────────────────────────────
    grp_contracts = contracts[contracts["contract_type"] == "Group"]
    print(f"    → Processing {len(grp_contracts):,} Group contracts...")

    for _, contract in grp_contracts.iterrows():
        emp_id = contract["employer_id"]

        # Assign a random number of employees to this employer (10–80)
        n_employees = random.randint(10, 80)
        emp_member_ids = []
        for _ in range(n_employees):
            if member_idx >= len(all_member_ids):
                break
            emp_member_ids.append(all_member_ids[member_idx])
            member_idx += 1

        if not emp_member_ids:
            continue

        for employee_id in emp_member_ids:
            # Each employee independently chooses their coverage level:
            # 40% choose Individual, 60% choose Family
            chosen_level = random.choices(
                ["Individual", "Family"],
                weights=[40, 60]
            )[0]

            pol = new_policy(contract["contract_id"], chosen_level)
            policy_rows.append(pol)

            if chosen_level == "Individual":
                # Policy covers only the employee
                add_bridge(pol["policy_id"], employee_id, "Employee")

            else:
                # Policy covers employee + their family members
                # Assign 1–4 family members to this employee
                add_bridge(pol["policy_id"], employee_id, "Employee")
                n_dependents = random.randint(1, 4)
                for _ in range(n_dependents):
                    if member_idx >= len(all_member_ids):
                        break
                    dep_id = all_member_ids[member_idx]
                    member_idx += 1
                    # All dependents share the SAME policy as the employee
                    # (no mixing within a family — v3 rule enforced here)
                    add_bridge(pol["policy_id"], dep_id, "Dependent")

    policies_df = pd.DataFrame(policy_rows)
    bridge_df   = pd.DataFrame(bridge_rows)

    print(f"  Building dim_policies        ({len(policies_df):>6,} rows)")
    print(f"  Building policy_members      ({len(bridge_df):>6,} rows)  [Bridge table]")

    return policies_df, bridge_df


# ─────────────────────────────────────────────
# 7. FACT_CLAIMS
#    Every claim is filed by a member under a policy
#    that actually covers them — enforced via bridge lookup.
# ─────────────────────────────────────────────
def build_claims(n: int, bridge: pd.DataFrame,
                 providers: pd.DataFrame) -> pd.DataFrame:
    print(f"  Building fact_claims         ({n:>6,} rows)...")

    # Build lookup: member_id → list of policy_ids covering them
    member_policies = (
        bridge.groupby("member_id")["policy_id"]
              .apply(list)
              .to_dict()
    )

    # Only use members that actually have a policy
    covered_members = list(member_policies.keys())
    provider_ids    = providers["provider_id"].tolist()
    rows = []

    for i in range(n):
        member_id  = random.choice(covered_members)
        policy_id  = random.choice(member_policies[member_id])
        claim_date = rand_date(date(2019,1,1), date(2024,12,31))
        is_inpatient = random.random() < 0.20
        los          = random.randint(1, 14) if is_inpatient else 0
        discharge    = claim_date + timedelta(days=los) if is_inpatient else None
        claim_amt    = round(np.random.lognormal(7, 1.2), 2)

        rows.append({
            "claim_id":              f"CLM{i+1:08d}",
            "policy_id":             policy_id,
            "member_id":             member_id,
            "provider_id":           random.choice(provider_ids),
            "claim_date":            claim_date,
            "claim_type":            random.choice(CLAIM_TYPES),
            "claim_status":          random.choices(CLAIM_STATUSES,
                                         weights=[65,10,15,10])[0],
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


# ─────────────────────────────────────────────
# 8. FACT_CLAIM_DIAGNOSES
# ─────────────────────────────────────────────
def build_claim_diagnoses(claims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    diag_counter = 1
    for cid in claims["claim_id"]:
        n_diags = random.choices([1,2,3], weights=[60,30,10])[0]
        for rank in range(1, n_diags + 1):
            rows.append({
                "diagnosis_id":   f"DGN{diag_counter:09d}",
                "claim_id":       cid,
                "diagnosis_rank": rank,
                "diagnosis_code": random.choice(ICD10_CODES),
                "procedure_code": random.choice(PROCEDURE_CODES),
            })
            diag_counter += 1
    print(f"  Building fact_claim_diagnoses ({len(rows):>6,} rows)...")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 9. FACT_PAYMENTS
# ─────────────────────────────────────────────
def build_payments(claims: pd.DataFrame) -> pd.DataFrame:
    approved = claims[claims["claim_status"] == "Approved"].copy()
    print(f"  Building fact_payments       ({len(approved):>6,} rows)...")
    rows = []
    for i, (_, row) in enumerate(approved.iterrows()):
        rows.append({
            "payment_id":        f"PAY{i+1:08d}",
            "claim_id":          row["claim_id"],
            "member_id":         row["member_id"],
            "payment_date":      row["claim_date"] + timedelta(days=random.randint(3,45)),
            "paid_amount":       row["paid_amount"],
            "co_payment_amount": row["co_payment_amount"],
            "payment_method":    random.choice(["Bank Transfer","Check",
                                                "Direct Deposit","Digital Wallet"]),
            "payment_status":    random.choices(
                                     ["Completed","Failed","Reversed"],
                                     weights=[92,5,3])[0],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Health Insurance DuckDB — NORMALIZED DATABASE v3")
    print("  NEW: dim_contracts / mixed Group coverage / family rule")
    print("=" * 65)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Removed existing '{DB_PATH}'")

    print("\n[1/2] Generating simulated data...")

    families  = build_families(N_FAMILIES)
    employers = build_employers(N_EMPLOYERS)
    members   = build_members(N_MEMBERS)
    providers = build_providers(N_PROVIDERS)

    # Individual contracts — enough to cover remaining members
    # after families and group employees are assigned
    # Approximate: 1/3 of members will be Individual
    n_individual_contracts = N_MEMBERS // 3
    contracts = build_contracts(families, employers, n_individual_contracts)

    policies, bridge = build_policies_and_bridge(members, contracts, families)
    claims    = build_claims(N_CLAIMS, bridge, providers)
    diagnoses = build_claim_diagnoses(claims)
    payments  = build_payments(claims)

    print("\n[2/2] Loading into DuckDB...")
    con = duckdb.connect(DB_PATH)

    table_map = {
        "dim_families":         families,
        "dim_employers":        employers,
        "dim_members":          members,
        "dim_providers":        providers,
        "dim_contracts":        contracts,    # ← NEW
        "dim_policies":         policies,
        "policy_members":       bridge,
        "fact_claims":          claims,
        "fact_claim_diagnoses": diagnoses,
        "fact_payments":        payments,
    }

    for name, df in table_map.items():
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        marker = " ← NEW" if name == "dim_contracts" else ""
        print(f"  ✓  {name:<28} {count:>8,} rows{marker}")

    # ── Indexes
    print("\n  Adding indexes...")
    indexes = [
        ("idx_contracts_family",   "dim_contracts",        "family_id"),
        ("idx_contracts_employer", "dim_contracts",        "employer_id"),
        ("idx_policies_contract",  "dim_policies",         "contract_id"),
        ("idx_bridge_policy",      "policy_members",       "policy_id"),
        ("idx_bridge_member",      "policy_members",       "member_id"),
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

    # ── Verification
    print("\n" + "─" * 65)
    print("  VERIFICATION 1: Coverage distribution by contract type")
    r1 = con.execute("""
        SELECT
            c.contract_type,
            COUNT(DISTINCT c.contract_id)  AS num_contracts,
            COUNT(DISTINCT p.policy_id)    AS num_policies,
            COUNT(DISTINCT pm.member_id)   AS members_covered
        FROM dim_contracts  c
        JOIN dim_policies   p  ON p.contract_id  = c.contract_id
        JOIN policy_members pm ON pm.policy_id   = p.policy_id
        GROUP BY c.contract_type
        ORDER BY c.contract_type
    """).df()
    print(r1.to_string(index=False))

    print("\n  VERIFICATION 2: Family rule — all family members share same policy?")
    r2 = con.execute("""
        -- For Family contracts, every policy should cover >= 2 members
        -- and all members of the same family should be on the same policy
        SELECT
            p.coverage_level,
            COUNT(DISTINCT p.policy_id)                    AS num_policies,
            ROUND(AVG(member_count), 1)                    AS avg_members_per_policy,
            MIN(member_count)                              AS min_members,
            MAX(member_count)                              AS max_members
        FROM dim_policies p
        JOIN (
            SELECT policy_id, COUNT(member_id) AS member_count
            FROM policy_members
            GROUP BY policy_id
        ) mc ON p.policy_id = mc.policy_id
        GROUP BY p.coverage_level
        ORDER BY p.coverage_level
    """).df()
    print(r2.to_string(index=False))

    print("\n  VERIFICATION 3: Group contract — mixed Individual/Family coverage")
    r3 = con.execute("""
        SELECT
            c.contract_type,
            p.coverage_level,
            COUNT(DISTINCT p.policy_id)   AS num_policies,
            COUNT(DISTINCT pm.member_id)  AS members_covered
        FROM dim_contracts  c
        JOIN dim_policies   p  ON p.contract_id = c.contract_id
        JOIN policy_members pm ON pm.policy_id  = p.policy_id
        WHERE c.contract_type = 'Group'
        GROUP BY c.contract_type, p.coverage_level
        ORDER BY p.coverage_level
    """).df()
    print(r3.to_string(index=False))

    print("\n  VERIFICATION 4: Sample claim with full chain of joins")
    r4 = con.execute("""
        SELECT
            c.claim_id,
            m.first_name || ' ' || m.last_name  AS member_name,
            pm.role                             AS member_role,
            p.coverage_level,
            p.plan_type,
            con.contract_type,
            pr.provider_type,
            c.claim_amount,
            c.claim_status
        FROM fact_claims     c
        JOIN dim_members     m   ON c.member_id   = m.member_id
        JOIN policy_members  pm  ON c.policy_id   = pm.policy_id
                                AND c.member_id   = pm.member_id
        JOIN dim_policies    p   ON c.policy_id   = p.policy_id
        JOIN dim_contracts   con ON p.contract_id = con.contract_id
        JOIN dim_providers   pr  ON c.provider_id = pr.provider_id
        LIMIT 8
    """).df()
    print(r4.to_string(index=False))

    con.close()

    size_mb = os.path.getsize(DB_PATH) / (1024**2)
    print(f"\n{'=' * 65}")
    print(f"✅  Database saved → '{DB_PATH}'  ({size_mb:.1f} MB)")
    print(f"{'=' * 65}")
    print("  Tables (in relationship order):")
    print("    dim_families")
    print("    dim_employers")
    print("    dim_members")
    print("    dim_providers")
    print("    dim_contracts        ← NEW: WHO is buying coverage")
    print("    dim_policies         ← WHAT plan (within a contract)")
    print("    policy_members       ← bridge: WHO is on each policy")
    print("    fact_claims")
    print("    fact_claim_diagnoses")
    print("    fact_payments")
    print(f"\n  Key rules enforced in v3:")
    print(f"    ✓  dim_contracts separates buying agreement from plan details")
    print(f"    ✓  Every member has universal member_id regardless of contract type")
    print(f"    ✓  All members of a family share exactly ONE policy")
    print(f"    ✓  Group contracts support mixed Individual/Family coverage per employee")
    print(f"    ✓  Claims always filed under a policy that genuinely covers the member")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
# %%
