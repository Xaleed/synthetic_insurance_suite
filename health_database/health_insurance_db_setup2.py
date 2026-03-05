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
