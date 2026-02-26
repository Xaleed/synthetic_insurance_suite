"""
Health Insurance DuckDB — Practice Queries
===========================================
A collection of SQL and Python practice examples
organized by difficulty and topic.

Run after: health_insurance_db_normalized.py

Connect:
    import duckdb
    con = duckdb.connect("health_insurance_normalized.duckdb")
"""

import duckdb
import pandas as pd

con = duckdb.connect("health_insurance_normalized.duckdb")

# ═══════════════════════════════════════════════════════
# LEVEL 1 — BEGINNER  (Basic SELECT, Filter, Count)
# ═══════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LEVEL 1 — BEGINNER")
print("="*60)

# ── Q1. How many total claims are in the database?
q1 = con.execute("""
    SELECT COUNT(*) AS total_claims
    FROM fact_claims
""").df()
print("\nQ1. Total number of claims:")
print(q1.to_string(index=False))

# ── Q2. How many claims per status?
q2 = con.execute("""
    SELECT
        claim_status,
        COUNT(*) AS total
    FROM fact_claims
    GROUP BY claim_status
    ORDER BY total DESC
""").df()
print("\nQ2. Claims by status:")
print(q2.to_string(index=False))

# ── Q3. How many members per gender?
q3 = con.execute("""
    SELECT
        gender,
        COUNT(*) AS total_members
    FROM dim_members
    GROUP BY gender
    ORDER BY total_members DESC
""").df()
print("\nQ3. Members by gender:")
print(q3.to_string(index=False))

# ── Q4. How many providers per type?
q4 = con.execute("""
    SELECT
        provider_type,
        COUNT(*) AS total
    FROM dim_providers
    GROUP BY provider_type
    ORDER BY total DESC
""").df()
print("\nQ4. Providers by type:")
print(q4.to_string(index=False))

# ── Q5. What is the total and average claim amount?
q5 = con.execute("""
    SELECT
        ROUND(SUM(claim_amount), 2)  AS total_claimed,
        ROUND(AVG(claim_amount), 2)  AS avg_claim,
        ROUND(MIN(claim_amount), 2)  AS min_claim,
        ROUND(MAX(claim_amount), 2)  AS max_claim
    FROM fact_claims
""").df()
print("\nQ5. Claim amount summary:")
print(q5.to_string(index=False))


# ═══════════════════════════════════════════════════════
# LEVEL 2 — INTERMEDIATE  (JOINs, GROUP BY, Date filters)
# ═══════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LEVEL 2 — INTERMEDIATE")
print("="*60)

# ── Q6. Total paid amount per provider type (requires JOIN)
q6 = con.execute("""
    SELECT
        p.provider_type,
        COUNT(c.claim_id)               AS total_claims,
        ROUND(SUM(c.paid_amount), 2)    AS total_paid,
        ROUND(AVG(c.paid_amount), 2)    AS avg_paid
    FROM fact_claims   c
    JOIN dim_providers p ON c.provider_id = p.provider_id
    GROUP BY p.provider_type
    ORDER BY total_paid DESC
""").df()
print("\nQ6. Paid amount by provider type:")
print(q6.to_string(index=False))

# ── Q7. Claims per year — trend over time
q7 = con.execute("""
    SELECT
        YEAR(claim_date)    AS claim_year,
        COUNT(*)            AS total_claims,
        ROUND(SUM(claim_amount), 2) AS total_amount
    FROM fact_claims
    GROUP BY claim_year
    ORDER BY claim_year
""").df()
print("\nQ7. Claims trend by year:")
print(q7.to_string(index=False))

# ── Q8. Top 10 most expensive claims with member and provider info
q8 = con.execute("""
    SELECT
        c.claim_id,
        c.claim_date,
        c.claim_amount,
        c.claim_type,
        m.first_name || ' ' || m.last_name  AS member_name,
        m.gender,
        p.provider_name,
        p.provider_type
    FROM fact_claims   c
    JOIN dim_members   m ON c.member_id   = m.member_id
    JOIN dim_providers p ON c.provider_id = p.provider_id
    ORDER BY c.claim_amount DESC
    LIMIT 10
""").df()
print("\nQ8. Top 10 most expensive claims:")
print(q8.to_string(index=False))

# ── Q9. Fraud rate per claim type
q9 = con.execute("""
    SELECT
        claim_type,
        COUNT(*)                                        AS total_claims,
        SUM(CAST(is_fraud_flagged AS INTEGER))          AS fraud_count,
        ROUND(
            SUM(CAST(is_fraud_flagged AS INTEGER)) * 100.0 / COUNT(*), 2
        )                                               AS fraud_rate_pct
    FROM fact_claims
    GROUP BY claim_type
    ORDER BY fraud_rate_pct DESC
""").df()
print("\nQ9. Fraud rate by claim type:")
print(q9.to_string(index=False))

# ── Q10. Average length of stay for inpatient claims by provider type
q10 = con.execute("""
    SELECT
        p.provider_type,
        COUNT(*)                          AS inpatient_claims,
        ROUND(AVG(c.length_of_stay), 1)  AS avg_los_days,
        MAX(c.length_of_stay)            AS max_los_days
    FROM fact_claims   c
    JOIN dim_providers p ON c.provider_id = p.provider_id
    WHERE c.claim_type = 'Inpatient'
      AND c.length_of_stay > 0
    GROUP BY p.provider_type
    ORDER BY avg_los_days DESC
""").df()
print("\nQ10. Average length of stay by provider type:")
print(q10.to_string(index=False))


# ═══════════════════════════════════════════════════════
# LEVEL 3 — ADVANCED  (Window functions, CTEs, Subqueries)
# ═══════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LEVEL 3 — ADVANCED")
print("="*60)

# ── Q11. Running total of paid amounts over time (Window Function)
q11 = con.execute("""
    SELECT
        claim_date,
        paid_amount,
        ROUND(
            SUM(paid_amount) OVER (ORDER BY claim_date), 2
        ) AS running_total_paid
    FROM fact_claims
    WHERE claim_status = 'Approved'
    ORDER BY claim_date
    LIMIT 15
""").df()
print("\nQ11. Running total of paid amounts (first 15 rows):")
print(q11.to_string(index=False))

# ── Q12. Rank members by total claim amount (Window Function)
q12 = con.execute("""
    SELECT
        m.member_id,
        m.first_name || ' ' || m.last_name  AS member_name,
        m.gender,
        COUNT(c.claim_id)                   AS total_claims,
        ROUND(SUM(c.claim_amount), 2)       AS total_claimed,
        RANK() OVER (
            ORDER BY SUM(c.claim_amount) DESC
        )                                   AS cost_rank
    FROM fact_claims c
    JOIN dim_members m ON c.member_id = m.member_id
    GROUP BY m.member_id, m.first_name, m.last_name, m.gender
    ORDER BY cost_rank
    LIMIT 10
""").df()
print("\nQ12. Top 10 highest-cost members (ranked):")
print(q12.to_string(index=False))

# ── Q13. CTE — Members with claims ABOVE average claim amount
q13 = con.execute("""
    WITH avg_claim AS (
        SELECT AVG(claim_amount) AS avg_amount
        FROM fact_claims
    ),
    member_totals AS (
        SELECT
            member_id,
            ROUND(AVG(claim_amount), 2) AS member_avg_claim
        FROM fact_claims
        GROUP BY member_id
    )
    SELECT
        m.member_id,
        m.first_name || ' ' || m.last_name  AS member_name,
        mt.member_avg_claim,
        ROUND(a.avg_amount, 2)              AS overall_avg,
        ROUND(mt.member_avg_claim - a.avg_amount, 2) AS diff_from_avg
    FROM member_totals mt
    JOIN dim_members   m ON mt.member_id = m.member_id
    CROSS JOIN avg_claim a
    WHERE mt.member_avg_claim > a.avg_amount
    ORDER BY diff_from_avg DESC
    LIMIT 10
""").df()
print("\nQ13. Members with above-average claim costs:")
print(q13.to_string(index=False))

# ── Q14. Most common diagnosis per claim type
q14 = con.execute("""
    WITH ranked_diags AS (
        SELECT
            c.claim_type,
            d.diagnosis_code,
            COUNT(*) AS freq,
            RANK() OVER (
                PARTITION BY c.claim_type
                ORDER BY COUNT(*) DESC
            ) AS rnk
        FROM fact_claims           c
        JOIN fact_claim_diagnoses  d ON c.claim_id = d.claim_id
        GROUP BY c.claim_type, d.diagnosis_code
    )
    SELECT claim_type, diagnosis_code, freq
    FROM ranked_diags
    WHERE rnk = 1
    ORDER BY freq DESC
""").df()
print("\nQ14. Most common diagnosis per claim type:")
print(q14.to_string(index=False))

# ── Q15. Month-over-month claim growth rate
q15 = con.execute("""
    WITH monthly AS (
        SELECT
            YEAR(claim_date)  AS yr,
            MONTH(claim_date) AS mo,
            COUNT(*)          AS total_claims
        FROM fact_claims
        GROUP BY yr, mo
    )
    SELECT
        yr,
        mo,
        total_claims,
        LAG(total_claims) OVER (ORDER BY yr, mo)  AS prev_month,
        ROUND(
            (total_claims - LAG(total_claims) OVER (ORDER BY yr, mo))
            * 100.0
            / NULLIF(LAG(total_claims) OVER (ORDER BY yr, mo), 0),
        2) AS growth_pct
    FROM monthly
    ORDER BY yr, mo
    LIMIT 18
""").df()
print("\nQ15. Month-over-month claim growth rate:")
print(q15.to_string(index=False))


# ═══════════════════════════════════════════════════════
# LEVEL 4 — ANALYTICS  (Risk, Fraud, Cohort Analysis)
# ═══════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LEVEL 4 — ANALYTICS")
print("="*60)

# ── Q16. High-risk members — chronic condition + high claim frequency
q16 = con.execute("""
    SELECT
        m.member_id,
        m.first_name || ' ' || m.last_name  AS member_name,
        m.health_risk_score,
        m.chronic_condition_flag,
        COUNT(c.claim_id)                   AS claim_count,
        ROUND(SUM(c.claim_amount), 2)       AS total_claimed
    FROM dim_members m
    JOIN fact_claims c ON m.member_id = c.member_id
    WHERE m.chronic_condition_flag = TRUE
    GROUP BY
        m.member_id, m.first_name, m.last_name,
        m.health_risk_score, m.chronic_condition_flag
    HAVING COUNT(c.claim_id) >= 10
    ORDER BY total_claimed DESC
    LIMIT 10
""").df()
print("\nQ16. High-risk chronic members with 10+ claims:")
print(q16.to_string(index=False))

# ── Q17. Fraud detection — flagged claims by provider
q17 = con.execute("""
    SELECT
        p.provider_id,
        p.provider_name,
        p.provider_type,
        COUNT(c.claim_id)                              AS total_claims,
        SUM(CAST(c.is_fraud_flagged AS INTEGER))       AS fraud_flags,
        ROUND(
            SUM(CAST(c.is_fraud_flagged AS INTEGER)) * 100.0
            / COUNT(c.claim_id), 2
        )                                              AS fraud_rate_pct,
        ROUND(SUM(c.claim_amount), 2)                  AS total_billed
    FROM fact_claims   c
    JOIN dim_providers p ON c.provider_id = p.provider_id
    GROUP BY p.provider_id, p.provider_name, p.provider_type
    HAVING fraud_flags > 0
    ORDER BY fraud_rate_pct DESC
    LIMIT 10
""").df()
print("\nQ17. Providers with highest fraud rates:")
print(q17.to_string(index=False))

# ── Q18. Claim approval rate by insurance category
q18 = con.execute("""
    SELECT
        pol.insurance_category,
        COUNT(c.claim_id)                              AS total_claims,
        SUM(CASE WHEN c.claim_status = 'Approved'
                 THEN 1 ELSE 0 END)                    AS approved,
        ROUND(
            SUM(CASE WHEN c.claim_status = 'Approved'
                     THEN 1 ELSE 0 END) * 100.0
            / COUNT(c.claim_id), 2
        )                                              AS approval_rate_pct
    FROM fact_claims  c
    JOIN dim_policies pol ON c.policy_id = pol.policy_id
    GROUP BY pol.insurance_category
    ORDER BY approval_rate_pct DESC
""").df()
print("\nQ18. Approval rate by insurance category:")
print(q18.to_string(index=False))

# ── Q19. Age group analysis — claim cost by age band
q19 = con.execute("""
    SELECT
        CASE
            WHEN DATE_DIFF('year', m.date_of_birth, CURRENT_DATE) < 18  THEN 'Under 18'
            WHEN DATE_DIFF('year', m.date_of_birth, CURRENT_DATE) < 35  THEN '18-34'
            WHEN DATE_DIFF('year', m.date_of_birth, CURRENT_DATE) < 50  THEN '35-49'
            WHEN DATE_DIFF('year', m.date_of_birth, CURRENT_DATE) < 65  THEN '50-64'
            ELSE '65+'
        END                                AS age_group,
        COUNT(c.claim_id)                  AS total_claims,
        ROUND(AVG(c.claim_amount), 2)      AS avg_claim,
        ROUND(SUM(c.claim_amount), 2)      AS total_claimed
    FROM fact_claims c
    JOIN dim_members m ON c.member_id = m.member_id
    GROUP BY age_group
    ORDER BY total_claimed DESC
""").df()
print("\nQ19. Claim cost by age group:")
print(q19.to_string(index=False))

# ── Q20. Payment gap analysis — days between claim and payment
q20 = con.execute("""
    SELECT
        p.payment_method,
        COUNT(*)                                           AS total_payments,
        ROUND(AVG(
            DATE_DIFF('day', c.claim_date, p.payment_date)
        ), 1)                                              AS avg_days_to_pay,
        MIN(DATE_DIFF('day', c.claim_date, p.payment_date)) AS min_days,
        MAX(DATE_DIFF('day', c.claim_date, p.payment_date)) AS max_days
    FROM fact_payments p
    JOIN fact_claims   c ON p.claim_id = c.claim_id
    GROUP BY p.payment_method
    ORDER BY avg_days_to_pay
""").df()
print("\nQ20. Average days to payment by method:")
print(q20.to_string(index=False))

con.close()
print("\n" + "="*60)
print("  All queries completed!")
print("="*60)