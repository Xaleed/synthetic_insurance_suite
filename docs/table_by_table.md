Imagine the Smith family — 2 parents and 2 kids:
REAL WORLD SITUATION:
    John Smith (father)   goes to the insurance company
    and buys a Family contract to cover his whole household.

How this flows through every table in the database:
STEP 1 — dim_families
         The household itself is registered:

         family_id      family_size    state    income_category
         FAM000042      4              CA       Middle

         This record says:
         "There is a household of 4 people in California"
         It knows NOTHING about who those people are yet.
         It is just a container/label for the household.
STEP 2 — dim_contracts
         The BUYING AGREEMENT is recorded:
         John walked in and signed a contract.

         contract_id    contract_type    family_id
         CON0000087     Family           FAM000042  ← points back to the household

         This record says:
         "Household FAM000042 has signed a Family insurance contract with us"
         ONE contract for the WHOLE household — not one per person.
STEP 3 — dim_policies
         The SPECIFIC PLAN details are recorded:
         John chose an HMO plan with a $1,000 deductible.

         policy_id      contract_id     coverage_level   plan_type   deductible
         POL0000234     CON0000087      Family           HMO         $1,000

         This record says:
         "Under contract CON0000087, the chosen plan is HMO with these details"
         ONE policy for the WHOLE family — every member shares these same rules.
         If one family member hits the deductible, it counts toward the FAMILY deductible.
STEP 4 — policy_members  (the bridge table)
         NOW we link the actual people to the policy:

         bridge_id      policy_id      member_id      role
         BRG000000001   POL0000234     MEM0000101     Policyholder  ← John (the buyer)
         BRG000000002   POL0000234     MEM0000102     Dependent     ← Mary (wife)
         BRG000000003   POL0000234     MEM0000103     Dependent     ← Tommy (child)
         BRG000000004   POL0000234     MEM0000104     Dependent     ← Lisa (child)

         This table says:
         "These 4 specific people are all covered under policy POL0000234"
         Notice: ALL 4 share the EXACT SAME policy_id — this is the v3 rule.
         No family member can be on a different policy.
STEP 5 — dim_members
         The actual personal details of each person:

         member_id      first_name   last_name   date_of_birth   gender
         MEM0000101     John         Smith       1975-03-14      Male
         MEM0000102     Mary         Smith       1978-09-22      Female
         MEM0000103     Tommy        Smith       2008-06-01      Male
         MEM0000104     Lisa         Smith       2010-11-30      Female

         This table says:
         "Here are the personal details of every insured individual"
         dim_members knows NOTHING about contracts or policies.
         It just stores WHO each person is.

So reading the chain top to bottom:
dim_families   →  "There is a household"
     ↓
dim_contracts  →  "That household signed a Family contract"
     ↓
dim_policies   →  "Under that contract, they chose this specific plan"
     ↓
policy_members →  "These specific people are covered under that plan"
     ↓
dim_members    →  "Here are the personal details of each of those people"

And reading bottom to top (how a SQL query works):
sql-- "Show me all claims for the Smith family"

SELECT
    m.first_name,
    m.last_name,
    c.claim_amount,
    c.claim_date
FROM dim_members m
JOIN policy_members  pm  ON m.member_id   = pm.member_id
JOIN dim_policies    p   ON pm.policy_id  = p.policy_id
JOIN dim_contracts   con ON p.contract_id = con.contract_id
JOIN dim_families    f   ON con.family_id = f.family_id
JOIN fact_claims     c   ON c.member_id   = m.member_id
WHERE f.family_id = 'FAM000042'

-- Result: claims for ALL 4 Smith family members
-- because they all trace back to the same FAM000042
```

---

**The key insight:**
```
dim_families does not STORE family members.
dim_families just IDENTIFIES that a household exists.

The actual people live in dim_members.
The CONNECTION between the household and its people
passes through THREE tables:
    dim_contracts → dim_policies → policy_members → dim_members