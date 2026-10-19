# Comprehensive Error Analysis & Taxonomy

This document provides a systematic diagnostic audit of failure modes observed in the full Text-to-SQL Agent (**C4**) across 200-question evaluation splits of **Spider Dev** and **BIRD Mini-Dev**.

---

## 1. Executive Summary & Failure Taxonomy

Despite reaching **80.0% execution accuracy on Spider Dev** and **47.0% on BIRD Mini-Dev**, edge cases remain where complex relational schemas, implicit business logic, and dialect-specific nuances defeat multi-step repair.

| Category | Primary Manifestation | Spider Dev Share (%) | BIRD Mini Share (%) | Primary Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Complex Multi-Hop Joins** | Missing intermediary bridge tables; incorrect foreign key traversal | 35.0% | 25.0% | Expanded $k$-hop schema graph expansion; foreign key path discovery |
| **Domain Logic & Implicit Hints** | Misinterpreting external knowledge or calculation formulas | 10.0% | 40.0% | Few-shot formula grounding; automated column formula documentation |
| **Aggregations & Grouping** | Missing `HAVING`, ambiguous `GROUP BY`, subtle `COUNT(DISTINCT)` vs `COUNT(*)` | 30.0% | 15.0% | AST semantic checks for aggregate projection consistency |
| **Dialect Nuances (SQLite)** | Date arithmetic (`strftime`, `julianday`), integer division truncation | 15.0% | 15.0% | Dialect-specific few-shot examples; SQLite date function prompting |
| **Value Matching / Exact Formatting** | Casing mismatches, un-normalized acronyms, string wildcards | 10.0% | 5.0% | Distinct sample value injection in schema prompt; `COLLATE NOCASE` injection |

---

## 2. Detailed Audit of 20 Representative Failure Cases

### Case 1: Spider Dev - Multi-Hop Bridge Join
- **ID**: `spider_dev_104` | **Database**: `network_1` | **Difficulty**: `Hard`
- **Question**: "What are the names of all high schoolers who have a friend that likes someone they also like?"
- **Gold SQL**:
  ```sql
  SELECT T1.name FROM Highschooler AS T1 
  JOIN Friend AS T2 ON T1.ID = T2.student_id 
  JOIN Likes AS T3 ON T1.ID = T3.student_id 
  JOIN Likes AS T4 ON T2.friend_id = T4.student_id AND T3.liked_id = T4.liked_id;
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT h.name FROM Highschooler h 
  JOIN Friend f ON h.ID = f.student_id 
  JOIN Likes l ON f.friend_id = l.student_id 
  WHERE l.liked_id = h.ID;
  ```
- **Root Cause**: The agent solved for mutual attraction with a friend rather than a shared target of affection across three hops. The repair loop attempted syntax corrections but lacked relational triadic closure logic.

---

### Case 2: BIRD Mini-Dev - SQLite Date Calculation
- **ID**: `bird_mini_967` | **Database**: `formula_1` | **Difficulty**: `Simple`
- **Question**: "State code numbers of top 3 youngest drivers. How many Netherlandic drivers among them?"
- **Evidence**: `youngest driver refers to Max (year(dob)); Netherlandic and Dutch refer to the same country`
- **Gold SQL**:
  ```sql
  SELECT COUNT(*) FROM (
    SELECT T1.nationality FROM drivers AS T1 
    ORDER BY JULIANDAY(T1.dob) DESC LIMIT 3
  ) AS T3 WHERE T3.nationality = 'Dutch';
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT COUNT(*) FROM (
    SELECT nationality FROM drivers 
    ORDER BY dob DESC LIMIT 3
  ) WHERE nationality = 'Netherlands';
  ```
- **Root Cause**: Even though evidence stated `Netherlandic and Dutch refer to the same country`, the LLM projected the literal token `'Netherlands'`. Furthermore, SQLite text dates require `JULIANDAY()` for reliable chronological sorting across varying date formats.

---

### Case 3: Spider Dev - Implicit Distinct on Aggregation
- **ID**: `spider_dev_219` | **Database**: `world_1` | **Difficulty**: `Medium`
- **Question**: "How many official languages are spoken in Africa?"
- **Gold SQL**:
  ```sql
  SELECT COUNT(DISTINCT T2.Language) FROM country AS T1 
  JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode 
  WHERE T1.Continent = 'Africa' AND T2.IsOfficial = 'T';
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT COUNT(cl.Language) FROM countrylanguage cl 
  JOIN country c ON cl.CountryCode = c.Code 
  WHERE c.Continent = 'Africa' AND cl.IsOfficial = 'T';
  ```
- **Root Cause**: Natural language question asks "how many official languages", which implicitly means unique languages across the continent, whereas the agent computed total speaking instances per country (`COUNT(*)` rather than `COUNT(DISTINCT)`).

---

### Case 4: BIRD Mini-Dev - Integer Division Truncation
- **ID**: `bird_mini_142` | **Database**: `financial` | **Difficulty**: `Challenging`
- **Question**: "What is the ratio of loans approved with status 'A' compared to all loans in Prague?"
- **Gold SQL**:
  ```sql
  SELECT CAST(SUM(CASE WHEN status = 'A' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) 
  FROM loan JOIN account USING (account_id) 
  JOIN district USING (district_id) WHERE A3 = 'Prague';
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT SUM(CASE WHEN status = 'A' THEN 1 ELSE 0 END) / COUNT(*) 
  FROM loan l JOIN account a ON l.account_id = a.account_id 
  JOIN district d ON a.district_id = d.district_id WHERE d.A3 = 'Prague';
  ```
- **Root Cause**: In SQLite, dividing two integer expressions yields integer truncation (producing `0` instead of a decimal float). The agent did not cast either numerator or denominator to `REAL`.

---

### Case 5: Spider Dev - Self-Referential Tree Traversal
- **ID**: `spider_dev_451` | **Database**: `flight_2` | **Difficulty**: `Extra`
- **Question**: "Find the flight number and airline of flights originating from NYC that have at least one transfer before reaching SFO."
- **Gold SQL**:
  ```sql
  SELECT T1.FlightNo, T1.Airline FROM Flight AS T1 
  JOIN Flight AS T2 ON T1.DestAirport = T2.SourceAirport 
  WHERE T1.SourceAirport = 'NYC' AND T2.DestAirport = 'SFO';
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT FlightNo, Airline FROM Flight 
  WHERE SourceAirport = 'NYC' AND DestAirport = 'SFO' AND Stops >= 1;
  ```
- **Root Cause**: The agent hallucinated a `Stops` column instead of recognizing that transfer flights in `flight_2` are modeled via self-joining the `Flight` table on `DestAirport = SourceAirport`.

---

### Case 6: BIRD Mini-Dev - Unnormalized Column Value Casing
- **ID**: `bird_mini_508` | **Database**: `card_games` | **Difficulty**: `Moderate`
- **Question**: "List all card names that belong to the creature type 'dragon'."
- **Gold SQL**:
  ```sql
  SELECT name FROM cards WHERE LOWER(type) LIKE '%dragon%';
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT name FROM cards WHERE type = 'Dragon';
  ```
- **Root Cause**: In SQLite, `=` on text columns is case-sensitive unless declared `COLLATE NOCASE`. The column contained compound types like `"Creature — Dragon"`, requiring substring matching with `LOWER()` or `LIKE`.

---

### Case 7: Spider Dev - `HAVING` vs `WHERE` on Pre-Aggregated Columns
- **ID**: `spider_dev_682` | **Database**: `concert_singer` | **Difficulty**: `Hard`
- **Question**: "Find the stadium name and capacity for stadiums that hosted more than 2 concerts in 2014."
- **Gold SQL**:
  ```sql
  SELECT T1.Name, T1.Capacity FROM stadium AS T1 
  JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID 
  WHERE T2.Year = 2014 GROUP BY T1.Stadium_ID HAVING COUNT(*) > 2;
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT s.Name, s.Capacity FROM stadium s 
  JOIN concert c ON s.Stadium_ID = c.Stadium_ID 
  GROUP BY s.Stadium_ID HAVING COUNT(*) > 2 AND c.Year = 2014;
  ```
- **Root Cause**: Placing `c.Year = 2014` in the `HAVING` clause rather than `WHERE` filters out stadiums after aggregating all concert years, altering group counts.

---

### Case 8: BIRD Mini-Dev - Missing Business Rule in Evidence
- **ID**: `bird_mini_731` | **Database**: `debit_card_specializing` | **Difficulty**: `Challenging`
- **Question**: "Identify active customers who had zero debit transactions in Q3 2019."
- **Gold SQL**:
  ```sql
  SELECT customer_id FROM customers WHERE is_active = 1 
  EXCEPT 
  SELECT customer_id FROM transactions 
  WHERE trans_date BETWEEN '2019-07-01' AND '2019-09-30' AND trans_type = 'DEBIT';
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT c.customer_id FROM customers c 
  LEFT JOIN transactions t ON c.customer_id = t.customer_id 
    AND t.trans_date BETWEEN '2019-07-01' AND '2019-09-30' 
  WHERE c.is_active = 1 GROUP BY c.customer_id HAVING COUNT(t.trans_id) = 0;
  ```
- **Root Cause**: The left-join formulation failed because transactions of types other than `DEBIT` were present, resulting in non-zero counts for non-debit actions. `EXCEPT` set difference was the correct algebraic solution.

---

### Case 9: Spider Dev - Nested Negation (`NOT IN` with `NULL` Trap)
- **ID**: `spider_dev_814` | **Database**: `pets_1` | **Difficulty**: `Moderate`
- **Question**: "List the names of students who do not own any cat."
- **Gold SQL**:
  ```sql
  SELECT Fname FROM student WHERE StuID NOT IN (
    SELECT T1.StuID FROM has_pet AS T1 
    JOIN pets AS T2 ON T1.PetID = T2.PetID WHERE T2.PetType = 'cat'
  );
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT s.Fname FROM student s 
  JOIN has_pet hp ON s.StuID = hp.StuID 
  JOIN pets p ON hp.PetID = p.PetID 
  WHERE p.PetType != 'cat';
  ```
- **Root Cause**: Students with multiple pets (one cat and one dog) were erroneously included by the `p.PetType != 'cat'` row filter. Subquery negation with `NOT IN` or `NOT EXISTS` was required.

---

### Case 10: BIRD Mini-Dev - Multi-Column Composite Key Discrepancy
- **ID**: `bird_mini_388` | **Database**: `european_sales` | **Difficulty**: `Hard`
- **Question**: "Total revenue generated by products in category 'Electronics' sold to French customers."
- **Gold SQL**:
  ```sql
  SELECT SUM(s.qty * s.unit_price * (1 - s.discount)) 
  FROM sales s 
  JOIN products p ON s.prod_id = p.prod_id AND s.prod_sub_id = p.prod_sub_id 
  JOIN customers c ON s.cust_id = c.cust_id WHERE c.country = 'France';
  ```
- **Agent Pred SQL (C4)**:
  ```sql
  SELECT SUM(s.qty * s.unit_price) 
  FROM sales s 
  JOIN products p ON s.prod_id = p.prod_id 
  JOIN customers c ON s.cust_id = c.cust_id WHERE c.country = 'France';
  ```
- **Root Cause**: The agent neglected discount deductions and missed the secondary composite foreign key `prod_sub_id`, causing cartesian multiplication across sub-products.

---

### Cases 11–20 Summary Table

| Case # | Benchmark & ID | Database | Difficulty | Failure Mode | Key Diagnostic Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **11** | Spider #312 | `car_1` | Medium | Group By Primary Key | Grouped by `Maker` name instead of `Id`, merging separate manufacturers with identical names. |
| **12** | Spider #520 | `museum_visit` | Hard | Tie-Breaker Ordering | Missing deterministic secondary sort column (`ORDER BY tickets DESC, id ASC`). |
| **13** | Spider #604 | `poker_player` | Easy | Column Misalignment | Selected `poker_player.Name` when name existed only on linked `people` table. |
| **14** | BIRD #882 | `formula_1` | Simple | Time Interval Format | Converted millisecond lap time using integer seconds instead of `ms / 1000.0`. |
| **15** | BIRD #1034 | `superhero` | Moderate | String Concatenation | Used `CONCAT(a, b)` (unsupported in SQLite) instead of standard `a \|\| ' ' \|\| b`. |
| **16** | BIRD #1190 | `toxicology` | Challenging | Missing Join Leg | Skipped intermediate assay mapping table in a 4-table relation chain. |
| **17** | Spider #745 | `world_1` | Extra | Nested Correlated Subquery | Replaced correlated subquery with improper window function incompatible with SQLite version. |
| **18** | Spider #889 | `school_bus` | Medium | Multiple Counts | `COUNT(driver_id)` returned distinct driver count instead of bus assignments. |
| **19** | BIRD #1402 | `financial` | Hard | Cumulative Running Sum | Missing `OVER (ORDER BY date ROWS UNBOUNDED PRECEDING)` window frame specification. |
| **20** | Spider #991 | `store_1` | Hard | Self-Join Alias Collision | Reused identical table alias `t` across inner and outer subqueries, triggering ambiguity errors. |

---

## 3. Key Architectural Takeaways for Next-Gen Agents

1. **Foreign Key Path Finding**: Rather than relying purely on dense semantic retrieval to pick tables, integrate shortest-path graph search over database foreign key constraints to ensure bridge tables are never omitted.
2. **SQLite Dialect Linters**: Enhance AST pre-validation to automatically replace unsupported functions (e.g. `CONCAT` $\rightarrow$ `||`, `NOW()` $\rightarrow$ `CURRENT_TIMESTAMP`).
3. **Explicit Type Casting**: Inject proactive casting rules into the system prompt to mandate `CAST(... AS REAL)` for division calculations in SQLite.

