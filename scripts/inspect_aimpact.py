import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
conn = psycopg.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

# Check non-empty sources in campaign 81
query = """
SELECT model_name, count(*)
FROM audit_scores
WHERE campaign_id = 81
  AND sources IS NOT NULL
  AND sources::text != '[]'
  AND sources::text != 'null'
GROUP BY model_name;
"""
cur.execute(query)
print("Campaign 81 models with sources:", cur.fetchall())

# Check across the entire database what campaigns have the most sources
query2 = """
SELECT campaign_id, count(*)
FROM audit_scores
WHERE sources IS NOT NULL
  AND sources::text != '[]'
  AND sources::text != 'null'
GROUP BY campaign_id
ORDER BY count(*) DESC
LIMIT 10;
"""
cur.execute(query2)
print("Top campaigns by non-empty sources:", cur.fetchall())

# Sample non-empty sources from campaign 81
query3 = """
SELECT a.id, a.model_name, a.sources, cp.prompt_text, a.response_text, a.created_at
FROM audit_scores a
LEFT JOIN campaign_prompts cp ON cp.id = a.prompt_id
WHERE a.campaign_id = 81
  AND a.sources IS NOT NULL
  AND a.sources::text != '[]'
LIMIT 2;
"""
cur.execute(query3)
for row in cur.fetchall():
    print("--- Sample Row ---")
    print("ID:", row[0], "Model:", row[1], "Time:", row[5])
    print("Prompt:", row[3])
    print("Sources count:", len(row[2]) if isinstance(row[2], list) else "non-list")
    print("First source:", row[2][0] if isinstance(row[2], list) and row[2] else row[2])
conn.close()
