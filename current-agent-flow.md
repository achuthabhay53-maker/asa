# Current Agent Workflow

```mermaid
flowchart LR
    U["User inputs<br/>Campaign ID · Prompt ID<br/>Customer website · optional name<br/>Source limit · target word count"]
    DB[("AImpact PostgreSQL<br/>Prompt · AI answers · citations")]

    A1["Agent 1<br/>Prompt Intelligence<br/>Interprets prompt and stored AI answers"]
    A2["Agent 2<br/>Source Intelligence<br/>Analyzes cited pages, structure, and style"]
    A3["Agent 3<br/>Ground Truth & Research<br/>Searches pages and checks evidence"]
    G{"Prompt/customer<br/>verification passes?"}
    A4["Agent 4<br/>Content Brief<br/>Creates article plan and writing guidance"]
    B{"Brief has<br/>usable sections?"}
    A5["Agent 5<br/>Blog Writer<br/>Drafts article and lists evidence references"]
    STOP["Workflow stops<br/>Verification or brief issue"]
    UI["Streamlit app<br/>Displays results and saves agent JSON outputs"]
    HUMAN["Human review<br/>Check factual claims against sources"]

    U --> A1
    U --> A3
    DB --> A1
    DB --> A2
    A1 --> A2
    A2 -. "Passed in, but currently ignored by Agent 3" .-> A3
    A1 --> A3
    A3 --> G
    G -- "No" --> STOP
    G -- "Yes" --> A4
    A1 --> A4
    A2 --> A4
    A3 --> A4
    A4 --> B
    B -- "No" --> STOP
    B -- "Yes" --> A5
    A4 --> A5
    A3 --> A5
    A5 --> UI
    UI --> HUMAN
```

## Notes

- Streamlit runs the five-agent workflow through LangGraph.
- Agents 1 and 2 independently read prompt evidence from the database. Their current score thresholds are 70 and 1000; the meaning and rationale for the difference have not been confirmed.
- Agent 2 analyzes pages cited in the stored AI answers. Agent 3 conducts separate web research and currently does not use Agent 2's output.
- Agent 2 provides source/content observations, not independent fact-checking.
- The workflow creates a draft; a person should verify claims and sources before publication.
