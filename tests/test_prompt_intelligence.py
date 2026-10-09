import asyncio
from aca.agents import prompt_intelligence as P
from aca.orchestrator.state import PromptEvidence, ModelEvidence, CitedSource
from aca.orchestrator.workflow import ACAState
from aca.orchestrator.agent_schemas import OrgInputs

G = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
A_GOOGLE = ("Several vendors offer multiplex panels. Sigma-Aldrich MILLIPLEX assays use Luminex xMAP technology. "
            "Thermo Fisher ProcartaPlex panels cover neurology. Bio-Rad Bio-Plex is another Luminex option.")
A_CLAUDE = ("Olink uses proximity extension assay technology for high-plex panels. Meso Scale Discovery offers U-PLEX. "
            "Thermo Fisher also sells multiplex immunoassays. Alamar Biosciences offers NULISA panels.")
A_CHATGPT = "Alamar Biosciences NULISA leads in sensitivity. Olink and Meso Scale Discovery are alternatives. Luminex is common."

def rec(pid, model, answer, score, cites=(), intent="discovery"):
    return P.AnswerRecord(pid, f"prompt {pid}", model, answer, intent, score, tuple(cites))

CITES_G = [P.CitationRecord(G+"abc", None, "sigmaaldrich.com"),
           P.CitationRecord(G+"abc", None, "sigmaaldrich.com"),      # duplicate in same answer
           P.CitationRecord(G+"def", "thermofisher.com", "Neurobiology ELISA Kits | Thermo Fisher")]
CITES_C = [P.CitationRecord("https://www.olink.com/products/olink-target-96/", "olink.com", "Olink Target 96"),
           P.CitationRecord("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7072187/", None, None)]
CITES_C2 = [P.CitationRecord("https://olink.com/products/olink-target-96", "olink.com", None)]

class FakeLLM:
    def __init__(self, outputs): self.outputs, self.calls = outputs, []
    def with_structured_output(self, schema):
        outer = self
        class S:
            async def ainvoke(self, messages):
                outer.calls.append((schema.__name__, messages)); return outer.outputs[schema.__name__]
        return S()

def llm_out(framing="not_mentioned"):
    return P._LLMUnderstanding(
        topic="Multiplex protein assay panels", intent="vendor discovery", entities=["Luminex", "NULISA"],
        concepts=[P._Concept(name="Luminex", aliases=["xMAP"]), P._Concept(name="proximity extension assay"),
                  P._Concept(name="quantum entanglement")],            # hallucinated -> must be dropped
        companies=[P._Company(name="Sigma-Aldrich", attributes=["MILLIPLEX", " "]), P._Company(name="Thermo Fisher"),
                   P._Company(name="Olink"), P._Company(name="Meso Scale Discovery", aliases=["MSD"]),
                   P._Company(name="Bio-Techne")],                      # never in text -> must be dropped
        answer_criteria=["sensitivity", "panel breadth"], client_framing=framing)

client = P.ClientInfo(name="Alamar Biosciences", domain="https://www.alamarbio.com/")
def ok(cond, message):
    assert cond, message
    return cond
results = []

async def main():
    # --- Scenario 1: client absent
    records = [rec("1379", "GoogleAI", A_GOOGLE, 40, CITES_G), rec("1379", "Claude", A_CLAUDE.replace("Alamar Biosciences offers NULISA panels.", ""), 55, CITES_C),
               rec("1378", "Claude", "Olink and Thermo Fisher dominate.", 30, CITES_C2), rec("1378", "Empty", "   ", None)]
    agent = P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out()}))
    u = await agent.analyze(records, client)
    names = {c.name: c for c in u.companies_mentioned}
    results.append(ok(u.client_presence == "absent", "client absent when never named"))
    results.append(ok("Bio-Techne" not in names and "Alamar Biosciences" not in names, "hallucinated company dropped; client not listed when unmentioned"))
    results.append(ok({c.name for c in u.concepts} == {"Luminex", "proximity extension assay"}, "hallucinated concept dropped"))
    freq = {c.name: c.frequency for c in u.concepts}
    results.append(ok(abs(freq["Luminex"] - 1/3) < 1e-9, f"concept frequency from text (Luminex 1 of 3 answers = {freq['Luminex']:.2f})"))
    results.append(ok(names["Sigma-Aldrich"].rank == 1 and names["Thermo Fisher"].rank >= 1, "rank from mention order"))
    results.append(ok(names["Sigma-Aldrich"].attributes == ["MILLIPLEX"], "blank attributes removed"))
    results.append(ok(u.baseline_scores == {"1379": 47.5, "1378": 30.0}, f"baseline scores {u.baseline_scores}"))
    cm = {c.final_url: c for c in u.citation_map}
    olink = [c for c in u.citation_map if "olink.com" in c.final_url]
    results.append(ok(len(olink) == 1 and olink[0].count == 2 and olink[0].models == ["Claude"], "same URL (www / trailing slash) merged across answers"))
    g = cm[G + "abc"]
    results.append(ok(g.count == 1 and g.domain == "sigmaaldrich.com", "duplicate within one answer counted once; redirect domain taken from title"))
    results.append(ok(cm[G + "def"].domain == "thermofisher.com", "redirect uses provided domain"))
    results.append(ok([p.prompt_id for p in u.cluster] == ["1379", "1378"] and u.cluster[0].intent == "discovery", "cluster refs keep order and intent"))
    sent = agent._model.calls[0][1][1][1]
    results.append(ok("[A1] prompt=1379 model=GoogleAI" in sent and "Empty" not in sent, "empty answers excluded from the LLM prompt"))

    # --- Scenario 2: client present but late -> low_rank ; first -> present ; negative
    late = [rec("1", "Claude", A_CLAUDE, 50), rec("1", "GoogleAI", A_CLAUDE, 50)]
    u2 = await P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out()})).analyze(late, client)
    results.append(ok(u2.client_presence == "low_rank", "client ranked last -> low_rank"))
    first = [rec("1", "ChatGPT", A_CHATGPT, 50), rec("1", "Claude", A_CHATGPT, 50)]
    u3 = await P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out()})).analyze(first, client)
    results.append(ok(u3.client_presence == "present", "client first in most answers -> present"))
    u4 = await P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out("negative")})).analyze(first, client)
    results.append(ok(u4.client_presence == "negative_framing", "negative framing honoured only when client is mentioned"))
    u5 = await P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out("negative")})).analyze(records, client)
    results.append(ok(u5.client_presence == "absent", "LLM cannot claim framing for an unmentioned client"))
    rare = [rec("1", "A", A_CHATGPT, 50), rec("1", "B", "Olink only.", 50), rec("1", "C", "Olink only.", 50)]
    u6 = await P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out()})).analyze(rare, client)
    results.append(ok(u6.client_presence == "low_rank", "client in under half of answers -> low_rank"))

    # --- domain alias stem
    results.append(ok("alamarbio" in client.all_names(), f"aliases include domain stem {client.all_names()}"))
    alias = [rec("1", "Claude", "Try alamarbio for NULISA. Olink too.", 50)]
    u7 = await P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out()})).analyze(alias, client)
    results.append(ok(u7.client_presence in ("present", "low_rank") and u7.client_presence != "absent", "client found through domain-stem alias"))

    # --- errors
    for bad, label in (([], "no records"), ([rec("1", "x", "  ", 1)], "only empty answers")):
        try: await agent.analyze(bad, client); results.append(ok(False, f"raises on {label}"))
        except ValueError: results.append(ok(True, f"raises on {label}"))

    # --- plan_clusters repair + ordering
    cr = [rec("A", "m", "x", 30), rec("B", "m", "x", 65), rec("C", "m", "x", 10), rec("D", "m", "x", 69)]
    plan = P._ClusterPlan(groups=[["B", "D", "ZZZ"], ["B"], []])     # unknown id, duplicate, empty, A and C missing
    groups = await P.PromptIntelligenceAgent(model=FakeLLM({"_ClusterPlan": plan})).plan_clusters(cr, 70)
    flat = [p for g in groups for p in g]
    results.append(ok(sorted(flat) == ["A", "B", "C", "D"] and len(flat) == 4, f"plan repaired: every id once {groups}"))
    results.append(ok(groups[0] == ["C"] and groups[1] == ["A"], "clusters ranked by score deficit"))
    single = await P.PromptIntelligenceAgent(model=FakeLLM({})).plan_clusters([rec("A", "m", "x", 1)], 70)
    results.append(ok(single == [["A"]], "single prompt skips the LLM"))

    # --- node
    pe = [PromptEvidence(prompt_id="1379", prompt_text="prompt 1379", per_model={
            "GoogleAI": ModelEvidence(response_text=A_GOOGLE, sources=[CitedSource(model="GoogleAI", url=G + "abc", domain=None, title="sigmaaldrich.com", excerpt="x")])})]
    state = ACAState(job_id="j", tenant_id=1, campaign_id=81, company_name="Alamar Biosciences", prompts=pe, org=OrgInputs(domain="alamarbio.com"))
    dataset = {"responses": [
        {"prompt_id": "1379", "prompt": "prompt 1379", "intent_category": "discovery", "model": "GoogleAI", "score": 40, "answer": A_GOOGLE, "citations": [{"url": G + "abc", "title": "sigmaaldrich.com", "domain": ""}]},
        {"prompt_id": "1379", "prompt": "prompt 1379", "intent_category": "discovery", "model": "Claude", "score": 95, "answer": A_CLAUDE, "citations": []},   # above threshold: excluded
        {"prompt_id": "9999", "prompt": "other", "intent_category": "x", "model": "Claude", "score": 10, "answer": "zzz", "citations": []}]}  # other prompt: excluded
    node = P.make_agent1_node(P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out()})), dataset_loader=lambda cid: dataset)
    out = await node(state)
    u8 = out["understanding"]
    results.append(ok([p.prompt_id for p in u8.cluster] == ["1379"] and u8.baseline_scores == {"1379": 40.0}, "node filters to cluster ids and score threshold using AImpact data"))
    results.append(ok(u8.cluster[0].intent == "discovery", "node gets intent from AImpact data"))
    def boom(cid): raise RuntimeError("db down")
    node2 = P.make_agent1_node(P.PromptIntelligenceAgent(model=FakeLLM({"_LLMUnderstanding": llm_out()})), dataset_loader=boom)
    u9 = (await node2(state))["understanding"]
    results.append(ok(u9.baseline_scores == {} and u9.cluster[0].prompt_id == "1379", "node falls back to state evidence when AImpact is unavailable"))
    try:
        await P.make_agent1_node(P.PromptIntelligenceAgent(model=FakeLLM({})))(ACAState(job_id="j", tenant_id=1, campaign_id=1, company_name="X"))
        results.append(ok(False, "node raises with no prompts"))
    except RuntimeError: results.append(ok(True, "node raises with no prompts"))

def test_prompt_intelligence():
    asyncio.run(main())
    assert all(results)