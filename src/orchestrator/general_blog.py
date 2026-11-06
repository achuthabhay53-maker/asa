"""Orchestration for the standalone General Blog workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeVar
from uuid import uuid4

from src.agents.general_blog.brief import GeneralBlogBriefAgent
from src.agents.general_blog.company_research import CompanyWebsiteResearchAgent
from src.agents.general_blog.content_strategy import GeneralBlogIntakeAgent
from src.agents.general_blog.reference_research import UserReferenceResearchAgent
from src.agents.aimpact_blog.reviewer import ReviewerAgent
from src.agents.aimpact_blog.writer import WriterAgent
from src.models.token_usage import ClaudeTokenUsage
from src.models.general_blog import (
    GeneralBlogRequest,
    GeneralBlogResult,
)
from src.observability.token_usage import (
    take_run_token_usage,
    token_usage_scope,
)
from src.utils.output_storage import save_general_blog_stage_output
from src.tools.reference_fetcher import MAX_UPLOADED_PDFS

MAX_REVIEW_CYCLES = 2
GENERAL_BLOG_WRITER_MAX_TOKENS = 24_000
logger = logging.getLogger(__name__)
_T = TypeVar("_T")
ProgressStatus = Literal["running", "done", "skipped", "error"]


@dataclass(frozen=True)
class GeneralBlogProgress:
    run_id: str
    stage: str
    label: str
    status: ProgressStatus
    message: str
    completed_stages: int
    total_stages: int
    token_usage: list[ClaudeTokenUsage]


class GeneralBlogWorkflowError(RuntimeError):
    """A stage-specific workflow failure with a support-friendly run ID."""

    def __init__(
        self,
        run_id: str,
        stage: str,
        label: str,
        cause: Exception,
        token_usage: list[ClaudeTokenUsage],
    ) -> None:
        self.run_id = run_id
        self.stage = stage
        self.label = label
        self.cause = cause
        self.token_usage = token_usage
        super().__init__(
            f"{label} failed ({type(cause).__name__}). "
            f"Run ID: {run_id}. Details: {cause}"
        )


class GeneralBlogWorkflow:
    """Run a new non-AImpact research, writing, and review path."""

    def __init__(
        self,
        *,
        intake_agent: GeneralBlogIntakeAgent | None = None,
        company_research_agent: CompanyWebsiteResearchAgent | None = None,
        reference_research_agent: UserReferenceResearchAgent | None = None,
        brief_agent: GeneralBlogBriefAgent | None = None,
        writer_agent: WriterAgent | None = None,
        reviewer_agent: ReviewerAgent | None = None,
        max_review_cycles: int = MAX_REVIEW_CYCLES,
        progress_callback: Callable[[GeneralBlogProgress], None] | None = None,
    ) -> None:
        if max_review_cycles < 1:
            raise ValueError("max_review_cycles must be at least 1.")
        self.intake_agent = intake_agent or GeneralBlogIntakeAgent()
        self.company_research_agent = company_research_agent or CompanyWebsiteResearchAgent()
        self.reference_research_agent = reference_research_agent or UserReferenceResearchAgent()
        self.brief_agent = brief_agent or GeneralBlogBriefAgent()
        self.writer_agent = writer_agent or WriterAgent(
            max_tokens=GENERAL_BLOG_WRITER_MAX_TOKENS,
            tool_choice="auto",
        )
        self.reviewer_agent = reviewer_agent or ReviewerAgent()
        self.max_review_cycles = max_review_cycles
        self.progress_callback = progress_callback
        self.run_id = ""
        self._completed_stages: set[str] = set()
        self._token_usage: list[ClaudeTokenUsage] = []

    def run(
        self,
        request: GeneralBlogRequest,
        uploaded_reference_pdfs: list[tuple[str, bytes]] | None = None,
    ) -> GeneralBlogResult:
        self.run_id = uuid4().hex
        self._completed_stages = set()
        self._token_usage = []
        uploaded_reference_pdfs = uploaded_reference_pdfs or []
        if len(uploaded_reference_pdfs) > MAX_UPLOADED_PDFS:
            raise ValueError(f"Upload no more than {MAX_UPLOADED_PDFS} PDFs per run.")
        total_stages = 7
        plan, reference_urls = self._run_stage(
            "intake",
            "Input & research plan",
            "general_blog_intake",
            total_stages,
            lambda: self.intake_agent.prepare(request),
            debug_stage="intake",
            debug_input={"request": request},
        )
        company_evidence, warnings = self._run_stage(
            "company_research",
            "Company website research",
            "general_blog_company_research",
            total_stages,
            lambda: self.company_research_agent.research_company(request, plan),
            debug_stage="company_research",
            debug_input={"request": request, "plan": plan},
        )
        if reference_urls or uploaded_reference_pdfs:
            reference_evidence, reference_warnings, uploaded_pdf_statuses = self._run_stage(
                "reference_research",
                "Reference research",
                "general_blog_reference_research",
                total_stages,
                lambda: self.reference_research_agent.research_references(
                    request,
                    reference_urls,
                    evidence_offset=len(company_evidence),
                    uploaded_pdfs=uploaded_reference_pdfs,
                ),
                debug_stage="reference_research",
                debug_input={
                    "request": request,
                    "reference_urls": reference_urls,
                    "evidence_offset": len(company_evidence),
                    "uploaded_pdfs": [
                        {"file_name": file_name, "size_bytes": len(content)}
                        for file_name, content in uploaded_reference_pdfs
                    ],
                },
            )
        else:
            reference_evidence, reference_warnings, uploaded_pdf_statuses = [], [], []
            self._save_skipped_stage(
                "reference_research",
                {
                    "request": request,
                    "reference_urls": reference_urls,
                    "uploaded_pdfs": [],
                },
                "No user-provided references or PDFs",
            )
            self._mark_skipped(
                "reference_research",
                "Reference research",
                "No user-provided references or PDFs",
                total_stages,
            )
        warnings.extend(reference_warnings)
        evidence = [*company_evidence, *reference_evidence]
        brief, writer_input = self._run_stage(
            "brief",
            "Blog brief",
            "general_blog_brief",
            total_stages,
            lambda: self.brief_agent.build(request, plan, evidence),
            debug_stage="brief",
            debug_input={
                "request": request,
                "plan": plan,
                "evidence": evidence,
            },
        )

        draft = self._run_stage(
            "writer",
            "Blog writer",
            "agent5",
            total_stages,
            lambda: self.writer_agent.write(writer_input),
            debug_stage="writer",
            debug_input={"writer_input": writer_input},
        )
        review = self._run_stage(
            "review",
            "Quality review",
            "agent6",
            total_stages,
            lambda: self.reviewer_agent.review(writer_input, draft),
            debug_stage="review_1",
            debug_input={"writer_input": writer_input, "draft": draft},
        )
        review_cycles = 1
        if not self._needs_revision(review):
            self._mark_skipped(
                "revision",
                "Review revision",
                "No revision required",
                total_stages,
            )
        while self._needs_revision(review) and review_cycles < self.max_review_cycles:
            instructions = self._revision_instructions(review)
            draft = self._run_stage(
                "revision",
                "Review revision",
                "agent5_revision",
                total_stages,
                lambda: self.writer_agent.revise(writer_input, draft, instructions),
                debug_stage=f"writer_revision_{review_cycles}",
                debug_input={
                    "writer_input": writer_input,
                    "draft": draft,
                    "instructions": instructions,
                },
            )
            review = self._run_stage(
                "review",
                "Quality review",
                "agent6",
                total_stages,
                lambda: self.reviewer_agent.review(writer_input, draft),
                debug_stage=f"review_{review_cycles + 1}",
                debug_input={"writer_input": writer_input, "draft": draft},
            )
            review_cycles += 1

        quality_passed = not self._needs_revision(review)
        if not quality_passed:
            warnings.append(
                "The draft still has blocking review findings after "
                f"{review_cycles} review cycle(s); human review is required."
            )
            logger.warning(
                "General Blog quality gate did not pass (run_id=%s, review_cycles=%d, "
                "findings=%d, unresolved=%d)",
                self.run_id,
                review_cycles,
                len(review.findings),
                len(review.unresolved_items),
            )
        if "revision" not in self._completed_stages:
            self._mark_skipped(
                "revision",
                "Review revision",
                "Revision limit reached" if not quality_passed else "No revision required",
                total_stages,
            )
        token_usage = take_run_token_usage(self.run_id, self._token_usage)
        self._notify(
            "complete",
            "Workflow complete",
            "done",
            "General Blog workflow complete",
            total_stages,
        )
        return GeneralBlogResult(
            plan=plan,
            brief=brief,
            writer_input=writer_input,
            draft=draft,
            review=review,
            sources=evidence,
            uploaded_pdf_statuses=uploaded_pdf_statuses,
            quality_passed=quality_passed,
            review_cycles=review_cycles,
            warnings=list(dict.fromkeys(warnings)),
            token_usage=token_usage,
        )

    def _run_stage(
        self,
        stage: str,
        label: str,
        agent_name: str,
        total_stages: int,
        operation: Callable[[], _T],
        *,
        debug_stage: str,
        debug_input: dict[str, object],
    ) -> _T:
        self._notify(stage, label, "running", f"{label} is running", total_stages)
        records: list[ClaudeTokenUsage] = []
        started_at = perf_counter()
        try:
            with token_usage_scope(self.run_id, agent_name) as records:
                result = operation()
        except Exception as exc:
            self._token_usage.extend(records)
            save_general_blog_stage_output(
                self.run_id,
                debug_stage,
                {
                    "run_id": self.run_id,
                    "stage": stage,
                    "agent": agent_name,
                    "status": "error",
                    "input": debug_input,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "token_usage": records,
                    "duration_seconds": round(perf_counter() - started_at, 3),
                },
            )
            logger.exception(
                "General Blog stage failed (run_id=%s, stage=%s, error_type=%s)",
                self.run_id,
                stage,
                type(exc).__name__,
            )
            self._notify(
                stage,
                label,
                "error",
                f"{type(exc).__name__}: {exc}",
                total_stages,
            )
            usage = take_run_token_usage(self.run_id, self._token_usage)
            raise GeneralBlogWorkflowError(
                self.run_id,
                stage,
                label,
                exc,
                usage,
            ) from exc
        self._token_usage.extend(records)
        save_general_blog_stage_output(
            self.run_id,
            debug_stage,
            {
                "run_id": self.run_id,
                "stage": stage,
                "agent": agent_name,
                "status": "complete",
                "input": debug_input,
                "output": result,
                "token_usage": records,
                "duration_seconds": round(perf_counter() - started_at, 3),
            },
        )
        self._mark_done(stage, label, total_stages)
        logger.info(
            "General Blog stage complete (run_id=%s, stage=%s, claude_calls=%d)",
            self.run_id,
            stage,
            len(records),
        )
        return result

    def _save_skipped_stage(
        self,
        debug_stage: str,
        debug_input: dict[str, object],
        reason: str,
    ) -> None:
        save_general_blog_stage_output(
            self.run_id,
            debug_stage,
            {
                "run_id": self.run_id,
                "stage": debug_stage,
                "status": "skipped",
                "input": debug_input,
                "output": None,
                "reason": reason,
                "token_usage": [],
            },
        )

    def _mark_done(
        self,
        stage: str,
        label: str,
        total_stages: int,
        message: str = "Complete",
    ) -> None:
        self._completed_stages.add(stage)
        self._notify(stage, label, "done", message, total_stages)

    def _mark_skipped(
        self,
        stage: str,
        label: str,
        message: str,
        total_stages: int,
    ) -> None:
        self._completed_stages.add(stage)
        self._notify(stage, label, "skipped", message, total_stages)

    def _notify(
        self,
        stage: str,
        label: str,
        status: ProgressStatus,
        message: str,
        total_stages: int,
    ) -> None:
        if self.progress_callback is not None:
            self.progress_callback(
                GeneralBlogProgress(
                    run_id=self.run_id,
                    stage=stage,
                    label=label,
                    status=status,
                    message=message,
                    completed_stages=len(self._completed_stages),
                    total_stages=total_stages,
                    token_usage=list(self._token_usage),
                )
            )

    @staticmethod
    def _needs_revision(review) -> bool:
        if review.unresolved_items:
            return True
        return any(
            finding.severity in {"critical", "high"}
            or finding.area == "factual_claims"
            for finding in review.findings
        )

    @staticmethod
    def _revision_instructions(review) -> str:
        findings = [
            f"- [{finding.severity.upper()} / {finding.area}] {finding.issue} "
            f"Recommendation: {finding.recommendation}"
            for finding in review.findings
            if finding.severity != "low" or finding.area == "factual_claims"
        ]
        unresolved = [f"- Unresolved: {item}" for item in review.unresolved_items]
        return (
            "Address the review findings below using only the supplied brief and "
            "evidence. Preserve accurate source qualifications and citations. "
            "Do not add unsupported factual claims.\n"
            + "\n".join([*findings, *unresolved])
        )
