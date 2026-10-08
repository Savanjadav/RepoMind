"use client";

import { useEffect, useRef, useState } from "react";
import { startIndexing, type IndexingSummary, type Repository } from "./actions";
import RepositoryQA from "./repository-qa";

type Operation = {
  sending: boolean;
  message: string;
  // undefined = unknown baseline; null = successfully observed no previous job.
  awaiting?: { jobId?: string; previousJobId: string | null | undefined };
};
type View = {
  summaries: Record<string, IndexingSummary>;
  operations: Record<string, Operation>;
  readinessRefresh: Record<string, true>;
  warning: string;
  reading: boolean;
};

export default function RepositoryIndexing({ repositories, initial }: {
  repositories: Repository[];
  initial: IndexingSummary[] | null;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<View>(() => ({
    summaries: Object.fromEntries((initial ?? []).map((item) => [item.repository_id, item])),
    operations: {}, readinessRefresh: {}, warning: initial === null ? "Status unavailable. Refresh status to try again." : "", reading: false,
  }));
  const controls = useRef<{ refresh: () => void; start: (id: string, form: FormData) => void } | null>(null);

  useEffect(() => {
    // One coordinator per mounted list, not one timer per repository. The page
    // keys this component by its server snapshot so new props get fresh state.
    let state: View = {
      summaries: Object.fromEntries((initial ?? []).map((item) => [item.repository_id, item])),
      operations: {}, readinessRefresh: {}, warning: initial === null ? "Status unavailable. Refresh status to try again." : "", reading: false,
    };
    let alive = true;
    let epoch = 0;
    let failures = 0;
    let paused = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    const queued = new Set<string>();
    const publish = () => { if (alive) setView({ ...state }); };
    const cancelTimer = () => { if (timer !== undefined) clearTimeout(timer); timer = undefined; };
    const activeIds = () => repositories.map((repo) => repo.id).filter((id) => {
      const operation = state.operations[id];
      if (operation?.sending) return false;
      const status = state.summaries[id]?.latest_job?.status;
      return status === "pending" || status === "running" || Boolean(operation?.awaiting?.jobId);
    });
    const schedule = () => {
      cancelTimer();
      if (!alive || document.hidden || paused || state.reading) return;
      if (queued.size) { void read(); return; }
      if (activeIds().length) timer = setTimeout(() => { void read(); }, [3000, 6000, 12000][failures] ?? 12000);
    };
    async function read() {
      if (!alive || document.hidden || state.reading) return;
      cancelTimer();
      const ids = [...new Set([...queued, ...activeIds()])].filter((id) => !state.operations[id]?.sending);
      queued.clear();
      if (!ids.length) return;
      const token = epoch;
      const abort = new AbortController();
      controller = abort;
      state = { ...state, reading: true };
      publish();
      try {
        const query = new URLSearchParams();
        ids.forEach((id) => query.append("repository_id", id));
        const response = await fetch(`/api/indexing-summary?${query}`, {
          cache: "no-store", redirect: "error", signal: AbortSignal.any([abort.signal, AbortSignal.timeout(7000)]),
        });
        if (!alive || token !== epoch) return;
        if (!response.ok) {
          if (response.status === 404) {
            paused = true;
            state = { ...state, warning: "Repository list changed. Reload the page to refresh it." };
          }
          throw new Error("Status unavailable");
        }
        // This fixed same-origin handler already validates and projects the DTO.
        const data: { items: IndexingSummary[] } = await response.json();
        if (!alive || token !== epoch) return;
        if (!Array.isArray(data.items)) throw new Error("Status unavailable");
        const summaries = { ...state.summaries };
        const operations = { ...state.operations };
        const readinessRefresh = { ...state.readinessRefresh };
        for (const item of data.items) {
          // Conflict freshness is restored by observation, not a new job ID.
          delete readinessRefresh[item.repository_id];
          const operation = operations[item.repository_id];
          const waiting = operation?.awaiting;
          if (waiting) {
            // An unchanged old terminal job (or no job) cannot disprove a POST
            // that may still finish. Never enable an automatic mutation retry.
            const id = item.latest_job?.job_id;
            if (!id || (waiting.jobId ? id !== waiting.jobId :
              waiting.previousJobId === undefined || id === waiting.previousJobId)) continue;
            operations[item.repository_id] = { sending: false, message: "" };
          }
          summaries[item.repository_id] = item;
        }
        failures = 0;
        paused = false;
        state = { ...state, summaries, operations, readinessRefresh, warning: "" };
      } catch {
        if (!alive || token !== epoch || abort.signal.aborted) return;
        failures++;
        paused = paused || failures >= 3;
        state = { ...state, warning: state.warning.startsWith("Repository list changed") ? state.warning :
          `Could not refresh indexing status. Showing the last known state.${paused ? " Automatic refresh paused. Use Refresh status." : ""}` };
      } finally {
        if (alive) {
          state = { ...state, reading: false };
          publish();
          schedule();
        }
      }
    }
    const refresh = (ids: string[]) => {
      failures = 0;
      paused = false;
      ids.forEach((id) => queued.add(id));
      schedule();
    };
    const invalidateRead = () => { epoch++; cancelTimer(); controller?.abort(); };
    async function start(id: string, form: FormData) {
      if (state.operations[id]?.sending || state.operations[id]?.awaiting) return;
      const baseline = state.summaries[id];
      const current = baseline?.latest_job;
      if (current?.status === "pending" || current?.status === "running") return;
      invalidateRead();
      state = { ...state, operations: { ...state.operations, [id]: { sending: true, message: "Requesting indexing…" } } };
      publish();
      let result: Awaited<ReturnType<typeof startIndexing>>;
      try { result = await startIndexing(form); }
      catch { result = { status: "uncertain", message: "The indexing request outcome is uncertain. Refresh status before another start." }; }
      if (!alive) return;
      invalidateRead();
      const awaiting = result.status === "accepted" || result.status === "uncertain" ?
        { jobId: result.jobId, previousJobId: baseline ? current?.job_id ?? null : undefined } : undefined;
      if (result.status === "uncertain" && !baseline) {
        result = { ...result, message: "The indexing request outcome is uncertain. Status could not be matched to this request." };
      }
      state = { ...state, operations: { ...state.operations, [id]: { sending: false, message: result.message, awaiting } },
        readinessRefresh: result.status === "conflict" ? { ...state.readinessRefresh, [id]: true } : state.readinessRefresh };
      publish();
      if (awaiting || result.status === "conflict") refresh([id]);
      else schedule();
    }
    controls.current = { refresh: () => refresh(repositories.map((repo) => repo.id)), start: (id, form) => { void start(id, form); } };
    const visibility = () => {
      if (document.hidden) invalidateRead();
      else { activeIds().forEach((id) => queued.add(id)); schedule(); }
    };
    document.addEventListener("visibilitychange", visibility);
    schedule();
    return () => {
      alive = false;
      invalidateRead();
      controls.current = null;
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [initial, repositories]);

  const readiness = (id: string) => {
    const item = view.summaries[id];
    const operation = view.operations[id];
    if (view.warning || !item || operation?.awaiting || view.readinessRefresh[id]) return "Refresh indexing status before asking.";
    if (operation?.sending || item.latest_job?.status === "pending" || item.latest_job?.status === "running") return "Wait for indexing to finish before asking questions.";
    if (!item.latest_job) return "Index this repository before asking questions.";
    if (item.latest_job.status === "failed") return "The latest indexing run failed. Complete indexing before asking here.";
    return item.snapshot_counts && item.snapshot_counts.code_units > 0 ? "" : "No searchable code units are available.";
  };
  const selected = repositories.find((repo) => repo.id === selectedId);
  return <>
    <button className="retry" type="button" disabled={view.reading} onClick={() => controls.current?.refresh()}>Refresh status</button>
    <p role="status" aria-live="polite">{view.warning}</p>
    <p className="detail">Indexing runs in this backend process. An interruption may leave Pending or Indexing unfinished; there is no automatic recovery. A previous completed snapshot may still be available during or after a failed reindex.</p>
    <ul className="repositories">
      {repositories.map((repo) => {
        const item = view.summaries[repo.id];
        const operation = view.operations[repo.id];
        const status = item?.latest_job?.status;
        const waiting = Boolean(operation?.awaiting);
        const label = operation?.sending ? "Requesting indexing…" : operation?.awaiting ? (operation.awaiting.jobId ? "Pending" : "Request outcome uncertain") :
          !item ? "Status unavailable" : !item.latest_job ? "Not indexed" :
          status === "pending" ? "Pending" : status === "running" ? "Indexing…" :
          status === "failed" ? "Indexing failed" :
          item.snapshot_counts && item.snapshot_counts.code_units > 0 ? "Ready" : "Completed — no searchable code units found";
        return <li key={repo.id}>
          <h3>{repo.name}</h3>
          <p className="repository-source">{repo.source}</p>
          <p className="detail">Registered <time dateTime={repo.created_at}>{new Date(repo.created_at).toISOString().slice(0, 10)} UTC</time></p>
          <p className="indexing-state" aria-live="polite">{label}</p>
          {!waiting && item?.latest_job === null && <p className="detail">No indexing run recorded.</p>}
          {!waiting && status === "failed" && <p>Indexing failed. No changes from this run were committed.</p>}
          {!waiting && !operation?.sending && item?.snapshot_counts && <p>Indexed snapshot — Files: {item.snapshot_counts.files} · Code units: {item.snapshot_counts.code_units}</p>}
          <form onSubmit={(event) => {
            event.preventDefault();
            controls.current?.start(repo.id, new FormData(event.currentTarget));
          }}>
            <input type="hidden" name="repository_id" value={repo.id} />
            <button className="submit" type="submit" disabled={operation?.sending || waiting || status === "pending" || status === "running"}>
              {operation?.sending ? "Requesting indexing…" : status === "completed" ? "Reindex repository" : status === "failed" ? "Try indexing again" : "Index repository"}
            </button>
          </form>
          <p role="status" aria-live="polite">{operation?.message ?? ""}</p>
          <button className="retry" type="button" disabled={Boolean(readiness(repo.id))} onClick={() => setSelectedId(repo.id)}>Ask about this repository</button>
          {readiness(repo.id) && <p className="detail">{readiness(repo.id)}</p>}
        </li>;
      })}
    </ul>
    {selected && <RepositoryQA key={`${selected.id}:${readiness(selected.id)}`} repositoryId={selected.id} name={selected.name} disabledReason={readiness(selected.id)} />}
  </>;
}
