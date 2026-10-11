"use client";

import { useEffect, useState } from "react";
import type { RepositoryOverview as Overview } from "@/lib/api";

export default function RepositoryOverview({ repositoryId, name, revision }: {
  repositoryId: string; name: string; revision: string;
}) {
  const [refresh, setRefresh] = useState(0);
  const [state, setState] = useState<{ identity: string; data?: Overview; error?: boolean } | null>(null);
  const identity = `${repositoryId}:${revision}:${refresh}`;
  useEffect(() => {
    let alive = true;
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(`/api/repository-overview?repository_id=${encodeURIComponent(repositoryId)}`, {
          cache: "no-store", redirect: "error", signal: AbortSignal.any([controller.signal, AbortSignal.timeout(7000)]),
        });
        if (!response.ok) throw new Error("Overview unavailable");
        const data: Overview = await response.json();
        if (data.repository_id !== repositoryId) throw new Error("Unexpected repository");
        if (alive) setState({ identity, data });
      } catch {
        if (alive) setState({ identity, error: true });
      }
    }
    void load();
    return () => { alive = false; controller.abort(); };
  }, [repositoryId, identity]);
  const current = state?.identity === identity ? state : null;
  const data = current?.data;
  return <section className="overview" aria-labelledby="overview-heading">
    <h3 id="overview-heading">Repository overview — {name}</h3>
    <button type="button" className="retry" onClick={() => setRefresh(value => value + 1)}>Refresh overview</button>
    <p role="status">{!current ? "Loading indexed repository overview…" : current.error ? "Overview unavailable. Refresh to try again." : "Indexed overview loaded."}</p>
    {data && <>
      <p>Latest indexing attempt: {data.latest_job?.status ?? "No indexing run recorded"}.</p>
      {(data.latest_job?.status === "failed" || data.latest_job?.status === "running" || data.latest_job?.status === "pending") &&
        <p>These figures describe currently persisted indexed data. An earlier successful snapshot may remain available; the latest attempt does not determine these counts.</p>}
      <dl className="overview-counts"><div><dt>Indexed files</dt><dd>{data.snapshot_counts.files}</dd></div>
        <div><dt>Indexed code units</dt><dd>{data.snapshot_counts.code_units}</dd></div>
        <div><dt>Named code units</dt><dd>{data.snapshot_counts.named_code_units}</dd></div></dl>
      {data.snapshot_counts.code_units === 0 ? <p>No indexed code units are currently available.{data.snapshot_counts.files > 0 ? " Files can be indexed without producing code units." : ""}</p> : <div className="overview-tables">
        <table><caption>Indexed code units by language/category</caption><thead><tr><th scope="col">Language/category</th><th scope="col">Code units</th></tr></thead>
          <tbody>{data.code_units_by_language.map(item => <tr key={item.language}><th scope="row">{item.language}</th><td>{item.count}</td></tr>)}</tbody></table>
        <table><caption>Indexed code units by kind</caption><thead><tr><th scope="col">Kind</th><th scope="col">Code units</th></tr></thead>
          <tbody>{data.code_units_by_kind.map(item => <tr key={item.kind}><th scope="row">{item.kind}</th><td>{item.count}</td></tr>)}</tbody></table>
      </div>}
      <p className="detail">Counts describe indexed data, not every file or symbol in the repository. Named code units may repeat symbol names or name documentation sections. Documentation and config are content categories. This is a last-observed snapshot, not live monitoring.</p>
    </>}
  </section>;
}
