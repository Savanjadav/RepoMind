import { connection } from "next/server";
import { getBackendHealth, getRepositoryIndexingSummaries, listRepositories } from "@/lib/api";
import RepositoryForm from "./repository-form";
import RepositoryIndexing from "./repository-indexing";

const messages = {
  connected: "Connected",
  unavailable: "Cannot reach the backend. Check that FastAPI is running.",
  configuration: "Backend configuration invalid.",
  response: "Unexpected backend response.",
  http: "Backend health request failed.",
};

export default async function Home() {
  await connection();
  const [health, repositories] = await Promise.all([getBackendHealth(), listRepositories()]);
  const summaries = repositories.ok && repositories.items.length ?
    await getRepositoryIndexingSummaries(repositories.items.map((repo) => repo.id)) : null;
  const initial = summaries?.ok ? summaries.items : null;
  return (
    <main className="shell">
      <header>
        <p className="eyebrow">CODEBASE INTELLIGENCE</p>
        <h1>RepoMind<span aria-hidden="true">.</span></h1>
        <p className="intro">AI-powered codebase intelligence.</p>
      </header>
      <section className="connection" aria-labelledby="connection-heading">
        <h2 id="connection-heading">Backend connection</h2>
        <p role="status" className={`status ${health.state}`}>
          <span className="indicator" aria-hidden="true" />
          {messages[health.state]}
        </p>
        <p className="detail">
          This checks FastAPI connectivity, not database, Redis or model readiness.
        </p>
        <form action="/" method="get">
          <button className="retry" type="submit">Check again <span aria-hidden="true">↗</span></button>
        </form>
      </section>
      <section className="connection" aria-labelledby="register-heading">
        <h2 id="register-heading">Add a repository</h2>
        <RepositoryForm />
      </section>
      <section className="connection" aria-labelledby="repositories-heading">
        <h2 id="repositories-heading">Latest registered repositories</h2>
        <p className="detail">Latest 100 registrations. Registration does not mean indexing is complete.</p>
        {!repositories.ok ? (
          <p role="status">Could not load repositories. Check the backend connection and try Check again.</p>
        ) : repositories.items.length === 0 ? (
          <p>No repositories registered yet. Add a GitHub URL above.</p>
        ) : (
          <RepositoryIndexing key={JSON.stringify([repositories.items, initial])} repositories={repositories.items} initial={initial} />
        )}
      </section>
    </main>
  );
}
