import { connection } from "next/server";
import { getBackendHealth } from "@/lib/api";

const messages = {
  connected: "Connected",
  unavailable: "Cannot reach the backend. Check that FastAPI is running.",
  configuration: "Backend configuration invalid.",
  response: "Unexpected backend response.",
  http: "Backend health request failed.",
};

export default async function Home() {
  await connection();
  const health = await getBackendHealth();
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
    </main>
  );
}
