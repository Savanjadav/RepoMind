"use client";

import { useEffect, useRef, useState } from "react";

type Result = { answer: string; citations: unknown[] };
type State = { status: "idle" | "submitting" | "success" | "error"; repositoryId?: string; question?: string; name?: string; result?: Result; message?: string };
const errors: Record<string, string> = {
  invalid: "Question/request invalid.", not_found: "Repository unavailable. Refresh the repository list.",
  unavailable: "Local answer service unavailable.", timeout: "Answer request timed out.",
  provider: "Answer service failed.", network: "Could not reach the answer service.",
  response: "Answer service returned an invalid response.", failed: "Could not generate an answer.",
};

export default function RepositoryQA({ repositoryId, name, disabledReason }: {
  repositoryId: string; name: string; disabledReason: string;
}) {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ status: "idle" });
  const request = useRef<{ token: number; controller?: AbortController; timer?: ReturnType<typeof setTimeout> }>({ token: 0 });
  useEffect(() => {
    const current = request.current;
    return () => { current.token++; current.controller?.abort(); clearTimeout(current.timer); current.controller = undefined; };
  }, [repositoryId, disabledReason]);

  async function submit() {
    if (disabledReason || request.current.controller) return;
    // Match Python's whitespace definition without changing the submitted text.
    if (!/[^\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/u.test(question) ||
        Array.from(question).length > 2000 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/u.test(question)) {
      setState({ status: "error", message: "Enter a question of 1–2,000 characters without prohibited control characters." });
      return;
    }
    const token = ++request.current.token;
    const controller = new AbortController();
    request.current.controller = controller;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 190_000);
    request.current.timer = timer;
    const submitted = { repositoryId, question, name };
    setState({ status: "submitting", ...submitted });
    try {
      const response = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repository_id: repositoryId, q: question }), cache: "no-store", redirect: "error", signal: controller.signal });
      const body = await response.json();
      if (token !== request.current.token) return;
      if (!response.ok) {
        setState({ status: "error", ...submitted, message: errors[body?.error] ?? errors.failed });
      } else if (typeof body?.answer !== "string" || !Array.isArray(body.citations)) {
        setState({ status: "error", ...submitted, message: errors.response });
      } else setState({ status: "success", ...submitted, result: body });
    } catch {
      if (token !== request.current.token) return;
      setState({ status: "error", ...submitted, message: timedOut ? errors.timeout : errors.network });
    } finally {
      clearTimeout(timer);
      if (token === request.current.token) request.current.controller = undefined;
    }
  }

  return <section className="qa" aria-labelledby="qa-heading">
    <h3 id="qa-heading">Ask about {name}</h3>
    <p>Each question is independent. No conversation history is sent.</p>
    {disabledReason && <p role="status">{disabledReason}</p>}
    <form onSubmit={(event) => { event.preventDefault(); void submit(); }}>
      <label htmlFor="qa-question">Repository question</label>
      <textarea id="qa-question" value={question} onChange={(event) => setQuestion(event.target.value)} disabled={Boolean(disabledReason) || state.status === "submitting"} rows={4} />
      <button className="submit" disabled={Boolean(disabledReason) || state.status === "submitting"}>Ask question</button>
    </form>
    {!disabledReason && <>
      <p role="status" aria-live="polite">{state.status === "submitting" ? "Searching indexed code and generating an answer…" : state.status === "success" ? "Answer received." : state.message ?? ""}</p>
      {state.question !== undefined && <><h4>Submitted question — {state.name}</h4><p className="qa-text">{state.question}</p></>}
      {state.result && <><h4>Answer</h4><div className="qa-text">{state.result.answer}</div>
        {!state.result.answer.trim() && <p>No answer text was returned.</p>}
        <p>Sources returned: {state.result.citations.length}</p></>}
    </>}
  </section>;
}
