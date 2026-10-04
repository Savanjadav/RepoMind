"use client";

import { useActionState } from "react";
import { addRepository, type RegistrationState } from "./actions";

const initial: RegistrationState = { status: "idle", message: "", source: "" };

export default function RepositoryForm() {
  const [state, action, pending] = useActionState(addRepository, initial);
  return (
    <form action={action} className="repository-form">
      <label htmlFor="repository-source">Repository URL</label>
      <input
        key={`${state.status}:${state.source}:${state.message}`}
        id="repository-source" name="source" type="url" required maxLength={2048}
        pattern={"[Hh][Tt][Tt][Pp][Ss]://[Gg][Ii][Tt][Hh][Uu][Bb]\\.[Cc][Oo][Mm](?::443)?/.*"}
        placeholder="https://github.com/owner/repository"
        defaultValue={state.source} readOnly={pending}
        aria-invalid={state.invalid || undefined}
        aria-describedby="repository-help repository-feedback"
      />
      <p id="repository-help" className="detail">
        Public GitHub HTTPS repository. Registration saves metadata only; it does not verify access or start indexing.
      </p>
      <button className="submit" type="submit" disabled={pending}>
        {pending ? "Adding repository…" : "Add repository"}
      </button>
      <p id="repository-feedback" role="status" aria-live="polite">{state.message}</p>
    </form>
  );
}
