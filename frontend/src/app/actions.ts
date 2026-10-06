"use server";

import { revalidatePath } from "next/cache";
import { registerRepository, startRepositoryIndexing } from "@/lib/api";

export type { IndexingSummary, Repository } from "@/lib/api";

export type StartIndexingState = {
  status: "accepted" | "conflict" | "uncertain" | "error";
  message: string;
  jobId?: string;
};

export async function startIndexing(form: FormData): Promise<StartIndexingState> {
  const entries = [...form.entries()];
  const id = form.get("repository_id");
  if (entries.length !== 1 || entries[0][0] !== "repository_id" || typeof id !== "string" ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu.test(id)) {
    return { status: "error", message: "Invalid repository identifier." };
  }
  const result = await startRepositoryIndexing(id);
  if (result.ok) return { status: "accepted", jobId: result.job_id, message: "Indexing request accepted." };
  if (result.error === "conflict") return { status: "conflict", message: "Indexing is already active or the repository is busy." };
  if (result.error === "uncertain") return {
    status: "uncertain", message: "The indexing request outcome is uncertain. Refreshing status before another start.",
  };
  const messages = {
    invalid: "This repository cannot be indexed. Check its source.",
    not_found: "Repository no longer exists. Reload the repository list.",
    configuration: "Indexing service configuration is invalid.",
    unavailable: "Indexing service is temporarily unavailable.",
    response: "Unexpected indexing service response.",
  };
  return { status: "error", message: messages[result.error] };
}

export type RegistrationState = {
  status: "idle" | "success" | "error";
  message: string;
  source: string;
  invalid?: boolean;
};

export async function addRepository(
  _previous: RegistrationState, form: FormData,
): Promise<RegistrationState> {
  const entries = form.getAll("source");
  const value = entries[0];
  const source = typeof value === "string" ? value.slice(0, 2048) : "";
  let valid = entries.length === 1 && typeof value === "string" &&
    value.length > 0 && value.length <= 2048 && value === value.trim() &&
    !/[\u0000-\u001f\u007f]/u.test(value);
  try {
    const parsed = new URL(source);
    valid = valid && parsed.protocol === "https:" && parsed.hostname === "github.com";
  } catch {
    valid = false;
  }
  if (!valid) return {
    status: "error", source, invalid: true,
    message: "Enter a public GitHub HTTPS repository URL.",
  };
  const result = await registerRepository(source);
  if (!result.ok) {
    const messages = {
      invalid: "Enter a supported public GitHub repository URL, without credentials, query or fragment.",
      duplicate: "This repository is already registered.",
      configuration: "Repository service configuration is invalid.",
      unavailable: "Repository service is temporarily unavailable.",
      uncertain: "The registration result is uncertain. Check the repository list before submitting again.",
      response: "Unexpected repository service response.",
    };
    return { status: "error", message: messages[result.error], source, invalid: result.error === "invalid" };
  }
  revalidatePath("/");
  return { status: "success", message: "Repository registered. Indexing has not started.", source: "" };
}
