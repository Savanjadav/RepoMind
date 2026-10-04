"use server";

import { revalidatePath } from "next/cache";
import { registerRepository } from "@/lib/api";

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
