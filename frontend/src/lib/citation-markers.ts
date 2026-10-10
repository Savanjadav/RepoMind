export type AnswerPart = { text: string; evidenceId?: number };

/** Tokenize plain text; only IDs supplied by structured citations are controls. */
export function citationMarkers(answer: string, evidenceIds: readonly number[]): AnswerPart[] {
  const ids = new Map(evidenceIds.map(id => [String(id), id]));
  const parts: AnswerPart[] = [];
  let offset = 0;
  for (const match of answer.matchAll(/\[Evidence ([1-9][0-9]*)\]/gu)) {
    const id = ids.get(match[1]);
    if (id === undefined) continue;
    if (match.index > offset) parts.push({ text: answer.slice(offset, match.index) });
    parts.push({ text: match[0], evidenceId: id });
    offset = match.index + match[0].length;
  }
  if (offset < answer.length) parts.push({ text: answer.slice(offset) });
  return parts;
}
